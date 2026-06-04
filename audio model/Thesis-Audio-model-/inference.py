"""
Single-file inference for deployed engine health grading.

Usage:
    python inference.py --audio engine_sample.wav --checkpoint experiments/results/msda_net_ordinal_ce_best.pt
    python inference.py --audio /path/to/dir/ --checkpoint ...   # batch mode
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.data.preprocessing import AudioPreprocessor
from src.features.extractor import FeatureExtractor
from src.models.factory import build_model
from src.models.ordinal_head import OrdinalHead


HEALTH_GRADE_INFO = {
    0: {"label": "Normal",   "emoji": "✅", "action": "No action required."},
    1: {"label": "Warning",  "emoji": "⚠️",  "action": "Schedule preventive maintenance."},
    2: {"label": "Fault",    "emoji": "🔴", "action": "Immediate maintenance required."},
    3: {"label": "Critical", "emoji": "🚨", "action": "Stop engine immediately."},
}


def load_model(ckpt_path: str, cfg, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_file(
    audio_path: str,
    model: torch.nn.Module,
    preprocessor: AudioPreprocessor,
    feature_extractor: FeatureExtractor,
    device: torch.device,
    ordinal: bool = True,
) -> dict:
    """
    Predict engine health grade for a single audio file.
    Uses majority voting over all extracted segments.
    """
    from src.models.ordinal_head import OrdinalHead
    import torch.nn.functional as F

    segments = preprocessor.process(audio_path)
    all_probs = []

    for seg in segments:
        spec = feature_extractor.extract(seg).unsqueeze(0).to(device)
        logits = model(spec)
        if ordinal:
            probs = OrdinalHead.to_class_probs(logits)
        else:
            probs = F.softmax(logits, dim=1)
        all_probs.append(probs.squeeze(0).cpu().numpy())

    avg_probs = np.mean(all_probs, axis=0)
    grade = int(np.argmax(avg_probs))
    confidence = float(avg_probs[grade])

    return {
        "file": audio_path,
        "grade": grade,
        "grade_label": HEALTH_GRADE_INFO[grade]["label"],
        "confidence": confidence,
        "probabilities": {HEALTH_GRADE_INFO[i]["label"]: float(avg_probs[i])
                         for i in range(len(avg_probs))},
        "action": HEALTH_GRADE_INFO[grade]["action"],
        "n_segments": len(segments),
    }


def format_result(result: dict) -> str:
    info = HEALTH_GRADE_INFO[result["grade"]]
    lines = [
        "",
        "━" * 55,
        f"  FILE:        {Path(result['file']).name}",
        f"  GRADE:       {info['emoji']}  {result['grade_label']} (Grade {result['grade']})",
        f"  CONFIDENCE:  {result['confidence']:.1%}",
        "  PROBABILITIES:",
    ]
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        lines.append(f"    {label:<10} {bar:<20} {prob:.1%}")
    lines += [
        f"  ACTION:      {result['action']}",
        "━" * 55,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Engine health grade inference")
    parser.add_argument("--audio", required=True, help="Path to .wav file or directory")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    if args.model:
        cfg.model.name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device)

    preprocessor = AudioPreprocessor(
        sample_rate=cfg.data.sample_rate,
        segment_duration=cfg.data.segment_duration,
        hop_duration=cfg.data.hop_duration,
    )
    feature_extractor = FeatureExtractor(
        sample_rate=cfg.data.sample_rate,
        n_fft=cfg.features.n_fft,
        hop_length=cfg.features.hop_length,
        n_mels=cfg.features.n_mels,
        f_min=cfg.features.f_min,
        f_max=cfg.features.f_max,
        normalize=cfg.features.normalize,
    )

    audio_path = Path(args.audio)
    audio_files = (
        list(audio_path.rglob("*.wav")) + list(audio_path.rglob("*.mp3"))
        if audio_path.is_dir()
        else [audio_path]
    )

    all_results = []
    for fp in audio_files:
        result = predict_file(str(fp), model, preprocessor, feature_extractor, device, cfg.model.ordinal)
        print(format_result(result))
        all_results.append(result)

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
