"""
Extract audio embeddings using the EXACT same split the audio model
was trained with — reproducing stratified_split(random.Random(42)).

The original embedding extractor used np.random.default_rng(42) which
produces a different shuffle than the audio model's random.Random(42).
This means the fusion pipeline's audio "test" set overlaps with the
audio model's training set — inflating the audio-only accuracy.

This script fixes that by:
  1. Calling the audio model's own scan_folder + stratified_split
     (same code path, same RNG, same seed) to get the exact test files.
  2. Extracting encoder embeddings + ordinal-head probabilities for
     only those files.
  3. Saving to embeddings/audio_test_original.npy  (and _labels, _probs).

Usage:
    python extract_audio_clean_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

AUDIO_MODEL_DIR = Path("E:/Thesis/Car_Engine_audio_model(final)")
EMB_DIR         = Path("embeddings")
OUT_EMB         = EMB_DIR / "audio_test_original.npy"
OUT_LABELS      = EMB_DIR / "audio_test_original_labels.npy"
OUT_PROBS       = EMB_DIR / "audio_test_original_probs.npy"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load audio config ─────────────────────────────────────────────────────
    with open(AUDIO_MODEL_DIR / "configs" / "config.yaml") as f:
        raw = yaml.safe_load(f)
    cfg = OmegaConf.create(raw)

    # ── Step 1: Reconstruct the EXACT split the audio model used ──────────────
    # Import audio model's own dataset utilities — same code, same RNG
    audio_root = str(AUDIO_MODEL_DIR)
    _saved = {k: sys.modules.pop(k)
              for k in list(sys.modules)
              if k == "src" or k.startswith("src.")}
    if audio_root in sys.path:
        sys.path.remove(audio_root)
    sys.path.insert(0, audio_root)

    try:
        from src.data.dataset import scan_folder, stratified_split
        from src.data.preprocessing import AudioPreprocessor
        from src.features.extractor import FeatureExtractor
        from src.models.factory import build_model as _audio_build

        # Exact same call as train.py → build_dataloaders
        all_samples = scan_folder(str(AUDIO_MODEL_DIR / "data"))
        _, _, test_samples = stratified_split(
            all_samples,
            val_ratio   = cfg.data.val_split,    # 0.15
            test_ratio  = cfg.data.test_split,   # 0.15
            seed        = cfg.project.seed,      # 42  →  random.Random(42)
        )

        print(f"\nAudio model original test split: {len(test_samples):,} files")
        grade_counts = {}
        for _, g in test_samples:
            grade_counts[g] = grade_counts.get(g, 0) + 1
        for g, cnt in sorted(grade_counts.items()):
            print(f"  Grade {g}: {cnt:,} files")

        # Preprocessing + feature extraction (same params as audio model training)
        preprocessor = AudioPreprocessor(
            sample_rate      = cfg.data.sample_rate,
            segment_duration = cfg.data.segment_duration,
            hop_duration     = cfg.data.hop_duration,
        )
        feature_extractor = FeatureExtractor(
            sample_rate = cfg.data.sample_rate,
            n_fft       = cfg.features.n_fft,
            hop_length  = cfg.features.hop_length,
            n_mels      = cfg.features.n_mels,
            f_min       = cfg.features.f_min,
            f_max       = cfg.features.f_max,
            normalize   = cfg.features.normalize,
        )

        # Load full audio model (encoder + ordinal head)
        ckpt_path = AUDIO_MODEL_DIR / "experiments/results/msda_net_ordinal_ce_best.pt"
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        model     = _audio_build(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(_saved)
        if audio_root in sys.path:
            sys.path.remove(audio_root)

    # Re-import local OrdinalHead AFTER restoring sys.path
    from src.models.fusion_model import OrdinalHead as LocalOH

    # ── Step 2: Extract embeddings + probs for original test files ────────────
    print(f"\nExtracting embeddings from {len(test_samples):,} original test files ...")

    emb_list, prob_list, label_list = [], [], []
    skipped = 0

    for file_path, grade in tqdm(test_samples, desc="Audio test (original split)"):
        try:
            segments = preprocessor.process(file_path)
        except Exception:
            skipped += 1
            continue

        for seg in segments:
            spec = feature_extractor.extract(seg).unsqueeze(0).to(device)
            with torch.no_grad():
                # Run through full model to get both embedding and probs
                x = model.stem(spec)
                x = model.stage1(x)
                x = model.stage2(x)
                x = model.stage3(x)
                x = model.freq_attn(x)
                x = model.temporal_attn(x)
                avg   = model.gap(x).flatten(1)
                mx    = model.gmp(x).flatten(1)
                feats = torch.cat([avg, mx], dim=1)
                emb   = model.classifier(feats)          # (1, 512) — encoder output
                logits = model.head(emb)                 # (1, 3)  — ordinal logits
                probs  = LocalOH.to_class_probs(logits)  # (1, 4)

            emb_list.append(emb.squeeze(0).cpu().numpy())
            prob_list.append(probs.squeeze(0).cpu().numpy())
            label_list.append(grade)

    if skipped:
        print(f"  Skipped {skipped} files (read errors)")

    embs   = np.stack(emb_list).astype(np.float32)
    probs  = np.stack(prob_list).astype(np.float32)
    labels = np.array(label_list, dtype=np.int64)

    print(f"\nExtracted: {embs.shape[0]:,} segments from {len(test_samples):,} files")
    for g in range(4):
        print(f"  Grade {g}: {int((labels == g).sum()):,} segments")

    # ── Step 3: Save ──────────────────────────────────────────────────────────
    EMB_DIR.mkdir(exist_ok=True)
    np.save(OUT_EMB,    embs)
    np.save(OUT_LABELS, labels)
    np.save(OUT_PROBS,  probs)

    print(f"\nSaved:")
    print(f"  {OUT_EMB}    {embs.shape}")
    print(f"  {OUT_LABELS} {labels.shape}")
    print(f"  {OUT_PROBS}  {probs.shape}")
    print("\nNext: python evaluate_late_fusion.py --audio_test_key original")


if __name__ == "__main__":
    main()
