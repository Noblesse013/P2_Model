"""
One-time embedding extraction script.

Runs both frozen pretrained encoders on all available data and saves
the resulting embedding vectors to disk as .npy files.

Run this ONCE before training the fusion model:
    python extract_embeddings.py
    python extract_embeddings.py --config configs/config.yaml

Output (saved to embeddings/):
    obd2_train.npy,  obd2_train_labels.npy
    obd2_val.npy,    obd2_val_labels.npy
    obd2_test.npy,   obd2_test_labels.npy
    audio_train.npy, audio_train_labels.npy
    audio_val.npy,   audio_val_labels.npy
    audio_test.npy,  audio_test_labels.npy
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.encoders import OBD2Encoder, AudioEncoder


def extract_obd2_embeddings(cfg, device, out_dir):
    obd2_cfg  = cfg["obd2"]
    model_dir = Path(obd2_cfg["model_dir"])
    processed = model_dir / "data" / "processed"

    print("\n[OBD2] Loading pretrained encoder ...")
    encoder = OBD2Encoder(
        model_dir  = str(model_dir),
        checkpoint = obd2_cfg["checkpoint"],
        config     = obd2_cfg["config"],
        device     = device,
    )
    encoder.eval()

    batch_size = 512

    for split in ("train", "val", "test"):
        x_path = processed / f"X_{split}.npy"
        y_path = processed / f"y_{split}.npy"

        if not x_path.exists():
            print(f"  [OBD2] {x_path} not found - skipping {split}")
            continue

        X = np.load(x_path)
        y = np.load(y_path)
        N = len(X)

        print(f"  [OBD2] {split}: {N:,} windows - extracting embeddings ...")
        embs = []
        with torch.no_grad():
            for start in tqdm(range(0, N, batch_size), desc=f"  OBD2 {split}", leave=False):
                batch = torch.tensor(
                    X[start:start + batch_size], dtype=torch.float32
                ).to(device)
                embs.append(encoder(batch).cpu().numpy())

        embs = np.concatenate(embs, axis=0)
        np.save(out_dir / f"obd2_{split}.npy",        embs)
        np.save(out_dir / f"obd2_{split}_labels.npy", y.astype(np.int64))
        print(f"  [OBD2] {split}: saved {embs.shape}, labels {y.shape}")


def extract_audio_embeddings(cfg, device, out_dir):
    audio_cfg = cfg["audio"]
    model_dir = Path(audio_cfg["model_dir"])
    data_dir  = model_dir / "data"

    print("\n[Audio] Loading pretrained encoder ...")
    encoder = AudioEncoder(
        model_dir  = str(model_dir),
        checkpoint = audio_cfg["checkpoint"],
        config     = audio_cfg["config"],
        device     = device,
    )
    encoder.eval()

    import yaml as _yaml
    from omegaconf import OmegaConf
    with open(model_dir / audio_cfg["config"]) as f:
        raw = _yaml.safe_load(f)
    acfg = OmegaConf.create(raw)

    audio_root = str(model_dir)
    _src_save = {k: sys.modules.pop(k)
                 for k in list(sys.modules)
                 if k == 'src' or k.startswith('src.')}
    if audio_root in sys.path:
        sys.path.remove(audio_root)
    sys.path.insert(0, audio_root)
    try:
        from src.data.preprocessing import AudioPreprocessor
        from src.features.extractor import FeatureExtractor
    finally:
        for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
            del sys.modules[k]
        sys.modules.update(_src_save)

    preprocessor = AudioPreprocessor(
        sample_rate      = acfg.data.sample_rate,
        segment_duration = acfg.data.segment_duration,
        hop_duration     = acfg.data.hop_duration,
    )
    feature_extractor = FeatureExtractor(
        sample_rate = acfg.data.sample_rate,
        n_fft       = acfg.features.n_fft,
        hop_length  = acfg.features.hop_length,
        n_mels      = acfg.features.n_mels,
        f_min       = acfg.features.f_min,
        f_max       = acfg.features.f_max,
        normalize   = acfg.features.normalize,
    )

    grade_map = {"normal": 0, "warning": 1, "fault": 2, "critical": 3}
    EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    all_files = []
    for folder, grade in grade_map.items():
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"  [Audio] {folder_path} not found - skipping")
            continue
        files = [f for f in folder_path.rglob("*") if f.suffix.lower() in EXTS]
        all_files.extend((f, grade) for f in files)
        print(f"  [Audio] {folder}: {len(files):,} files")

    # Stratified 70/15/15 split per grade
    rng = np.random.default_rng(42)
    split_files = {"train": [], "val": [], "test": []}

    for grade in range(4):
        grade_files = [(f, g) for f, g in all_files if g == grade]
        rng.shuffle(grade_files)
        N      = len(grade_files)
        n_val  = max(1, int(N * 0.15))
        n_test = max(1, int(N * 0.15))
        split_files["test"]  += grade_files[:n_test]
        split_files["val"]   += grade_files[n_test:n_test + n_val]
        split_files["train"] += grade_files[n_test + n_val:]

    for split, pairs in split_files.items():
        print(f"\n  [Audio] {split}: {len(pairs):,} files - extracting ...")
        embs_list, labels_list = [], []

        for file_path, grade in tqdm(pairs, desc=f"  Audio {split}", leave=False):
            try:
                segments = preprocessor.process(str(file_path))
            except Exception:
                continue

            for seg in segments:
                spec = feature_extractor.extract(seg).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = encoder(spec).squeeze(0).cpu().numpy()
                embs_list.append(emb)
                labels_list.append(grade)

        if not embs_list:
            print(f"  [Audio] {split}: no segments extracted - skipping")
            continue

        embs   = np.stack(embs_list).astype(np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        np.save(out_dir / f"audio_{split}.npy",        embs)
        np.save(out_dir / f"audio_{split}_labels.npy", labels)
        print(f"  [Audio] {split}: saved {embs.shape}, labels {labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract encoder embeddings (one-time)")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--out",        default="embeddings")
    parser.add_argument("--obd2_only",  action="store_true")
    parser.add_argument("--audio_only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Output : {out_dir.resolve()}")

    if not args.audio_only:
        extract_obd2_embeddings(cfg, device, out_dir)

    if not args.obd2_only:
        extract_audio_embeddings(cfg, device, out_dir)

    print("\nEmbedding extraction complete.")
    print(f"Files written to: {out_dir.resolve()}")
    print("Next step: python train_fusion.py")


if __name__ == "__main__":
    main()
