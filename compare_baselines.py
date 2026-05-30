"""
Unimodal vs Fusion ablation.

Runs OBD2-only and Audio-only predictions from the pre-extracted embeddings
using each branch's own classifier head, then compares to the fusion model.
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef

GRADE_LABELS = ["Normal", "Warning", "Fault", "Critical"]


# ── OBD2-only ────────────────────────────────────────────────────────────────
def obd2_only_metrics(cfg, device):
    from pathlib import Path as P
    obd2_dir = P(cfg["obd2"]["model_dir"])
    src = str(obd2_dir / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    import yaml as _y
    with open(obd2_dir / cfg["obd2"]["config"]) as f:
        ocfg = _y.safe_load(f)

    from model import build_model
    ckpt = torch.load(obd2_dir / cfg["obd2"]["checkpoint"], map_location=device, weights_only=False)
    full_model = build_model(ocfg)
    full_model.load_state_dict(ckpt["model_state_dict"])
    full_model.to(device).eval()

    h = torch.tensor(np.load("embeddings/obd2_test.npy"), dtype=torch.float32).to(device)
    y = np.load("embeddings/obd2_test_labels.npy")

    with torch.no_grad():
        logits = full_model.classifier(h)
        preds  = logits.argmax(dim=1).cpu().numpy()

    return _metrics(y, preds, "OBD2-only", len(y))


# ── Audio-only ───────────────────────────────────────────────────────────────
def audio_only_metrics(cfg, device):
    from pathlib import Path as P
    audio_dir = P(cfg["audio"]["model_dir"])
    audio_root = str(audio_dir)

    import yaml as _y
    from omegaconf import OmegaConf
    with open(audio_dir / cfg["audio"]["config"]) as f:
        raw = _y.safe_load(f)
    acfg = OmegaConf.create(raw)

    _src_save = {k: sys.modules.pop(k)
                 for k in list(sys.modules) if k == 'src' or k.startswith('src.')}
    if audio_root in sys.path:
        sys.path.remove(audio_root)
    sys.path.insert(0, audio_root)
    try:
        from src.models.factory import build_model
        from src.models.ordinal_head import OrdinalHead
        ckpt = torch.load(audio_dir / cfg["audio"]["checkpoint"], map_location=device, weights_only=False)
        full_model = build_model(acfg)
        full_model.load_state_dict(ckpt["model_state_dict"])
        full_model.to(device).eval()
    finally:
        for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
            del sys.modules[k]
        sys.modules.update(_src_save)

    h = torch.tensor(np.load("embeddings/audio_test.npy"), dtype=torch.float32).to(device)
    y = np.load("embeddings/audio_test_labels.npy")

    with torch.no_grad():
        logits = full_model.head(h)
        preds  = OrdinalHead.decode(logits).cpu().numpy()

    return _metrics(y, preds, "Audio-only", len(y))


# ── helpers ───────────────────────────────────────────────────────────────────
def _metrics(y, preds, name, n):
    return {
        "model":    name,
        "n":        n,
        "accuracy": round(accuracy_score(y, preds), 4),
        "macro_f1": round(f1_score(y, preds, average="macro",    zero_division=0), 4),
        "kappa":    round(cohen_kappa_score(y, preds, weights="quadratic"), 4),
        "mcc":      round(matthews_corrcoef(y, preds), 4),
    }


def print_table(rows):
    header = f"{'Model':<14}  {'N':>6}  {'Accuracy':>9}  {'Macro F1':>9}  {'QW Kappa':>9}  {'MCC':>7}"
    sep    = "-" * len(header)
    print(); print(sep); print(header); print(sep)
    for r in rows:
        print(f"  {r['model']:<12}  {r['n']:>6,}  {r['accuracy']:>9.4f}  "
              f"{r['macro_f1']:>9.4f}  {r['kappa']:>9.4f}  {r['mcc']:>7.4f}")
    print(sep); print()


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing OBD2-only metrics ...")
    r_obd2  = obd2_only_metrics(cfg, device)

    print("Computing Audio-only metrics ...")
    r_audio = audio_only_metrics(cfg, device)

    # Fusion results already computed
    fm = json.load(open("experiments/fusion_test_metrics.json"))
    r_fusion = {
        "model":    "Fusion (MLP)",
        "n":        fm["n_samples"],
        "accuracy": fm["accuracy"],
        "macro_f1": fm["macro_f1"],
        "kappa":    fm["quadratic_kappa"],
        "mcc":      fm["mcc"],
    }

    rows = [r_obd2, r_audio, r_fusion]
    print_table(rows)

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/baseline_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved -> experiments/baseline_comparison.json")


if __name__ == "__main__":
    main()
