# Implementation Roadmap — Multimodal Engine Health Grading
**Author:** Tareq Sujat — BRAC University  
**Date:** 2026-05-28

---

## Project Structure (to be built here)

```
E:\Thesis\Multimodal Engine Health grading using obd2 and audio\
├── MULTIMODAL_ARCHITECTURE_ANALYSIS.md     ← Architecture research (done)
├── IMPLEMENTATION_ROADMAP.md               ← This file
│
├── configs/
│   ├── config_late_fusion.yaml
│   ├── config_olrmf.yaml                   ← Arch 3: LMF
│   ├── config_sefmn.yaml                   ← Arch 2: MMTM
│   └── config_cmoft.yaml                   ← Arch 1: Cross-Modal Transformer
│
├── src/
│   ├── data/
│   │   ├── paired_dataset.py               ← Dataset that yields (obd2_window, audio_spec, label)
│   │   └── sync_utils.py                   ← OBD2 ↔ Audio temporal co-registration helpers
│   │
│   ├── encoders/
│   │   ├── obd2_encoder.py                 ← Loads CNN+BiGRU from OBD2 model (read-only import)
│   │   └── audio_encoder.py                ← Loads MSDA-Net from audio model (read-only import)
│   │
│   ├── fusion/
│   │   ├── late_fusion.py                  ← Temperature-scaled logit combiner
│   │   ├── lmf.py                          ← Low-Rank Multimodal Fusion (Arch 3)
│   │   ├── mmtm.py                         ← Multimodal Transfer Module (Arch 2)
│   │   └── cross_modal_transformer.py      ← MulT-style CMOFT (Arch 1)
│   │
│   ├── models/
│   │   ├── late_fusion_model.py
│   │   ├── olrmf_model.py                  ← Full Arch 3 model
│   │   ├── sefmn_model.py                  ← Full Arch 2 model
│   │   └── cmoft_model.py                  ← Full Arch 1 model
│   │
│   ├── losses/
│   │   ├── ordinal_losses.py               ← CORN, WeightedKappa, THOR, OT loss
│   │   └── multimodal_loss.py              ← Combined loss with auxiliary heads
│   │
│   ├── training/
│   │   ├── trainer.py                      ← Generic training loop (phased for CMOFT)
│   │   └── phased_trainer.py               ← 3-phase training for cross-modal attn
│   │
│   └── evaluation/
│       ├── metrics.py                       ← Kappa, MCC, per-class F1, modal disagreement
│       ├── attention_viz.py                 ← Cross-attention heatmap visualization
│       └── modality_contribution.py        ← SHAP on fused embeddings
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           ← Explore paired OBD2+audio dataset
│   ├── 02_late_fusion_baseline.ipynb
│   ├── 03_arch3_olrmf.ipynb
│   ├── 04_arch2_sefmn.ipynb
│   └── 05_arch1_cmoft.ipynb
│
├── experiments/
│   ├── results/                            ← JSON metrics, model checkpoints
│   ├── figures/                            ← Attention maps, confusion matrices
│   └── mlruns/                             ← MLflow tracking
│
├── train.py                                ← Entry point
├── evaluate.py                             ← Standalone evaluation
└── requirements.txt
```

---

## Phase-by-Phase Implementation Plan

### Phase 0: Data Pipeline (1–3 days)

**Goal:** Create a `PairedEngineDataset` that yields `(obd2_window, audio_spectrogram, label)` triplets.

**Key decisions:**
- Use the hierarchical temporal strategy (Option C from analysis): keep OBD2 window (30 steps) and audio (3-sec segment) as separate temporal scales
- Pair by health-grade label (if recorded simultaneously: by timestamp; if separate datasets: by label stratification)
- Apply existing scalers and preprocessing from both unimodal pipelines (import, do not copy)

**Files to create:**
- `src/data/paired_dataset.py`
- `src/data/sync_utils.py`

---

### Phase 1: Late Fusion Baseline (1–2 days)

**Goal:** Establish a ceiling for "no interaction" fusion. Load both pretrained models, freeze weights, combine logits.

```python
# Architecture sketch
class LateFusionModel(nn.Module):
    def __init__(self, obd2_model, audio_model):
        self.obd2_model = obd2_model   # frozen
        self.audio_model = audio_model  # frozen
        self.tau_obd2 = nn.Parameter(torch.ones(1))   # learned temperature
        self.tau_audio = nn.Parameter(torch.ones(1))

    def forward(self, x_obd2, x_audio):
        logits_obd2 = self.obd2_model(x_obd2)
        logits_audio = self.audio_model(x_audio)
        return (logits_obd2 / self.tau_obd2 + logits_audio / self.tau_audio) / 2
```

**Expected result:** ~94–96% accuracy. This is the "free lunch" from combining two strong models.

---

### Phase 2: OLRMF — Ordinal Low-Rank Fusion (2–3 days)

**Goal:** Add multiplicative cross-modal interactions via LMF. Establish that statistical fusion helps over late fusion.

```python
# LMF core equation
class LMF(nn.Module):
    def __init__(self, d1, d2, d_out, rank=8):
        # Factor matrices
        self.W1 = nn.Parameter(torch.randn(rank, d1+1, d_out))
        self.W2 = nn.Parameter(torch.randn(rank, d2+1, d_out))

    def forward(self, h1, h2):
        h1_aug = torch.cat([h1, torch.ones(h1.size(0),1)], dim=1)  # (B, d1+1)
        h2_aug = torch.cat([h2, torch.ones(h2.size(0),1)], dim=1)  # (B, d2+1)
        # Compute rank-r outer product
        fused = torch.einsum('bi,rio->bro', h1_aug, self.W1) * \
                torch.einsum('bj,rjo->bro', h2_aug, self.W2)      # (B, rank, d_out)
        return fused.sum(dim=1)                                     # (B, d_out)
```

**Loss:**
```
L = CE(z_fused, y) + 0.5·WeightedKappa(z_fused, y) + 0.3·(CE(z_obd2, y) + CE(z_audio, y))
```

---

### Phase 3: SE-MFN — MMTM Fusion (3–5 days)

**Goal:** Add channel-level cross-modal recalibration at intermediate feature levels.

```python
class MMTM(nn.Module):
    def __init__(self, c_obd2, c_audio, reduction=4):
        d = max(c_obd2 + c_audio, 8) // reduction
        self.fc = nn.Sequential(nn.Linear(c_obd2 + c_audio, d), nn.ReLU())
        self.fc_obd2 = nn.Linear(d, c_obd2)
        self.fc_audio = nn.Linear(d, c_audio)

    def forward(self, feat_obd2, feat_audio):
        # feat_obd2: (B, c_obd2, T_obd2)
        # feat_audio: (B, c_audio, F, T_audio)
        gap_obd2 = feat_obd2.mean(-1)           # (B, c_obd2)
        gap_audio = feat_audio.mean((-2,-1))     # (B, c_audio)
        z = self.fc(torch.cat([gap_obd2, gap_audio], dim=1))
        gate_obd2 = torch.sigmoid(self.fc_obd2(z)).unsqueeze(-1)
        gate_audio = torch.sigmoid(self.fc_audio(z)).unsqueeze(-1).unsqueeze(-1)
        return feat_obd2 * gate_obd2, feat_audio * gate_audio
```

**Insertion points:**
- L1: After OBD2 Conv1D block 1 + After MSDA-Net Stage 1
- L2: After OBD2 BiGRU + After MSDA-Net Frequency Attention

---

### Phase 4: CMOFT — Cross-Modal Ordinal Fusion Transformer (1–2 weeks)

**Goal:** Full bidirectional cross-modal attention. The primary thesis architecture.

```python
class CrossModalAttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        self.cross_attn_a2b = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_b2a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def forward(self, seq_a, seq_b):
        # seq_a: (B, T_a, d) — OBD2 tokens
        # seq_b: (B, T_b, d) — Audio tokens

        # Directional cross-attention
        cross_a, attn_a = self.cross_attn_a2b(seq_a, seq_b, seq_b)  # OBD2 queries audio
        cross_b, attn_b = self.cross_attn_b2a(seq_b, seq_a, seq_a)  # Audio queries OBD2

        seq_a = self.norm_a(seq_a + cross_a)
        seq_b = self.norm_b(seq_b + cross_b)

        # Self-attention within each stream
        seq_a, _ = self.self_attn_a(seq_a, seq_a, seq_a)
        seq_b, _ = self.self_attn_b(seq_b, seq_b, seq_b)

        return seq_a, seq_b, attn_a, attn_b  # return attn for visualization
```

**Training protocol:**
```
Phase 1 (5–10 epochs):  lr=1e-3, freeze cross-attn, train encoders only
Phase 2 (30–50 epochs): lr=1e-4, unfreeze all, full joint training
Phase 3 (10 epochs):    lr=5e-5, freeze CNN encoders, fine-tune attention + head
```

---

### Phase 5: Ordinal Loss Ablation (2–3 days)

Run Experiment 6 from the analysis on the best fusion arch:

| Run | Loss | Expected kappa |
|---|---|---|
| A | CE only | baseline |
| B | CE + WeightedKappa (α=0.5) | +0.02 |
| C | CE + CORN | +0.03 |
| D | CE + CORN + THOR + WeightedKappa | best |

---

### Phase 6: Explainability & Thesis Figures (3–5 days)

1. Cross-attention heatmaps (OBD2 timestep × audio frame)
2. MMTM SE gate weights (which OBD2 channels get boosted by audio)
3. SHAP on fused 256-dim embeddings (which modality contributes per grade)
4. Modal disagreement examples (transitional health states)
5. All confusion matrices, ROC curves, precision-recall curves

---

## Dependencies (requirements.txt additions)

```
# Existing from both unimodal projects (inherit, don't reinstall):
#   torch, torchaudio, torchvision, librosa, scikit-learn, numpy, pandas, mlflow

# New for multimodal:
einops>=0.7.0          # tensor operations for attention
timm>=0.9.12           # pretrained backbone utilities
x-transformers>=1.27   # optional: clean transformer implementations
```

---

## Reference Implementations

| Component | Reference |
|---|---|
| MulT (Cross-Modal Transformer) | github.com/yaohungt/Multimodal-Transformer |
| MMTM | github.com/haamoon/mmtm |
| LMF | github.com/Justin1904/Low-rank-Multimodal-Fusion |
| CORN ordinal head | dlordinal Python package |
| Weighted Kappa loss | torchmetrics.CohenKappa or custom |

---

## Key Invariant

**Never modify files in:**
- `E:\Thesis\Car_Engine_audio_model(final)\`
- `E:\Thesis\Car engine health ob2 model\`

These are read-only references. Import their models via:
```python
import sys
sys.path.insert(0, r"E:\Thesis\Car_Engine_audio_model(final)\src")
sys.path.insert(0, r"E:\Thesis\Car engine health ob2 model\src")
from models.msda_net import MSDANet
from model import CNNBiGRUAttention
```
