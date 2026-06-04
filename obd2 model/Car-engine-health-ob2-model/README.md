# Car Engine Health Grading — OBD-2 (4-Class)

CNN + Bidirectional GRU + Attention classifier that grades engine health into:
**Normal → Warning → Faulty → Critical**

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### Step 0 — Get Data
Download one (or both) of these Kaggle datasets and place CSVs in `data/raw/`:
- https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset
- https://www.kaggle.com/datasets/cephasax/obdii-ds3

### Step 1 — EDA
Open `notebooks/eda.ipynb` and run all cells. Use the threshold suggestions at
the bottom to update `config.yaml` → `data.label_thresholds`.

Also update `config.yaml` → `data.column_map` to match your CSV headers exactly.

### Step 2 — Run Full Pipeline
```bash
cd src
python main.py
```

Or run steps individually:
```bash
python src/preprocess.py    # label + window + split + save
python src/train.py         # train CNN+BiGRU+Attention
python src/evaluate.py      # confusion matrix, F1, ROC-AUC
python src/explain.py       # SHAP feature importance
```

## Output Files (`data/processed/`)

| File | Description |
|---|---|
| `X_train/val/test.npy` | Windowed feature tensors |
| `y_train/val/test.npy` | Class labels (0–3) |
| `scaler.pkl` | Fitted RobustScaler |
| `meta.pkl` | Feature names, window size |
| `best_model.pt` | Best checkpoint |
| `confusion_matrix.png` | Per-class accuracy heatmap |
| `roc_curves.png` | Per-class ROC-AUC curves |
| `training_history.png` | Loss and accuracy over epochs |
| `shap_feature_importance.png` | Global SHAP importance |
| `shap_per_class.png` | Per-class SHAP importance |

## Model Architecture

```
Input (batch, timesteps, features)
  → Conv1D(64, k=5) + BN + ReLU + MaxPool
  → Conv1D(128, k=3) + BN + ReLU + MaxPool
  → BiGRU(128 units, 2 layers, bidirectional)
  → Additive Attention
  → Linear(64) + ReLU + Dropout
  → Linear(4)  → Softmax
```

## Config

All hyperparameters are in `config.yaml`. Key settings:

| Parameter | Default | Notes |
|---|---|---|
| `window_size` | 30 | Timesteps per sample. Increase for slower trends. |
| `stride` | 5 | Sliding window overlap. Smaller = more samples. |
| `gru_hidden` | 128 | Increase to 256 for larger datasets. |
| `epochs` | 60 | Early stopping kicks in before this. |
| `patience` | 10 | Epochs without val improvement before stopping. |
