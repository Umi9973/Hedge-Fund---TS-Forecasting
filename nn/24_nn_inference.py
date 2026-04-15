"""
Pure NN inference for h=1 and h=25 (h=3 and h=10 already have test preds).
Loads saved model weights + normalization stats, generates test predictions,
then builds a full 4-horizon pure NN submission.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "autoencoder_results"
SUB_DIR     = PROJECT_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 3500

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'y_target', 'weight']
LOW_IC_FEATURES = ['feature_b', 'feature_c', 'feature_d', 'feature_e',
                   'feature_f', 'feature_g', 'feature_h', 'feature_i']


class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck_dim + input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),                        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        bottleneck     = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        prediction     = self.predictor(torch.cat([bottleneck, x], dim=1))
        return prediction.squeeze(1), reconstruction


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


print("Loading feature data...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")

drop_feat = []
for f in LOW_IC_FEATURES:
    drop_feat += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feat = [c for c in drop_feat if c in train.columns]
train = train.drop(columns=drop_feat)
test  = test.drop(columns=[c for c in drop_feat if c in test.columns])
feat_cols = [c for c in train.columns if c not in DROP_COLS]
INPUT_DIM = len(feat_cols)
print(f"  Features: {INPUT_DIM} | Device: {DEVICE}")

# ── Run inference for h=1 and h=25 ────────────────────────
for H in [1, 25]:
    print(f"\n{'='*50}\nInference h={H}\n{'='*50}")

    model_files = sorted(OUT_DIR.glob(f"model_h{H}_cv*.pt"))
    if not model_files:
        print(f"  No model found for h={H}, skipping.")
        continue
    model_path = model_files[-1]  # best score
    score = float(str(model_path).split('cv')[1].replace('.pt', ''))
    print(f"  Loading: {model_path.name} (val={score:.4f})")

    feat_mean = np.load(OUT_DIR / f"feat_mean_h{H}.npy")
    feat_std  = np.load(OUT_DIR / f"feat_std_h{H}.npy")

    te = test[test.horizon == H].copy()
    X_test_np = te[feat_cols].values.astype(np.float32)
    X_test_s  = (X_test_np - feat_mean) / feat_std
    X_test_t  = torch.from_numpy(X_test_s).to(DEVICE)

    model = SupervisedAutoencoder(INPUT_DIM, bottleneck_dim=32, dropout=0.1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds, _ = model(X_test_t)
        test_preds = preds.cpu().numpy()

    out = te[['id']].copy()
    out['prediction'] = test_preds
    out_path = OUT_DIR / f"test_preds_h{H}_nn_cv{score:.4f}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Test preds saved: {out_path.name} ({len(out):,} rows)")

# ── Build pure NN submission ───────────────────────────────
print(f"\n{'='*50}\nBuilding pure NN submission\n{'='*50}")

nn_test_files = {
    1:  sorted(OUT_DIR.glob("test_preds_h1_nn_cv*.csv"))[-1],
    3:  OUT_DIR / "test_preds_h3_nn.csv",
    10: OUT_DIR / "test_preds_h10_blend_a1.00_cv0.1932.csv",  # alpha=1.0 = pure NN
    25: sorted(OUT_DIR.glob("test_preds_h25_nn_cv*.csv"))[-1],
}

nn_val_scores = {1: 0.0501, 3: 0.1118, 10: 0.1932, 25: 0.2517}

all_preds = []
for H, fpath in nn_test_files.items():
    df = pd.read_csv(fpath)
    all_preds.append(df[['id', 'prediction']])
    print(f"  h={H}: {fpath.name} ({len(df):,} rows, val={nn_val_scores[H]:.4f})")

submission = pd.concat(all_preds, ignore_index=True)
submission = submission.sort_values('id').reset_index(drop=True)

assert submission['prediction'].isnull().sum() == 0, "NaN in predictions!"
print(f"\n  Total rows: {len(submission):,}")

avg_score = np.mean(list(nn_val_scores.values()))
out_path = SUB_DIR / f"submission_nn_pure_avg{avg_score:.4f}.csv"
submission.to_csv(out_path, index=False)
print(f"  Submission saved: {out_path}")
