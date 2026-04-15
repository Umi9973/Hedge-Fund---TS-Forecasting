"""
37_multitask_nn.py

Multi-task Supervised Autoencoder — all 4 horizons simultaneously.

Architecture:
  Input(388) → Encoder(256→128→32 bottleneck) → Decoder(128→256→388)
  + 4 separate predictor heads (one per horizon):
    concat(bottleneck, input) → Dense(256→128→1) per horizon

Key differences from per-horizon autoencoder (11_autoencoder_all_horizons.py):
  - All 4 horizons in one model (true shared representation)
  - horizon one-hot added as input feature (4 cols)
  - 4 output heads, each with its own weighted loss
  - Loss = sum(horizon_weight_i * pred_loss_i) + recon_loss
  - horizon weights match competition weight distribution:
    h=1: 0.20, h=3: 0.377, h=10: 0.25, h=25: 0.173 (approx from EDA)
  - VAL_SPLIT = 3400 (200-timestamp val window)
  - v2 features (384 + 4 horizon one-hots = 388)
  - GPU: cuda:1

Baselines:
  Per-horizon AE best: h=1=0.0501, h=3=0.1118, h=10=0.1932, h=25=0.2517
  Combined LightGBM CV: 0.2454
  Best LB: 0.2438
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "multitask_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device("cuda:1")
VAL_SPLIT = 3400
SEED      = 42
HORIZONS  = [1, 3, 10, 25]

torch.manual_seed(SEED)
np.random.seed(SEED)

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']

# Approximate horizon weight distribution from competition EDA
# h=3 carries ~37.7% of total weight — should dominate training
HORIZON_LOSS_WEIGHTS = {1: 0.20, 3: 0.377, 10: 0.25, 25: 0.173}

BATCH_SIZE   = 4096
MAX_EPOCHS   = 150
PATIENCE     = 15
LR           = 1e-3
RECON_WEIGHT = 0.1
L2_PEN       = 1e-4


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


# ── Model ─────────────────────────────────────────────────────
class MultiTaskAutoencoder(nn.Module):
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
        # 4 separate prediction heads — one per horizon
        # Each head sees bottleneck + full input (skip connection)
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(bottleneck_dim + input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128),                        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 1),
            ) for h in HORIZONS
        })

    def forward(self, x):
        bottleneck     = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        skip           = torch.cat([bottleneck, x], dim=1)
        predictions    = {h: self.heads[str(h)](skip).squeeze(1) for h in HORIZONS}
        return predictions, reconstruction


# ── Load data ─────────────────────────────────────────────────
print("Loading v2 features...")
train = pd.read_parquet(PROJECT_DIR / "train_features_v2.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features_v2.parquet")
print(f"  Train: {train.shape} | Test: {test.shape}")

# Add horizon one-hots
for h in HORIZONS:
    train[f'horizon_{h}'] = (train['horizon'] == h).astype(np.float32)
    test[f'horizon_{h}']  = (test['horizon']  == h).astype(np.float32)

feat_cols = [c for c in train.columns if c not in DROP_COLS]
INPUT_DIM = len(feat_cols)
print(f"  Input dim: {INPUT_DIM} (384 v2 + 4 horizon one-hots)")
print(f"  GPU: {torch.cuda.get_device_name(DEVICE)}")

# ── Train/val split ───────────────────────────────────────────
tr_df = train[train.ts_index <= VAL_SPLIT].reset_index(drop=True)
va_df = train[train.ts_index >  VAL_SPLIT].reset_index(drop=True)
te_df = test.reset_index(drop=True)

print(f"  VAL_SPLIT={VAL_SPLIT} | Train: {len(tr_df):,} | Val: {len(va_df):,} | Test: {len(te_df):,}")

# Normalize features (fit on train only)
# Compute normalization stats + normalize column-by-column to stay in float32
print("  Computing normalization stats and normalizing (per-column)...")
feat_mean = np.zeros(len(feat_cols), dtype=np.float32)
feat_std  = np.zeros(len(feat_cols), dtype=np.float32)

X_tr = np.empty((len(tr_df), len(feat_cols)), dtype=np.float32)
X_va = np.empty((len(va_df), len(feat_cols)), dtype=np.float32)
X_te = np.empty((len(te_df), len(feat_cols)), dtype=np.float32)

for i, c in enumerate(feat_cols):
    col_tr = tr_df[c].values.astype(np.float32)
    mu     = col_tr.mean()
    sigma  = col_tr.std() + 1e-8
    feat_mean[i] = mu
    feat_std[i]  = sigma
    X_tr[:, i]   = (col_tr - mu) / sigma
    X_va[:, i]   = (va_df[c].values.astype(np.float32) - mu) / sigma
    X_te[:, i]   = (te_df[c].values.astype(np.float32) - mu) / sigma
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{len(feat_cols)}] features normalized...")

np.save(OUT_DIR / "multitask_feat_mean.npy", feat_mean)
np.save(OUT_DIR / "multitask_feat_std.npy",  feat_std)
print(f"  Normalization done. X_tr: {X_tr.shape}")

y_tr = tr_df['y_target'].values.astype(np.float32)
w_tr = tr_df['weight'].values.astype(np.float32)
h_tr = tr_df['horizon'].values

y_va = va_df['y_target'].values.astype(np.float32)
w_va = va_df['weight'].values.astype(np.float32)
h_va = va_df['horizon'].values

# ── DataLoader ────────────────────────────────────────────────
train_ds = TensorDataset(
    torch.from_numpy(X_tr),
    torch.from_numpy(y_tr),
    torch.from_numpy(w_tr),
    torch.from_numpy(h_tr.astype(np.int32)),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)

X_va_t = torch.from_numpy(X_va).to(DEVICE)
X_te_t = torch.from_numpy(X_te).to(DEVICE)

# ── Train ─────────────────────────────────────────────────────
model     = MultiTaskAutoencoder(INPUT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

best_val_score = -np.inf
best_state     = None
no_improve     = 0
t0             = time.time()

print(f"\n{'='*60}")
print(f"Multi-task Autoencoder | {INPUT_DIM} features | 4 heads | GPU")
print(f"  VAL_SPLIT={VAL_SPLIT} | batch={BATCH_SIZE} | lr={LR} | max_epochs={MAX_EPOCHS}")
print(f"{'='*60}\n")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for X_b, y_b, w_b, h_b in train_loader:
        X_b = X_b.to(DEVICE)
        y_b = y_b.to(DEVICE)
        w_b = w_b.to(DEVICE)
        h_b = h_b.to(DEVICE)

        preds, recon = model(X_b)

        # Weighted prediction loss per horizon
        pred_loss = torch.tensor(0.0, device=DEVICE)
        for h in HORIZONS:
            mask = (h_b == h)
            if mask.sum() == 0:
                continue
            p  = preds[h][mask]
            yt = y_b[mask]
            wt = w_b[mask]
            wt = wt / (wt.sum() + 1e-8)
            h_pred_loss = (wt * (yt - p) ** 2).sum()
            pred_loss   = pred_loss + HORIZON_LOSS_WEIGHTS[h] * h_pred_loss

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, X_b)

        # L2 penalty on predictions (keep near zero)
        l2_pen = sum((preds[h] ** 2).mean() for h in HORIZONS) / len(HORIZONS)

        loss = pred_loss + RECON_WEIGHT * recon_loss + L2_PEN * l2_pen

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # ── Validation ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_preds_dict, _ = model(X_va_t)
        val_pred_np = np.zeros(len(va_df), dtype=np.float32)
        for h in HORIZONS:
            mask = h_va == h
            val_pred_np[mask] = val_preds_dict[h].cpu().numpy()[mask]

    val_score = skill_score(y_va, val_pred_np, w_va)
    scheduler.step(val_score)

    # Per-horizon breakdown every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        per_h = {}
        for h in HORIZONS:
            mask = h_va == h
            per_h[h] = skill_score(y_va[mask], val_pred_np[mask], w_va[mask])
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d} | loss={epoch_loss/len(train_loader):.5f} | "
              f"val={val_score:.4f} | lr={lr_now:.2e} | "
              f"h1={per_h[1]:.4f} h3={per_h[3]:.4f} h10={per_h[10]:.4f} h25={per_h[25]:.4f}")

    if val_score > best_val_score:
        best_val_score = val_score
        best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve     = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\n  Best val score: {best_val_score:.4f}  (time: {time.time()-t0:.1f}s)")

# ── Inference with best model ─────────────────────────────────
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    val_preds_dict, _ = model(X_va_t)
    test_preds_dict, _ = model(X_te_t)

val_pred_final  = np.zeros(len(va_df),  dtype=np.float32)
test_pred_final = np.zeros(len(te_df),  dtype=np.float32)

for h in HORIZONS:
    mask_va = h_va == h
    mask_te = te_df['horizon'].values == h
    val_pred_final[mask_va]  = val_preds_dict[h].cpu().numpy()[mask_va]
    test_pred_final[mask_te] = test_preds_dict[h].cpu().numpy()[mask_te]

final_score = skill_score(y_va, val_pred_final, w_va)

print(f"\n{'='*60}")
print(f"RESULTS — Multi-task Autoencoder")
print(f"{'='*60}")
print(f"  Combined val score : {final_score:.4f}")
print(f"  Baseline LGBM CV   : 0.2454  ({final_score - 0.2454:+.4f})")
print(f"  Best LB            : 0.2438")
print(f"\n  Per-horizon breakdown:")
for h in HORIZONS:
    mask = h_va == h
    h_score = skill_score(y_va[mask], val_pred_final[mask], w_va[mask])
    print(f"    h={h:2d}: {h_score:.4f}")

# ── Save ──────────────────────────────────────────────────────
torch.save(best_state, OUT_DIR / f"multitask_model_cv{final_score:.4f}.pt")

val_out           = va_df[['id', 'ts_index', 'y_target', 'weight', 'horizon']].copy()
val_out['y_pred'] = val_pred_final
val_out.to_csv(OUT_DIR / f"val_preds_multitask_cv{final_score:.4f}.csv", index=False)

test_out               = te_df[['id', 'horizon']].copy()
test_out['prediction'] = test_pred_final
sub = test_out[['id', 'prediction']].sort_values('id').reset_index(drop=True)
sub.to_csv(OUT_DIR / f"submission_multitask_cv{final_score:.4f}.csv", index=False)

print(f"\n  Model saved: multitask_model_cv{final_score:.4f}.pt")
print(f"  Submission: submission_multitask_cv{final_score:.4f}.csv")
