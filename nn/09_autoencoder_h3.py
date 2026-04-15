"""
Step 9: Supervised Autoencoder + MLP on h=3
Architecture:
  Input (328) -> Encoder (256->128->32) -> Bottleneck
  Bottleneck -> Decoder (128->256->328)  [reconstruction loss]
  concat(Bottleneck, Input) -> MLP (256->128->1) [prediction loss]

- Multi-task: reconstruction + prediction trained jointly
- Skip connection: bottleneck + raw input fed to prediction head
- GPU: uses cuda:1 (RTX 3060, 12GB)
- Baseline to beat: h=3 val score 0.1103 (Optuna LightGBM)
Outputs saved to: autoencoder_results/
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
OUT_DIR     = PROJECT_DIR / "autoencoder_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device("cuda:1")   # RTX 3060 12GB
VAL_SPLIT = 3500
SEED      = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DROP_COLS = [
    'id', 'code', 'sub_code', 'sub_category', 'horizon',
    'ts_index', 'y_target', 'weight'
]
LOW_IC_FEATURES = [
    'feature_b', 'feature_c', 'feature_d', 'feature_e',
    'feature_f', 'feature_g', 'feature_h', 'feature_i',
]

# ============================================================
# METRIC
# ============================================================
def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

# ============================================================
# MODEL
# ============================================================
class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout=0.1):
        super().__init__()

        # Encoder: input -> bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, bottleneck_dim),
        )

        # Decoder: bottleneck -> reconstruct input
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

        # Prediction head: concat(bottleneck, input) -> prediction
        # Skip connection gives direct access to raw features
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck_dim + input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        bottleneck   = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        pred_input   = torch.cat([bottleneck, x], dim=1)  # skip connection
        prediction   = self.predictor(pred_input)
        return prediction.squeeze(1), reconstruction

# ============================================================
# LOAD DATA
# ============================================================
print("Loading feature data...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")

drop_feature_cols = []
for f in LOW_IC_FEATURES:
    drop_feature_cols += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feature_cols = [c for c in drop_feature_cols if c in train.columns]
train = train.drop(columns=drop_feature_cols)
test  = test.drop(columns=[c for c in drop_feature_cols if c in test.columns])
feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Feature columns: {len(feat_cols)}")

# H=3 only
tr       = train[train.horizon == 3].copy()
te       = test[test.horizon == 3].copy()
train_df = tr[tr.ts_index <= VAL_SPLIT]
val_df   = tr[tr.ts_index  > VAL_SPLIT]

X_train_np = train_df[feat_cols].values.astype(np.float32)
X_val_np   = val_df[feat_cols].values.astype(np.float32)
X_test_np  = te[feat_cols].values.astype(np.float32)
y_train_np = train_df['y_target'].values.astype(np.float32)
y_val_np   = val_df['y_target'].values.astype(np.float32)
w_train_np = train_df['weight'].values.astype(np.float32)
w_val_np   = val_df['weight'].values.astype(np.float32)

# Standardize features (fit on train only)
feat_mean = X_train_np.mean(axis=0)
feat_std  = X_train_np.std(axis=0) + 1e-8
X_train_np = (X_train_np - feat_mean) / feat_std
X_val_np   = (X_val_np   - feat_mean) / feat_std
X_test_np  = (X_test_np  - feat_mean) / feat_std

print(f"  Train: {len(X_train_np):,} | Val: {len(X_val_np):,} | Test: {len(X_test_np):,}")

# Convert to tensors
X_train_t = torch.from_numpy(X_train_np).to(DEVICE)
y_train_t = torch.from_numpy(y_train_np).to(DEVICE)
w_train_t = torch.from_numpy(w_train_np).to(DEVICE)
X_val_t   = torch.from_numpy(X_val_np).to(DEVICE)
y_val_t   = torch.from_numpy(y_val_np).to(DEVICE)
w_val_t   = torch.from_numpy(w_val_np).to(DEVICE)
X_test_t  = torch.from_numpy(X_test_np).to(DEVICE)

# DataLoader for batched training
dataset    = TensorDataset(X_train_t, y_train_t, w_train_t)
loader     = DataLoader(dataset, batch_size=4096, shuffle=True,
                        pin_memory=False, drop_last=False)

# ============================================================
# TRAINING SETUP
# ============================================================
INPUT_DIM      = len(feat_cols)
BOTTLENECK_DIM = 32
EPOCHS         = 100
LR             = 3e-4
RECON_WEIGHT   = 0.1    # reconstruction loss weight vs prediction loss
L2_PRED_WEIGHT = 1e-4   # penalty to keep predictions near zero (combat high-weight y=0 rows)

model     = SupervisedAutoencoder(INPUT_DIM, BOTTLENECK_DIM, dropout=0.1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})")
print(f"Epochs: {EPOCHS} | Batch size: 4096 | LR: {LR}\n")

# ============================================================
# TRAINING LOOP
# ============================================================
start       = time.time()
best_score  = -1.0
best_epoch  = 0
best_state  = None
patience    = 15
no_improve  = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch, w_batch in loader:
        optimizer.zero_grad()

        pred, recon = model(X_batch)

        # Weighted prediction loss (MSE weighted by sample weights)
        pred_loss  = (w_batch * (pred - y_batch) ** 2).mean()

        # Reconstruction loss (unweighted MSE)
        recon_loss = ((recon - X_batch) ** 2).mean()

        # L2 penalty on raw predictions — penalizes large outputs, nudges toward zero
        pred_l2    = (pred ** 2).mean()

        loss = pred_loss + RECON_WEIGHT * recon_loss + L2_PRED_WEIGHT * pred_l2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred, _ = model(X_val_t)
        val_pred_np = val_pred.cpu().numpy()

    score = skill_score(y_val_np, val_pred_np, w_val_np)

    scheduler.step(score)  # ReduceLROnPlateau tracks val score

    if score > best_score:
        best_score = score
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(loader):.6f}  "
              f"val_score={score:.4f}  best={best_score:.4f}@ep{best_epoch}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

    if no_improve >= patience:
        print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

print(f"\nBest val score: {best_score:.4f} at epoch {best_epoch}")
print(f"Baseline LightGBM h=3: 0.1103")
print(f"Delta: {best_score - 0.1103:+.4f}")

# ============================================================
# GENERATE PREDICTIONS WITH BEST MODEL
# ============================================================
model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

with torch.no_grad():
    val_pred_final, _  = model(X_val_t)
    test_pred_final, _ = model(X_test_t)

val_preds_np  = val_pred_final.cpu().numpy()
test_preds_np = test_pred_final.cpu().numpy()

# ============================================================
# SAVE
# ============================================================
# Val predictions
val_out = val_df[['id', 'ts_index']].copy()
val_out['y_true']      = y_val_np
val_out['y_pred_nn']   = val_preds_np
val_out['weight']      = w_val_np
val_out.to_csv(OUT_DIR / f"val_preds_h3_nn_cv{best_score:.4f}.csv", index=False)

# Test predictions
te_out = te[['id']].copy()
te_out['prediction'] = test_preds_np
te_out.to_csv(OUT_DIR / f"test_preds_h3_nn.csv", index=False)

# Save model
torch.save(best_state, OUT_DIR / f"model_h3_cv{best_score:.4f}.pt")

# Save feature stats for inference
np.save(OUT_DIR / "feat_mean.npy", feat_mean)
np.save(OUT_DIR / "feat_std.npy",  feat_std)

with open(OUT_DIR / "results.txt", "w") as f:
    f.write(f"Supervised Autoencoder + MLP — h=3\n")
    f.write(f"====================================\n")
    f.write(f"Best val score   : {best_score:.4f}\n")
    f.write(f"LightGBM baseline: 0.1103\n")
    f.write(f"Delta            : {best_score - 0.1103:+.4f}\n")
    f.write(f"Best epoch       : {best_epoch}\n")
    f.write(f"Runtime          : {(time.time()-start)/60:.1f} min\n")
    f.write(f"Device           : {torch.cuda.get_device_name(DEVICE)}\n")

print(f"\nResults saved to: {OUT_DIR}")
print(f"Runtime: {(time.time()-start)/60:.1f} min")
