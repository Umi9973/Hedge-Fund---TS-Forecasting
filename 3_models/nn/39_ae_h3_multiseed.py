"""
39_ae_h3_multiseed.py

Supervised Autoencoder h=3, 3 NN seeds (42, 2024, 777).
Same architecture/params as 38_ae_h3.py (single seed got val=0.1143).

Extra indicators per epoch:
  - train/val skill score + gap (overfitting)
  - pred_mean / pred_std on val (catches score collapse early)
  - val_corr: Pearson correlation between val preds and y_true
  - val_loss: raw weighted MSE on val (separate from train loss)

Outputs (for future LGBM blending):
  - val_preds_h3_nn_seed{s}_cv{score}.csv  (id, ts_index, y_true, y_pred, weight)
  - val_preds_h3_nn_ensemble_cv{score}.csv
  - test_preds_h3_nn_seed{s}.csv           (id, prediction)
  - test_preds_h3_nn_ensemble.csv

VAL_SPLIT = 3400 | v2 features, bottom-10 IC skew/kurt dropped | GPU: cuda:1
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "autoencoder_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device("cuda:1")
VAL_SPLIT = 3400
H         = 3
NN_SEEDS  = [42, 2024, 777]

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']

DROP_SKEWKURT = [
    f'{feat}_{stat}'
    for feat in ['feature_bq', 'feature_ag', 'feature_ap', 'feature_br', 'feature_bp',
                 'feature_bs', 'feature_bn', 'feature_bo', 'feature_al', 'feature_an']
    for stat in ['cs_skew', 'cs_kurt']
]

# ── Same NN params as 38_ae_h3.py ────────────────────────────
BOTTLENECK_DIM  = 32
EPOCHS          = 100
LR              = 3e-4
RECON_WEIGHT    = 0.1
L2_PRED_WEIGHT  = 1e-4
PATIENCE        = 15
BATCH_SIZE      = 4096
TRAIN_EVAL_SIZE = 20000


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


def pearson_corr(a, b):
    a, b = np.array(a), np.array(b)
    if a.std() < 1e-8 or b.std() < 1e-8: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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


# ── Load data ────────────────────────────────────────────────
print("Loading v2 features (h=3 only)...")
train = pd.read_parquet(PROJECT_DIR / "train_features_v2.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features_v2.parquet")

tr_all = train[train.horizon == H].copy()
te     = test[test.horizon == H].copy()

feat_cols = [c for c in train.columns if c not in DROP_COLS and c not in DROP_SKEWKURT]
del train, test

INPUT_DIM = len(feat_cols)
print(f"  Features: {INPUT_DIM} | GPU: {torch.cuda.get_device_name(DEVICE)}")

train_df = tr_all[tr_all.ts_index <= VAL_SPLIT]
val_df   = tr_all[tr_all.ts_index  > VAL_SPLIT]

print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(te):,}")
print(f"  Val ts_index: {val_df.ts_index.min()} - {val_df.ts_index.max()}")

# y_target distribution
y_all = train_df['y_target'].values
print(f"\n  y_target stats (train):")
print(f"    mean={y_all.mean():.6f}  std={y_all.std():.6f}  median={np.median(y_all):.6f}")
print(f"    min={y_all.min():.4f}  max={y_all.max():.4f}")
print(f"    % near-zero (|y|<0.001): {(np.abs(y_all) < 0.001).mean()*100:.1f}%")

X_train_np = train_df[feat_cols].values.astype(np.float32)
X_val_np   = val_df[feat_cols].values.astype(np.float32)
X_test_np  = te[feat_cols].values.astype(np.float32)
y_train_np = train_df['y_target'].values.astype(np.float32)
y_val_np   = val_df['y_target'].values.astype(np.float32)
w_train_np = train_df['weight'].values.astype(np.float32)
w_val_np   = val_df['weight'].values.astype(np.float32)

# Standardize (fit on train only)
feat_mean = X_train_np.mean(axis=0)
feat_std  = X_train_np.std(axis=0) + 1e-8
X_train_s = (X_train_np - feat_mean) / feat_std
X_val_s   = (X_val_np   - feat_mean) / feat_std
X_test_s  = (X_test_np  - feat_mean) / feat_std

X_val_t  = torch.from_numpy(X_val_s).to(DEVICE)
X_test_t = torch.from_numpy(X_test_s).to(DEVICE)

# ── Multi-seed NN training ────────────────────────────────────
nn_val_preds_list  = []
nn_test_preds_list = []
nn_scores          = []

for seed in NN_SEEDS:
    print(f"\n{'='*60}")
    print(f"NN seed={seed}  h={H}  VAL_SPLIT={VAL_SPLIT}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.from_numpy(X_train_s).to(DEVICE)
    y_train_t = torch.from_numpy(y_train_np).to(DEVICE)
    w_train_t = torch.from_numpy(w_train_np).to(DEVICE)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t, w_train_t),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    model     = SupervisedAutoencoder(INPUT_DIM, BOTTLENECK_DIM, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    rng            = np.random.default_rng(seed)
    eval_idx       = rng.choice(len(X_train_s), size=min(TRAIN_EVAL_SIZE, len(X_train_s)), replace=False)
    X_train_eval_t = torch.from_numpy(X_train_s[eval_idx]).to(DEVICE)
    y_train_eval   = y_train_np[eval_idx]
    w_train_eval   = w_train_np[eval_idx]

    best_score = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    history = {'epoch': [], 'train': [], 'val': [], 'gap': [], 'corr': [], 'pred_std': []}

    header = (f"  {'Ep':>4}  {'tr_loss':>8}  {'train':>7}  {'val':>7}  {'gap':>7}  "
              f"{'best':>7}  {'val_mae':>8}  {'corr':>6}  {'p_mean':>7}  {'p_std':>7}  {'lr':>8}")
    print(header)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            pred, recon = model(X_batch)
            pred_loss  = (w_batch * (pred - y_batch) ** 2).mean()
            recon_loss = ((recon - X_batch) ** 2).mean()
            pred_l2    = (pred ** 2).mean()
            loss = pred_loss + RECON_WEIGHT * recon_loss + L2_PRED_WEIGHT * pred_l2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred_t,   _  = model(X_val_t)
            train_pred_t, _  = model(X_train_eval_t)

        val_pred_np   = val_pred_t.cpu().numpy()
        train_pred_np = train_pred_t.cpu().numpy()

        val_score   = skill_score(y_val_np,    val_pred_np,   w_val_np)
        train_score = skill_score(y_train_eval, train_pred_np, w_train_eval)
        gap         = train_score - val_score
        val_mae     = float(np.abs(y_val_np - val_pred_np).mean())
        val_corr    = pearson_corr(val_pred_np, y_val_np)
        pred_mean   = float(val_pred_np.mean())
        pred_std    = float(val_pred_np.std())

        scheduler.step(val_score)

        history['epoch'].append(epoch)
        history['train'].append(train_score)
        history['val'].append(val_score)
        history['gap'].append(gap)
        history['corr'].append(val_corr)
        history['pred_std'].append(pred_std)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            flag = "  *** OVERFIT ***" if gap > 0.05 else ""
            print(f"  {epoch:>4}  {epoch_loss/len(loader):>8.4f}  "
                  f"{train_score:>7.4f}  {val_score:>7.4f}  {gap:>+7.4f}  "
                  f"{best_score:>7.4f}  {val_mae:>8.5f}  {val_corr:>6.4f}  "
                  f"{pred_mean:>7.4f}  {pred_std:>7.4f}  "
                  f"{optimizer.param_groups[0]['lr']:>8.6f}{flag}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = (time.time() - t0) / 60
    print(f"\n  Seed {seed} best val: {best_score:.4f} at epoch {best_epoch}  ({elapsed:.1f} min)")

    # ── Plot training curves ──────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    epochs = history['epoch']

    axes[0].plot(epochs, history['train'], label='train score', color='blue')
    axes[0].plot(epochs, history['val'],   label='val score',   color='orange')
    axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'best ep={best_epoch}')
    axes[0].set_ylabel('Skill Score')
    axes[0].set_title(f'h={H} NN seed={seed} — Skill Score (best val={best_score:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['gap'], color='red', label='gap (train-val)')
    axes[1].axhline(0.05, color='red', linestyle=':', alpha=0.5, label='overfit threshold')
    axes[1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Gap')
    axes[1].set_title('Overfitting Gap (train - val)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    ax2a = axes[2]
    ax2b = ax2a.twinx()
    ax2a.plot(epochs, history['corr'],     color='green',  label='val corr')
    ax2b.plot(epochs, history['pred_std'], color='purple', linestyle='--', label='pred std')
    ax2a.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax2a.set_ylabel('Pearson Corr', color='green')
    ax2b.set_ylabel('Pred Std',     color='purple')
    ax2a.set_xlabel('Epoch')
    ax2a.set_title('Val Correlation & Prediction Spread')
    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2)
    ax2a.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT_DIR / f"training_curves_h{H}_seed{seed}.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"  Plot saved: {plot_path.name}")

    # Inference with best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vp, _ = model(X_val_t)
        tp, _ = model(X_test_t)
    vp_np = vp.cpu().numpy()
    tp_np = tp.cpu().numpy()

    nn_val_preds_list.append(vp_np)
    nn_test_preds_list.append(tp_np)
    nn_scores.append(best_score)

    torch.save(best_state, OUT_DIR / f"model_h{H}_seed{seed}_cv{best_score:.4f}.pt")

    # Save per-seed val preds
    val_out = val_df[['id', 'ts_index']].copy()
    val_out['y_true']  = y_val_np
    val_out['y_pred']  = vp_np
    val_out['weight']  = w_val_np
    val_out.to_csv(OUT_DIR / f"val_preds_h{H}_nn_seed{seed}_cv{best_score:.4f}.csv", index=False)

    # Save per-seed test preds
    te_out = te[['id']].copy().reset_index(drop=True)
    te_out['prediction'] = tp_np
    te_out.to_csv(OUT_DIR / f"test_preds_h{H}_nn_seed{seed}.csv", index=False)

# ── Ensemble ─────────────────────────────────────────────────
nn_val_ensemble  = np.mean(nn_val_preds_list,  axis=0)
nn_test_ensemble = np.mean(nn_test_preds_list, axis=0)
ensemble_score   = skill_score(y_val_np, nn_val_ensemble, w_val_np)
ensemble_corr    = pearson_corr(nn_val_ensemble, y_val_np)

print(f"\n{'='*60}")
print(f"RESULTS — h={H} NN Multi-seed Ensemble")
print(f"{'='*60}")
print(f"  Per-seed scores : {[f'{s:.4f}' for s in nn_scores]}")
print(f"  Ensemble score  : {ensemble_score:.4f}  (single seed: 0.1143)")
print(f"  Ensemble corr   : {ensemble_corr:.4f}")
print(f"  Pred mean/std   : {nn_val_ensemble.mean():.4f} / {nn_val_ensemble.std():.4f}")

# Save ensemble val preds
val_ens = val_df[['id', 'ts_index']].copy()
val_ens['y_true'] = y_val_np
val_ens['y_pred'] = nn_val_ensemble
val_ens['weight'] = w_val_np
val_ens.to_csv(OUT_DIR / f"val_preds_h{H}_nn_ensemble_cv{ensemble_score:.4f}.csv", index=False)

# Save ensemble test preds
te_ens = te[['id']].copy().reset_index(drop=True)
te_ens['prediction'] = nn_test_ensemble
te_ens.to_csv(OUT_DIR / f"test_preds_h{H}_nn_ensemble_cv{ensemble_score:.4f}.csv", index=False)

print(f"\n  Saved to autoencoder_results/")
print(f"  Per-seed + ensemble val/test preds ready for blending.")
