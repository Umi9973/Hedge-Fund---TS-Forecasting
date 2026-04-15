"""
Step 11: Supervised Autoencoder + MLP on h=1, h=10, h=25
Same architecture and fixes as h=3 run 2 (which beat LightGBM: 0.1118 vs 0.1103):
  - ReduceLROnPlateau (no cosine restarts)
  - L2 prediction penalty to keep predictions near zero
  - Skip connection: concat(bottleneck, input) -> MLP
  - Multi-task: reconstruction + prediction loss

After each horizon, blends NN with LightGBM (21 alpha ratios) and saves best blend.
GPU: cuda:1 (RTX 3060, 12GB)
Outputs saved to: autoencoder_results/
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightgbm as lgb
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "autoencoder_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device("cuda:1")
VAL_SPLIT = 3400
SEED      = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']
LOW_IC_FEATURES = [
    'feature_b', 'feature_c', 'feature_d', 'feature_e',
    'feature_f', 'feature_g', 'feature_h', 'feature_i',
]

# Optuna-tuned LightGBM params per horizon
LGBM_HORIZON_PARAMS = {
    1: {
        'num_leaves': 20,  'min_child_samples': 59,  'lambda_l2': 1.058,
        'max_depth': 9,    'learning_rate': 0.04407,  'feature_fraction': 0.622,
        'bagging_fraction': 0.309, 'early_stopping_rounds': 295,
    },
    10: {
        'num_leaves': 93,  'min_child_samples': 469, 'lambda_l2': 9.232,
        'max_depth': 9,    'learning_rate': 0.00698,  'feature_fraction': 0.490,
        'bagging_fraction': 0.859, 'early_stopping_rounds': 145,
    },
    25: {
        'num_leaves': 120, 'min_child_samples': 392, 'lambda_l2': 6.810,
        'max_depth': 9,    'learning_rate': 0.00519,  'feature_fraction': 0.737,
        'bagging_fraction': 0.599, 'early_stopping_rounds': 192,
    },
}
LGBM_BASE = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'bagging_freq': 5, 'lambda_l1': 0.5, 'verbosity': -1, 'n_jobs': -1,
}

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

# ============================================================
# LOAD DATA (once)
# ============================================================
print("Loading feature data (v2 with skew/kurt)...")
train = pd.read_parquet(PROJECT_DIR / "train_features_v2.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features_v2.parquet")
feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Feature columns: {len(feat_cols)}")
print(f"  GPU: {torch.cuda.get_device_name(DEVICE)}\n")

# ============================================================
# PER-HORIZON LOOP
# ============================================================
summary = []

for H in [1, 10, 25]:
    print(f"{'='*60}")
    print(f"HORIZON {H}")
    print(f"{'='*60}")
    t0 = time.time()

    # -- Data split --
    tr       = train[train.horizon == H].copy()
    te       = test[test.horizon == H].copy()
    train_df = tr[tr.ts_index <= VAL_SPLIT]
    val_df   = tr[tr.ts_index  > VAL_SPLIT]

    X_train_np = train_df[feat_cols].values.astype(np.float32)
    X_val_np   = val_df[feat_cols].values.astype(np.float32)
    X_test_np  = te[feat_cols].values.astype(np.float32)
    y_train_np = train_df['y_target'].values.astype(np.float32)
    y_val_np   = val_df['y_target'].values.astype(np.float32)
    w_train_np = train_df['weight'].values.astype(np.float32)
    w_val_np   = val_df['weight'].values.astype(np.float32)

    print(f"  Train: {len(X_train_np):,} | Val: {len(X_val_np):,} | Test: {len(X_test_np):,}")

    # Standardize (fit on train only)
    feat_mean = X_train_np.mean(axis=0)
    feat_std  = X_train_np.std(axis=0) + 1e-8
    X_train_s = (X_train_np - feat_mean) / feat_std
    X_val_s   = (X_val_np   - feat_mean) / feat_std
    X_test_s  = (X_test_np  - feat_mean) / feat_std

    # Tensors
    X_train_t = torch.from_numpy(X_train_s).to(DEVICE)
    y_train_t = torch.from_numpy(y_train_np).to(DEVICE)
    w_train_t = torch.from_numpy(w_train_np).to(DEVICE)
    X_val_t   = torch.from_numpy(X_val_s).to(DEVICE)
    X_test_t  = torch.from_numpy(X_test_s).to(DEVICE)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t, w_train_t),
                        batch_size=4096, shuffle=True, pin_memory=False)

    # -- Train NN --
    INPUT_DIM      = len(feat_cols)
    BOTTLENECK_DIM = 32
    EPOCHS         = 100
    LR             = 3e-4
    RECON_WEIGHT   = 0.1
    L2_PRED_WEIGHT = 1e-4

    model     = SupervisedAutoencoder(INPUT_DIM, BOTTLENECK_DIM, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    patience   = 15
    no_improve = 0

    print(f"  Training NN (up to {EPOCHS} epochs, patience={patience})...")
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
            val_pred, _ = model(X_val_t)
            val_pred_np = val_pred.cpu().numpy()
        score = skill_score(y_val_np, val_pred_np, w_val_np)
        scheduler.step(score)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(loader):.4f}  "
                  f"val={score:.4f}  best={best_score:.4f}@ep{best_epoch}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    print(f"  NN best val score: {best_score:.4f} at epoch {best_epoch}")

    # Save NN val/test preds
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        val_pred_final, _  = model(X_val_t)
        test_pred_final, _ = model(X_test_t)
    nn_val_preds  = val_pred_final.cpu().numpy()
    nn_test_preds = test_pred_final.cpu().numpy()

    val_out = val_df[['id', 'ts_index']].copy()
    val_out['y_true']    = y_val_np
    val_out['y_pred_nn'] = nn_val_preds
    val_out['weight']    = w_val_np
    val_out.to_csv(OUT_DIR / f"val_preds_h{H}_nn_cv{best_score:.4f}.csv", index=False)

    torch.save(best_state, OUT_DIR / f"model_h{H}_cv{best_score:.4f}.pt")
    np.save(OUT_DIR / f"feat_mean_h{H}.npy", feat_mean)
    np.save(OUT_DIR / f"feat_std_h{H}.npy",  feat_std)

    # -- Train LightGBM (3 seeds) --
    print(f"  Training LightGBM h={H} (3 seeds)...")
    lgbm_val_preds  = np.zeros(len(X_val_np))
    lgbm_test_preds = np.zeros(len(X_test_np))
    hp = {**LGBM_BASE, **LGBM_HORIZON_PARAMS[H]}
    es = hp.pop('early_stopping_rounds')

    for seed in [42, 2024, 777]:
        m = lgb.LGBMRegressor(**hp, seed=seed)
        m.fit(
            train_df[feat_cols], y_train_np,
            sample_weight=w_train_np,
            eval_set=[(val_df[feat_cols], y_val_np)],
            callbacks=[lgb.early_stopping(stopping_rounds=es, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        lgbm_val_preds  += m.predict(val_df[feat_cols])  / 3
        lgbm_test_preds += m.predict(te[feat_cols])      / 3
        print(f"    seed={seed}  best_iter={m.best_iteration_}")

    lgbm_score = skill_score(y_val_np, lgbm_val_preds, w_val_np)
    print(f"  LightGBM val score: {lgbm_score:.4f}")

    # -- Blend search --
    print(f"  Blend search (alpha = NN weight):")
    blend_results = []
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = (1 - alpha) * lgbm_val_preds + alpha * nn_val_preds
        s = skill_score(y_val_np, blended, w_val_np)
        blend_results.append((alpha, s))

    best_alpha, best_blend_score = max(blend_results, key=lambda x: x[1])
    print(f"  Best blend: alpha={best_alpha:.2f}  score={best_blend_score:.4f}  "
          f"(lgbm={lgbm_score:.4f}  nn={best_score:.4f}  delta={best_blend_score-lgbm_score:+.4f})")

    # Save best blend test preds
    best_test_blend = (1 - best_alpha) * lgbm_test_preds + best_alpha * nn_test_preds
    te_out = te[['id']].copy().reset_index(drop=True)
    te_out['prediction'] = best_test_blend
    te_out.to_csv(OUT_DIR / f"test_preds_h{H}_blend_a{best_alpha:.2f}_cv{best_blend_score:.4f}.csv", index=False)

    # Also save pure LightGBM test preds (for reference)
    te_lgbm = te[['id']].copy().reset_index(drop=True)
    te_lgbm['prediction'] = lgbm_test_preds
    te_lgbm.to_csv(OUT_DIR / f"test_preds_h{H}_lgbm_cv{lgbm_score:.4f}.csv", index=False)

    elapsed = (time.time() - t0) / 60
    summary.append({
        'horizon': H,
        'nn_score': best_score,
        'lgbm_score': lgbm_score,
        'best_alpha': best_alpha,
        'blend_score': best_blend_score,
        'delta': best_blend_score - lgbm_score,
        'runtime_min': elapsed,
    })

    print(f"  Runtime: {elapsed:.1f} min\n")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  {'H':>4}  {'NN':>8}  {'LGBM':>8}  {'Blend':>8}  {'Alpha':>6}  {'Delta':>8}")
for r in summary:
    print(f"  {r['horizon']:>4}  {r['nn_score']:>8.4f}  {r['lgbm_score']:>8.4f}  "
          f"{r['blend_score']:>8.4f}  {r['best_alpha']:>6.2f}  {r['delta']:>+8.4f}")

# Save summary
with open(OUT_DIR / "all_horizons_summary.txt", "w") as f:
    f.write("Supervised Autoencoder — All Horizons\n")
    f.write("======================================\n")
    f.write(f"{'H':>4}  {'NN':>8}  {'LGBM':>8}  {'Blend':>8}  {'Alpha':>6}  {'Delta':>8}\n")
    for r in summary:
        f.write(f"{r['horizon']:>4}  {r['nn_score']:>8.4f}  {r['lgbm_score']:>8.4f}  "
                f"{r['blend_score']:>8.4f}  {r['best_alpha']:>6.2f}  {r['delta']:>+8.4f}\n")

print(f"\nAll results saved to: {OUT_DIR}")
