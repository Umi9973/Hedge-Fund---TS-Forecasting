"""
38_ae_h3.py

Supervised Autoencoder + MLP for h=3 only.
Same architecture as 11_autoencoder_all_horizons.py.

Overfitting indicators printed every epoch:
  - train_score: skill score on training set (sampled, not full)
  - val_score:   skill score on validation set
  - gap = train_score - val_score  (large gap = overfitting)

VAL_SPLIT = 3400 (200-timestamp window)
v2 features (384 features)
GPU: cuda:1
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
H         = 3

torch.manual_seed(SEED)
np.random.seed(SEED)

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']

LGBM_PARAMS = {
    'boosting_type':    'gbdt',
    'objective':        'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'learning_rate':    0.01518,
    'max_depth':        7,    'num_leaves':       88,
    'feature_fraction': 0.50, 'bagging_fraction': 0.8048,
    'bagging_freq':     5,    'lambda_l1':        0.1182,
    'lambda_l2':        8.0,  'min_data_in_leaf': 200,
    'verbose':          -1,   'n_jobs':           -1,
}
# Same bottom-10 IC skew/kurt features to drop (same as 32_skewkurt_tuned_h3.py)
DROP_SKEWKURT = [
    f'{feat}_{stat}'
    for feat in ['feature_bq', 'feature_ag', 'feature_ap', 'feature_br', 'feature_bp',
                 'feature_bs', 'feature_bn', 'feature_bo', 'feature_al', 'feature_an']
    for stat in ['cs_skew', 'cs_kurt']
]
EARLY_STOPPING = 100


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


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
print(f"  Features: {len(feat_cols)} | GPU: {torch.cuda.get_device_name(DEVICE)}")

train_df = tr_all[tr_all.ts_index <= VAL_SPLIT]
val_df   = tr_all[tr_all.ts_index  > VAL_SPLIT]

print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(te):,}")
print(f"  Val ts_index: {val_df.ts_index.min()} – {val_df.ts_index.max()}")

X_train_np = train_df[feat_cols].values.astype(np.float32)
X_val_np   = val_df[feat_cols].values.astype(np.float32)
X_test_np  = te[feat_cols].values.astype(np.float32)
y_train_np = train_df['y_target'].values.astype(np.float32)
y_val_np   = val_df['y_target'].values.astype(np.float32)
w_train_np = train_df['weight'].values.astype(np.float32)
w_val_np   = val_df['weight'].values.astype(np.float32)

# Standardize
feat_mean = X_train_np.mean(axis=0)
feat_std  = X_train_np.std(axis=0) + 1e-8
X_train_s = (X_train_np - feat_mean) / feat_std
X_val_s   = (X_val_np   - feat_mean) / feat_std
X_test_s  = (X_test_np  - feat_mean) / feat_std

X_train_t = torch.from_numpy(X_train_s).to(DEVICE)
y_train_t = torch.from_numpy(y_train_np).to(DEVICE)
w_train_t = torch.from_numpy(w_train_np).to(DEVICE)
X_val_t   = torch.from_numpy(X_val_s).to(DEVICE)
X_test_t  = torch.from_numpy(X_test_s).to(DEVICE)

loader = DataLoader(TensorDataset(X_train_t, y_train_t, w_train_t),
                    batch_size=4096, shuffle=True, pin_memory=False)

# ── Train NN ─────────────────────────────────────────────────
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

# For train score: use a fixed random subset to avoid evaluating full train each epoch
TRAIN_EVAL_SIZE = 20000
rng = np.random.default_rng(SEED)
train_eval_idx = rng.choice(len(X_train_s), size=min(TRAIN_EVAL_SIZE, len(X_train_s)), replace=False)
X_train_eval_t = torch.from_numpy(X_train_s[train_eval_idx]).to(DEVICE)
y_train_eval   = y_train_np[train_eval_idx]
w_train_eval   = w_train_np[train_eval_idx]

print(f"\n{'='*60}")
print(f"Supervised Autoencoder h={H} | VAL_SPLIT={VAL_SPLIT} | v2 features")
print(f"  Overfitting indicator: gap = train_score - val_score")
print(f"  (gap > 0.02 suggests overfitting)")
print(f"{'='*60}")
print(f"  {'Ep':>4}  {'loss':>8}  {'train':>7}  {'val':>7}  {'gap':>7}  {'best':>7}  {'lr':>8}")

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
        val_pred_t, _   = model(X_val_t)
        train_pred_t, _ = model(X_train_eval_t)

    val_pred_np   = val_pred_t.cpu().numpy()
    train_pred_np = train_pred_t.cpu().numpy()

    val_score   = skill_score(y_val_np,       val_pred_np,   w_val_np)
    train_score = skill_score(y_train_eval,    train_pred_np, w_train_eval)
    gap         = train_score - val_score

    scheduler.step(val_score)

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
              f"{best_score:>7.4f}  {optimizer.param_groups[0]['lr']:>8.6f}{flag}")

    if no_improve >= patience:
        print(f"  Early stop at epoch {epoch} (patience={patience})")
        break

elapsed = (time.time() - t0) / 60
print(f"\n  NN best val: {best_score:.4f} at epoch {best_epoch}  ({elapsed:.1f} min)")

# ── Inference ────────────────────────────────────────────────
model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()
with torch.no_grad():
    val_pred_final, _  = model(X_val_t)
    test_pred_final, _ = model(X_test_t)
nn_val_preds  = val_pred_final.cpu().numpy()
nn_test_preds = test_pred_final.cpu().numpy()

# ── LightGBM (3 seeds) ───────────────────────────────────────
print(f"\n  Training LightGBM h={H} (3 seeds)...")
lgbm_val_preds  = np.zeros(len(X_val_np))
lgbm_test_preds = np.zeros(len(X_test_np))
es = LGBM_PARAMS.pop('early_stopping_rounds', EARLY_STOPPING)
params = LGBM_PARAMS.copy()

for seed in [42, 2024, 777]:
    m = lgb.LGBMRegressor(**params, seed=seed)
    m.fit(
        train_df[feat_cols], y_train_np,
        sample_weight=w_train_np,
        eval_set=[(val_df[feat_cols], y_val_np)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    lgbm_val_preds  += m.predict(val_df[feat_cols]) / 3
    lgbm_test_preds += m.predict(te[feat_cols])     / 3
    print(f"    seed={seed}  best_iter={m.best_iteration_}")

lgbm_score = skill_score(y_val_np, lgbm_val_preds, w_val_np)
print(f"  LightGBM val: {lgbm_score:.4f}")

# ── Blend search ─────────────────────────────────────────────
blend_results = []
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = (1 - alpha) * lgbm_val_preds + alpha * nn_val_preds
    s = skill_score(y_val_np, blended, w_val_np)
    blend_results.append((alpha, s))

best_alpha, best_blend_score = max(blend_results, key=lambda x: x[1])

print(f"\n{'='*60}")
print(f"RESULTS — h={H} Autoencoder + LightGBM")
print(f"{'='*60}")
print(f"  NN val       : {best_score:.4f}")
print(f"  LGBM val     : {lgbm_score:.4f}  (baseline dart_results: 0.1211)")
print(f"  Best blend   : {best_blend_score:.4f}  (alpha={best_alpha:.2f}, "
      f"delta vs LGBM: {best_blend_score - lgbm_score:+.4f})")

# Top 5 blend alphas
print(f"\n  Blend curve (top 5):")
for alpha, s in sorted(blend_results, key=lambda x: -x[1])[:5]:
    print(f"    alpha={alpha:.2f}  score={s:.4f}")

# ── Save ─────────────────────────────────────────────────────
val_out = val_df[['id', 'ts_index']].copy()
val_out['y_true']    = y_val_np
val_out['y_pred_nn'] = nn_val_preds
val_out['weight']    = w_val_np
val_out.to_csv(OUT_DIR / f"val_preds_h{H}_nn_cv{best_score:.4f}.csv", index=False)

torch.save(best_state, OUT_DIR / f"model_h{H}_cv{best_score:.4f}.pt")

best_test_blend = (1 - best_alpha) * lgbm_test_preds + best_alpha * nn_test_preds
te_out = te[['id']].copy().reset_index(drop=True)
te_out['prediction'] = best_test_blend
te_out.to_csv(OUT_DIR / f"test_preds_h{H}_blend_a{best_alpha:.2f}_cv{best_blend_score:.4f}.csv", index=False)

te_lgbm = te[['id']].copy().reset_index(drop=True)
te_lgbm['prediction'] = lgbm_test_preds
te_lgbm.to_csv(OUT_DIR / f"test_preds_h{H}_lgbm_cv{lgbm_score:.4f}.csv", index=False)

print(f"\n  Saved to autoencoder_results/")
