"""
41_blend_h3.py

Blend best NN ensemble (from autoencoder_results/) with LGBM for h=3.
Searches alpha=0.0→1.0 in 0.05 steps and saves best blend.

Reads NN val preds from autoencoder_results/ — picks highest CV ensemble file.
Runs LGBM inline (3 seeds, same params as 32_skewkurt_tuned_h3.py → best CV=0.1211).

VAL_SPLIT = 3400 (200-timestamp window, same as NN scripts)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import glob

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "autoencoder_results"

VAL_SPLIT = 3400
H         = 3

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']

DROP_SKEWKURT = [
    f'{feat}_{stat}'
    for feat in ['feature_bq', 'feature_ag', 'feature_ap', 'feature_br', 'feature_bp',
                 'feature_bs', 'feature_bn', 'feature_bo', 'feature_al', 'feature_an']
    for stat in ['cs_skew', 'cs_kurt']
]

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
EARLY_STOPPING = 100


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


# ── Load best NN ensemble val preds ──────────────────────────
# Pick highest CV score among all ensemble val pred files
ensemble_files = sorted(
    glob.glob(str(OUT_DIR / f"val_preds_h{H}_*ensemble*.csv")),
    key=lambda f: float(f.split('_cv')[-1].replace('.csv', '')),
    reverse=True,
)
if not ensemble_files:
    raise FileNotFoundError(f"No ensemble val pred files found in {OUT_DIR}")

best_nn_file = ensemble_files[0]
cv_score_in_name = float(best_nn_file.split('_cv')[-1].replace('.csv', ''))
print(f"Best NN ensemble val preds: {Path(best_nn_file).name}  (CV={cv_score_in_name:.4f})")

nn_val_df  = pd.read_csv(best_nn_file)
nn_val_preds = nn_val_df['y_pred'].values
y_val_np     = nn_val_df['y_true'].values
w_val_np     = nn_val_df['weight'].values
nn_score     = skill_score(y_val_np, nn_val_preds, w_val_np)
print(f"NN val score (recomputed): {nn_score:.4f}")

# Corresponding test ensemble file (same cv score prefix)
nn_test_file = best_nn_file.replace('val_preds', 'test_preds').replace('y_true', 'prediction')
# Handle naming: test files have slightly different naming convention
test_cv = cv_score_in_name
test_candidates = sorted(
    glob.glob(str(OUT_DIR / f"test_preds_h{H}_*ensemble*.csv")),
    key=lambda f: abs(float(f.split('_cv')[-1].replace('.csv', '')) - test_cv)
)
if not test_candidates:
    raise FileNotFoundError(f"No matching ensemble test pred files found")
nn_test_file = test_candidates[0]
print(f"NN test preds: {Path(nn_test_file).name}")
nn_test_df   = pd.read_csv(nn_test_file)
nn_test_preds = nn_test_df['prediction'].values

# ── Load data for LGBM ───────────────────────────────────────
print("\nLoading v2 features for LGBM...")
train = pd.read_parquet(PROJECT_DIR / "train_features_v2.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features_v2.parquet")

train_h = train[train.horizon == H].copy()
test_h  = test[test.horizon == H].copy()
del train, test

feat_cols = [c for c in train_h.columns if c not in DROP_COLS and c not in DROP_SKEWKURT]

train_df = train_h[train_h.ts_index <= VAL_SPLIT]
val_df   = train_h[train_h.ts_index  > VAL_SPLIT]

X_tr = train_df[feat_cols].values.astype(np.float32)
y_tr = train_df['y_target'].values
w_tr = train_df['weight'].values
X_va = val_df[feat_cols].values.astype(np.float32)
y_va = val_df['y_target'].values
w_va = val_df['weight'].values
X_te = test_h[feat_cols].values.astype(np.float32)

print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_h):,}")

# ── Train LGBM (3 seeds) ─────────────────────────────────────
print(f"\nTraining LGBM h={H} (3 seeds)...")
lgbm_val_preds  = np.zeros(len(X_va))
lgbm_test_preds = np.zeros(len(X_te))

for seed in [42, 2024, 777]:
    m = lgb.LGBMRegressor(**LGBM_PARAMS, seed=seed)
    m.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    lgbm_val_preds  += m.predict(X_va) / 3
    lgbm_test_preds += m.predict(X_te) / 3
    print(f"  seed={seed}  best_iter={m.best_iteration_}")

lgbm_score = skill_score(y_va, lgbm_val_preds, w_va)
print(f"  LGBM val score: {lgbm_score:.4f}  (32_skewkurt_tuned baseline at VAL_SPLIT=3500: 0.1211)")

# ── Blend search ─────────────────────────────────────────────
print(f"\nBlend search (alpha = NN weight, 1-alpha = LGBM weight):")
print(f"  {'alpha':>6}  {'score':>8}  {'delta_vs_lgbm':>14}  {'delta_vs_nn':>12}")

blend_results = []
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = (1 - alpha) * lgbm_val_preds + alpha * nn_val_preds
    s = skill_score(y_va, blended, w_va)
    blend_results.append((alpha, s))
    print(f"  {alpha:>6.2f}  {s:>8.4f}  {s - lgbm_score:>+14.4f}  {s - nn_score:>+12.4f}")

best_alpha, best_blend_score = max(blend_results, key=lambda x: x[1])

print(f"\n{'='*60}")
print(f"RESULTS — h={H} NN + LGBM Blend")
print(f"{'='*60}")
print(f"  NN val         : {nn_score:.4f}  ({Path(best_nn_file).name})")
print(f"  LGBM val       : {lgbm_score:.4f}")
print(f"  Best blend     : {best_blend_score:.4f}  (alpha={best_alpha:.2f})")
print(f"  Delta vs LGBM  : {best_blend_score - lgbm_score:+.4f}")
print(f"  Delta vs NN    : {best_blend_score - nn_score:+.4f}")
print(f"  Previous best blend (39): 0.1216")
print(f"  Delta vs prev  : {best_blend_score - 0.1216:+.4f}")

# ── Save ─────────────────────────────────────────────────────
best_test_blend = (1 - best_alpha) * lgbm_test_preds + best_alpha * nn_test_preds

te_out = test_h[['id']].copy().reset_index(drop=True)
te_out['prediction'] = best_test_blend
te_out.to_csv(OUT_DIR / f"test_preds_h{H}_blend_a{best_alpha:.2f}_cv{best_blend_score:.4f}.csv", index=False)

# Save LGBM preds separately for reference
te_lgbm = test_h[['id']].copy().reset_index(drop=True)
te_lgbm['prediction'] = lgbm_test_preds
te_lgbm.to_csv(OUT_DIR / f"test_preds_h{H}_lgbm41_cv{lgbm_score:.4f}.csv", index=False)

# Save val blend preds
val_blend = val_df[['id', 'ts_index']].copy()
val_blend['y_true']    = y_va
val_blend['y_pred']    = (1 - best_alpha) * lgbm_val_preds + best_alpha * nn_val_preds
val_blend['weight']    = w_va
val_blend.to_csv(OUT_DIR / f"val_preds_h{H}_blend_a{best_alpha:.2f}_cv{best_blend_score:.4f}.csv", index=False)

print(f"\n  Saved to autoencoder_results/")
print(f"  test_preds_h{H}_blend_a{best_alpha:.2f}_cv{best_blend_score:.4f}.csv")
