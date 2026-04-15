"""
Step 14: Blend LightGBM + Autoencoder NN for h=25
- Retrain h=25 LightGBM (3-seed ensemble) with Optuna params
- Load NN val predictions; recover NN test predictions from saved blend/lgbm files
- Test blend ratios 0.0 to 1.0 (step 0.05) on val score
- Save best blend test predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

H           = 25
PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
AE_DIR      = PROJECT_DIR / "autoencoder_results"
OUT_DIR     = PROJECT_DIR / "autoencoder_results"
VAL_SPLIT   = 3500

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']
LOW_IC_FEATURES = [
    'feature_b', 'feature_c', 'feature_d', 'feature_e',
    'feature_f', 'feature_g', 'feature_h', 'feature_i',
]

def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

BASE_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'bagging_freq': 5, 'lambda_l1': 0.5, 'verbosity': -1, 'n_jobs': -1,
}
HORIZON_PARAMS = {
    'num_leaves': 120, 'min_child_samples': 392, 'lambda_l2': 6.810,
    'max_depth': 9,    'learning_rate': 0.00519,  'feature_fraction': 0.737,
    'bagging_fraction': 0.599, 'early_stopping': 192,
}

# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading data (h={H})...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")

drop_feature_cols = []
for f in LOW_IC_FEATURES:
    drop_feature_cols += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feature_cols = [c for c in drop_feature_cols if c in train.columns]
train = train.drop(columns=drop_feature_cols)
test  = test.drop(columns=[c for c in drop_feature_cols if c in test.columns])
feat_cols = [c for c in train.columns if c not in DROP_COLS]

tr       = train[train.horizon == H].copy()
te       = test[test.horizon == H].copy()
train_df = tr[tr.ts_index <= VAL_SPLIT]
val_df   = tr[tr.ts_index  > VAL_SPLIT]

X_train = train_df[feat_cols]
X_val   = val_df[feat_cols]
X_test  = te[feat_cols]
y_train = train_df['y_target']
y_val   = val_df['y_target'].values
w_train = train_df['weight']
w_val   = val_df['weight'].values
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ============================================================
# TRAIN LGBM (3 seeds)
# ============================================================
print(f"\nTraining LightGBM h={H} (3 seeds)...")
hp = {**BASE_PARAMS, **HORIZON_PARAMS}
es = hp.pop('early_stopping')

lgbm_val_preds  = np.zeros(len(X_val))
lgbm_test_preds = np.zeros(len(X_test))
best_iters = []

for seed in [42, 2024, 777]:
    evals_result = {}
    m = lgb.LGBMRegressor(**hp, random_state=seed)
    m.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_sample_weight=[w_train, w_val],
        eval_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(es, verbose=False),
            lgb.log_evaluation(200),
            lgb.record_evaluation(evals_result),
        ],
    )
    best_iters.append(m.best_iteration_)
    lgbm_val_preds  += m.predict(X_val)  / 3
    lgbm_test_preds += m.predict(X_test) / 3
    print(f"  seed={seed}  best_iter={m.best_iteration_}")

print(f"  Avg best_iter: {np.mean(best_iters):.0f}")

lgbm_score = skill_score(y_val, lgbm_val_preds, w_val)
print(f"  LightGBM val score: {lgbm_score:.4f}")

# ============================================================
# LOAD NN PREDICTIONS
# ============================================================
# Val preds — directly saved by autoencoder script
nn_val_file  = AE_DIR / f"val_preds_h{H}_nn_cv0.2517.csv"
nn_val_df    = pd.read_csv(nn_val_file)
nn_val_df    = nn_val_df.set_index('id').reindex(val_df['id'].values).reset_index()
nn_val_preds = nn_val_df['y_pred_nn'].values

# Test preds — recover from saved blend + lgbm files (alpha=0.90 → nn = (blend - 0.10*lgbm) / 0.90)
PREV_ALPHA    = 0.90
blend_test_df = pd.read_csv(AE_DIR / f"test_preds_h{H}_blend_a{PREV_ALPHA:.2f}_cv0.2528.csv")
lgbm_test_df  = pd.read_csv(AE_DIR / f"test_preds_h{H}_lgbm_cv0.1616.csv")
nn_test_preds = (blend_test_df['prediction'].values - (1 - PREV_ALPHA) * lgbm_test_df['prediction'].values) / PREV_ALPHA

nn_score = skill_score(y_val, nn_val_preds, w_val)
print(f"  NN val score:       {nn_score:.4f}")

# ============================================================
# BLEND SEARCH
# ============================================================
print(f"\nBlend search (alpha = NN weight):")
print(f"  {'alpha':>6}  {'score':>8}")
print(f"  {'------':>6}  {'--------':>8}")

results = []
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = (1 - alpha) * lgbm_val_preds + alpha * nn_val_preds
    score   = skill_score(y_val, blended, w_val)
    results.append((alpha, score))
    print(f"  {alpha:6.2f}  {score:8.4f}")

best_alpha, best_score = max(results, key=lambda x: x[1])
print(f"\nBest blend: alpha={best_alpha:.2f}  score={best_score:.4f}")
print(f"LightGBM alone: {lgbm_score:.4f}  NN alone: {nn_score:.4f}")
print(f"Delta vs LightGBM: {best_score - lgbm_score:+.4f}")

# ============================================================
# SAVE
# ============================================================
# Val predictions
val_out = val_df[['id', 'ts_index']].copy().reset_index(drop=True)
val_out['y_true']       = y_val
val_out['y_pred_lgbm']  = lgbm_val_preds
val_out['y_pred_nn']    = nn_val_preds
val_out['y_pred_blend'] = (1 - best_alpha) * lgbm_val_preds + best_alpha * nn_val_preds
val_out['weight']       = w_val
val_out.to_csv(OUT_DIR / f"val_preds_h{H}_blend_cv{best_score:.4f}.csv", index=False)

# Test predictions
best_test_blend = (1 - best_alpha) * lgbm_test_preds + best_alpha * nn_test_preds
te_out = te[['id']].copy().reset_index(drop=True)
te_out['prediction'] = best_test_blend
te_out.to_csv(OUT_DIR / f"test_preds_h{H}_blend_a{best_alpha:.2f}_cv{best_score:.4f}.csv", index=False)
print(f"\nVal and test blend preds saved.")
