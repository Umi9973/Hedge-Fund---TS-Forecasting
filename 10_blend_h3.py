"""
Step 10: Blend LightGBM + Autoencoder NN for h=3
- Retrain h=3 LightGBM (3-seed ensemble) with Optuna params
- Load best NN val/test predictions
- Test blend ratios 0.0 to 1.0 (step 0.05) on val score
- Save best blend submission
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
AE_DIR      = PROJECT_DIR / "autoencoder_results"
OUT_DIR     = PROJECT_DIR / "autoencoder_results"

VAL_SPLIT = 3500
SEED      = 42
np.random.seed(SEED)

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

# Optuna-tuned params for h=3
LGBM_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'bagging_freq': 5, 'lambda_l1': 0.5, 'verbosity': -1, 'n_jobs': -1,
    'num_leaves': 33,  'min_child_samples': 300, 'lambda_l2': 21.435,
    'max_depth': 10,   'learning_rate': 0.03794, 'feature_fraction': 0.718,
    'bagging_fraction': 0.745, 'early_stopping_rounds': 237,
}

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")

drop_feature_cols = []
for f in LOW_IC_FEATURES:
    drop_feature_cols += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feature_cols = [c for c in drop_feature_cols if c in train.columns]
train = train.drop(columns=drop_feature_cols)
test  = test.drop(columns=[c for c in drop_feature_cols if c in test.columns])
feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Features: {len(feat_cols)}")

tr = train[train.horizon == 3].copy()
te = test[test.horizon == 3].copy()
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
print("\nTraining LightGBM h=3 (3 seeds)...")
lgbm_val_preds  = np.zeros(len(X_val))
lgbm_test_preds = np.zeros(len(X_test))

for seed in [42, 2024, 777]:
    params = {**LGBM_PARAMS, 'seed': seed}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=237, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    lgbm_val_preds  += model.predict(X_val)  / 3
    lgbm_test_preds += model.predict(X_test) / 3
    print(f"  seed={seed}  best_iter={model.best_iteration_}")

lgbm_score = skill_score(y_val, lgbm_val_preds, w_val)
print(f"  LightGBM val score: {lgbm_score:.4f}")

# ============================================================
# LOAD NN PREDICTIONS
# ============================================================
nn_val_file = sorted(AE_DIR.glob("val_preds_h3_nn_cv*.csv"))[-1]  # latest/best
nn_val_df   = pd.read_csv(nn_val_file)
nn_val_df   = nn_val_df.sort_values('ts_index')  # align order

nn_test_file  = AE_DIR / "test_preds_h3_nn.csv"
nn_test_df    = pd.read_csv(nn_test_file)

# Align val preds to same row order as val_df
val_df_sorted = val_df.reset_index(drop=True)
nn_val_df     = nn_val_df.set_index('id').reindex(val_df['id'].values).reset_index()
nn_val_preds  = nn_val_df['y_pred_nn'].values

nn_score = skill_score(y_val, nn_val_preds, w_val)
print(f"  NN val score:       {nn_score:.4f}")
print(f"  (loaded from: {nn_val_file.name})")

# ============================================================
# BLEND SEARCH: 0% to 100% NN, step 5%
# ============================================================
print("\nBlend search (alpha = NN weight):")
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
# SAVE BEST BLEND TEST PREDICTIONS
# ============================================================
best_blended_test = (1 - best_alpha) * lgbm_test_preds + best_alpha * nn_test_df['prediction'].values

te_out = te[['id']].copy().reset_index(drop=True)
te_out['prediction'] = best_blended_test
te_out.to_csv(OUT_DIR / f"test_preds_h3_blend_a{best_alpha:.2f}_cv{best_score:.4f}.csv", index=False)

print(f"\nBest blend test preds saved.")
