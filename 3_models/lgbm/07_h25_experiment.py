"""
Experiment: sub_code as native LightGBM categorical on h=25
Baseline h=25 val score: 0.2860
Using baseline params for h=25 from HORIZON_PARAMS.
Outputs saved to: h25_experiment_submission/
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "h25_experiment_submission"
OUT_DIR.mkdir(exist_ok=True)

VAL_SPLIT = 3500
SEEDS     = [42, 2024, 777]

DROP_COLS = [
    'id', 'code', 'sub_code', 'sub_category', 'horizon',
    'ts_index', 'y_target', 'weight'
]
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

# ============================================================
# LOAD
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

# Add sub_code as native LightGBM categorical
feat_cols = [c for c in train.columns if c not in DROP_COLS] + ['sub_code']
train['sub_code'] = train['sub_code'].astype('category')
test['sub_code']  = test['sub_code'].astype('category')
print(f"  Feature columns: {len(feat_cols)} (includes sub_code as categorical)")

# ============================================================
# H=25 ONLY
# ============================================================
start = time.time()
tr       = train[train.horizon == 25]
te       = test[test.horizon == 25]
train_df = tr[tr.ts_index <= VAL_SPLIT]
val_df   = tr[tr.ts_index  > VAL_SPLIT]

X_train = train_df[feat_cols]
X_val   = val_df[feat_cols]
X_test  = te[feat_cols]
y_train = train_df['y_target']
y_val   = val_df['y_target']
w_train = train_df['weight'].values
w_val   = val_df['weight'].values

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

# Baseline params for h=25 (from HORIZON_PARAMS in 04_model.py)
hp = {
    'objective':         'regression',
    'metric':            'rmse',
    'learning_rate':     0.02,
    'n_estimators':      3000,
    'num_leaves':        70,
    'min_child_samples': 220,
    'lambda_l2':         11.0,
    'max_depth':         8,
    'feature_fraction':  0.5,
    'bagging_fraction':  0.6,
    'bagging_freq':      5,
    'lambda_l1':         0.5,
    'verbosity':         -1,
    'n_jobs':            -1,
}
es = 120

# ============================================================
# TRAIN (val score)
# ============================================================
print(f"\nTraining h=25...")
val_preds  = np.zeros(len(X_val))
best_iters = []

for seed in SEEDS:
    model = lgb.LGBMRegressor(**hp, random_state=seed)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(es, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    val_preds  += model.predict(X_val) / len(SEEDS)
    best_iters.append(model.best_iteration_)
    print(f"  seed={seed}  best_iter={model.best_iteration_}")

score    = skill_score(y_val.values, val_preds, w_val)
avg_iter = int(np.mean(best_iters) * 1.1)
print(f"\nBest iters   : {best_iters} -> refit with {avg_iter}")
print(f"h=25 Val Score : {score:.4f}  (baseline: 0.2860)")

# ============================================================
# FULL REFIT + TEST PREDICTIONS
# ============================================================
print(f"\nFull-data refit ({len(tr):,} rows)...")
X_full      = tr[feat_cols]
y_full      = tr['y_target']
w_full      = tr['weight'].values
final_preds = np.zeros(len(X_test))

full_hp = {**hp, 'n_estimators': avg_iter}
for seed in SEEDS:
    m = lgb.LGBMRegressor(**full_hp, random_state=seed)
    m.fit(X_full, y_full, sample_weight=w_full)
    final_preds += m.predict(X_test) / len(SEEDS)

te = te.copy()
te['prediction'] = final_preds

# ============================================================
# SAVE
# ============================================================
val_out = val_df[['id', 'ts_index']].copy()
val_out['y_true'] = y_val.values
val_out['y_pred'] = val_preds
val_out['weight'] = w_val
val_out.to_csv(OUT_DIR / f"val_preds_h25_cv{score:.4f}.csv", index=False)

sub = te[['id', 'prediction']].sort_values('id').reset_index(drop=True)
sub.to_csv(OUT_DIR / f"submission_h25_cv{score:.4f}.csv", index=False)

with open(OUT_DIR / "results.txt", "w") as f:
    f.write(f"h=25 Experiment Results\n")
    f.write(f"======================\n")
    f.write(f"Val Score    : {score:.4f}  (baseline: 0.2860)\n")
    f.write(f"Best iters   : {best_iters} -> refit {avg_iter}\n")
    f.write(f"Feature cols : {len(feat_cols)}\n")
    f.write(f"Runtime      : {(time.time()-start)/60:.1f} min\n")

print(f"\nResults saved to: {OUT_DIR}")
print(f"Runtime: {(time.time()-start)/60:.1f} min")
