"""
Experiment: Era boosting on h=3
- Train base model with Optuna params
- Iteratively add 5 trees on worst 50% of training dates, 15 iterations
- Baseline h=3 val score (Optuna params, 3-seed): 0.1103
Outputs saved to: h3_experiment_submission/
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "h3_experiment_submission"
OUT_DIR.mkdir(exist_ok=True)

VAL_SPLIT      = 3500
SEEDS          = [42, 2024, 777]
ERA_BOOST_ITERS = 15   # iterations of era boosting
TREES_PER_ITER  = 5    # trees added per iteration (gentle for noisy h=3)
WORST_PCT       = 0.5  # retrain on worst 50% of dates

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

def skill_score_per_date(df, pred_col):
    """Compute skill score per ts_index, return series indexed by ts_index."""
    scores = {}
    for ts, grp in df.groupby('ts_index'):
        scores[ts] = skill_score(grp['y_true'].values, grp[pred_col].values, grp['weight'].values)
    return pd.Series(scores)

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
feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Feature columns: {len(feat_cols)}")

# ============================================================
# H=3 ONLY
# ============================================================
start = time.time()
tr       = train[train.horizon == 3]
te       = test[test.horizon == 3]
train_df = tr[tr.ts_index <= VAL_SPLIT].copy()
val_df   = tr[tr.ts_index  > VAL_SPLIT].copy()

X_train = train_df[feat_cols]
X_val   = val_df[feat_cols]
X_test  = te[feat_cols]
y_train = train_df['y_target']
y_val   = val_df['y_target']
w_train = train_df['weight'].values
w_val   = val_df['weight'].values

# Optuna-tuned params for h=3
hp = {
    'objective':         'regression',
    'metric':            'rmse',
    'learning_rate':     0.03794,
    'n_estimators':      3000,
    'num_leaves':        33,
    'min_child_samples': 300,
    'lambda_l2':         21.435,
    'max_depth':         10,
    'feature_fraction':  0.718,
    'bagging_fraction':  0.745,
    'bagging_freq':      5,
    'lambda_l1':         0.5,
    'verbosity':         -1,
    'n_jobs':            -1,
}
es = 237

# ============================================================
# STEP 1: BASE MODEL (3-seed ensemble with early stopping)
# ============================================================
print(f"\nTraining base model h=3 ({len(X_train):,} train | {len(X_val):,} val)...")
val_preds  = np.zeros(len(X_val))
best_iters = []
base_models = []

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
    val_preds += model.predict(X_val) / len(SEEDS)
    best_iters.append(model.best_iteration_)
    base_models.append(model)
    print(f"  seed={seed}  best_iter={model.best_iteration_}")

base_score = skill_score(y_val.values, val_preds, w_val)
print(f"\nBase model val score: {base_score:.4f}  (Optuna baseline: 0.1103)")

# ============================================================
# STEP 2: ERA BOOSTING
# Iteratively add trees trained on worst-performing dates
# ============================================================
print(f"\nEra boosting: {ERA_BOOST_ITERS} iters x {TREES_PER_ITER} trees on worst {int(WORST_PCT*100)}% dates...")

# Prepare train_df with predictions for per-date scoring
train_score_df = train_df[['ts_index', 'y_target', 'weight']].copy()
train_score_df.columns = ['ts_index', 'y_true', 'weight']

# Era boost params — minimal trees per step, no early stopping
era_hp = {**hp, 'n_estimators': TREES_PER_ITER}
era_hp.pop('metric', None)   # not needed without eval_set

boosted_models = list(base_models)  # start from base models
era_val_preds  = val_preds.copy()

for iteration in range(ERA_BOOST_ITERS):
    # Compute per-date train predictions (ensemble of current models)
    train_preds_era = np.zeros(len(X_train))
    for m in boosted_models[-len(SEEDS):]:   # use most recent seed ensemble
        train_preds_era += m.predict(X_train) / len(SEEDS)

    train_score_df['y_pred'] = train_preds_era
    date_scores = skill_score_per_date(train_score_df, 'y_pred')

    # Select worst 50% of dates
    threshold   = date_scores.quantile(WORST_PCT)
    worst_dates = date_scores[date_scores <= threshold].index
    worst_mask  = train_df['ts_index'].isin(worst_dates)

    X_worst = X_train[worst_mask]
    y_worst = y_train[worst_mask]
    w_worst = w_train[worst_mask]

    # Add trees on worst dates for each seed
    new_models = []
    new_val_preds = np.zeros(len(X_val))
    for i, seed in enumerate(SEEDS):
        m_new = lgb.LGBMRegressor(**era_hp, random_state=seed)
        m_new.fit(
            X_worst, y_worst,
            sample_weight=w_worst,
            init_model=boosted_models[-(len(SEEDS) - i)].booster_,
        )
        new_models.append(m_new)
        new_val_preds += m_new.predict(X_val) / len(SEEDS)

    era_score = skill_score(y_val.values, new_val_preds, w_val)
    print(f"  iter {iteration+1:2d}/{ERA_BOOST_ITERS}  worst_dates={len(worst_dates)}  val={era_score:.4f}")

    boosted_models = new_models
    era_val_preds  = new_val_preds

final_score = skill_score(y_val.values, era_val_preds, w_val)
print(f"\nFinal val score : {final_score:.4f}  (base: {base_score:.4f}  delta: {final_score-base_score:+.4f})")

# ============================================================
# FULL REFIT + TEST PREDICTIONS
# Use avg_iter from base model * 1.1, then era boost on full data
# ============================================================
avg_iter = int(np.mean(best_iters) * 1.1)
print(f"\nFull-data refit ({len(tr):,} rows, {avg_iter} trees)...")

X_full = tr[feat_cols]
y_full = tr['y_target']
w_full = tr['weight'].values
full_score_df = tr[['ts_index', 'y_target', 'weight']].copy()
full_score_df.columns = ['ts_index', 'y_true', 'weight']

# Base refit
full_hp = {**hp, 'n_estimators': avg_iter}
full_models = []
for seed in SEEDS:
    m = lgb.LGBMRegressor(**full_hp, random_state=seed)
    m.fit(X_full, y_full, sample_weight=w_full)
    full_models.append(m)

# Era boost on full data
era_full_hp = {**hp, 'n_estimators': TREES_PER_ITER}
era_full_hp.pop('metric', None)

for iteration in range(ERA_BOOST_ITERS):
    full_preds_era = np.zeros(len(X_full))
    for m in full_models[-len(SEEDS):]:
        full_preds_era += m.predict(X_full) / len(SEEDS)

    full_score_df['y_pred'] = full_preds_era
    date_scores  = skill_score_per_date(full_score_df, 'y_pred')
    threshold    = date_scores.quantile(WORST_PCT)
    worst_dates  = date_scores[date_scores <= threshold].index
    worst_mask   = tr['ts_index'].isin(worst_dates)

    X_worst = X_full[worst_mask]
    y_worst = y_full[worst_mask]
    w_worst = w_full[worst_mask]

    new_full_models = []
    for i, seed in enumerate(SEEDS):
        m_new = lgb.LGBMRegressor(**era_full_hp, random_state=seed)
        m_new.fit(
            X_worst, y_worst,
            sample_weight=w_worst,
            init_model=full_models[-(len(SEEDS) - i)].booster_,
        )
        new_full_models.append(m_new)
    full_models = new_full_models

# Final test predictions
final_preds = np.zeros(len(X_test))
for m in full_models:
    final_preds += m.predict(X_test) / len(SEEDS)

te = te.copy()
te['prediction'] = final_preds

# ============================================================
# SAVE
# ============================================================
val_out = val_df[['id', 'ts_index']].copy()
val_out['y_true'] = y_val.values
val_out['y_pred'] = era_val_preds
val_out['weight'] = w_val
val_out.to_csv(OUT_DIR / f"val_preds_h3_cv{final_score:.4f}.csv", index=False)

sub = te[['id', 'prediction']].sort_values('id').reset_index(drop=True)
sub.to_csv(OUT_DIR / f"submission_h3_cv{final_score:.4f}.csv", index=False)

with open(OUT_DIR / "results.txt", "w") as f:
    f.write(f"h=3 Era Boosting Results\n")
    f.write(f"========================\n")
    f.write(f"Base val score   : {base_score:.4f}\n")
    f.write(f"Final val score  : {final_score:.4f}  (delta: {final_score-base_score:+.4f})\n")
    f.write(f"Era boost iters  : {ERA_BOOST_ITERS} x {TREES_PER_ITER} trees\n")
    f.write(f"Worst pct        : {int(WORST_PCT*100)}%\n")
    f.write(f"Runtime          : {(time.time()-start)/60:.1f} min\n")

print(f"\nResults saved to: {OUT_DIR}")
print(f"Runtime: {(time.time()-start)/60:.1f} min")
