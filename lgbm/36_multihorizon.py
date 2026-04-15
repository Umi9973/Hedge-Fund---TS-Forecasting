"""
36_multihorizon.py

Single LightGBM model trained on ALL 4 horizons simultaneously.
- horizon encoded as one-hot feature (4 binary columns)
- VAL_SPLIT = 3400 (200-timestamp val window instead of 100)
- 5 seeds instead of 3
- v2 features (384 + horizon one-hot = 388 features)
- Params: averaged/middle-ground from Optuna per-horizon params

Key hypothesis: shared representation across horizons lets model learn
cross-horizon patterns (e.g. feature X predicts well at h=3 AND h=25).

Baselines (single-horizon, 3 seeds, VAL_SPLIT=3500):
  h=1: 0.0498 | h=3: 0.1211 | h=10: 0.2267 | h=25: 0.2903
  Combined CV: 0.2454
  Best LB: 0.2438 (optuna baseline)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "multihorizon_results"
OUT_DIR.mkdir(exist_ok=True)

VAL_SPLIT = 3400   # 200-timestamp val window (was 3500=100 timestamps)
SEEDS     = [42, 2024, 777, 123, 999]

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category',
             'ts_index', 'y_target', 'weight']

# Middle-ground params across all 4 horizon-specific Optuna results
# depth/leaves: between h=3 (depth=10, leaves=33) and h=25 (depth=9, leaves=120)
# lr: slow like h=10/25 since we have 4x more data
# regularization: moderate
LGBM_PARAMS = {
    'boosting_type':    'gbdt',
    'objective':        'regression',
    'metric':           'rmse',
    'n_estimators':     5000,
    'learning_rate':    0.01,
    'max_depth':        9,
    'num_leaves':       80,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq':     5,
    'lambda_l1':        0.5,
    'lambda_l2':        5.0,
    'min_data_in_leaf': 200,
    'verbose':          -1,
    'n_jobs':           -1,
}
EARLY_STOPPING = 200
LOG_PERIOD     = 500


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


# ── Load ─────────────────────────────────────────────────────
print("Loading v2 features...")
train = pd.read_parquet(PROJECT_DIR / "train_features_v2.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features_v2.parquet")
print(f"  Train: {train.shape} | Test: {test.shape}")

# ── One-hot encode horizon ────────────────────────────────────
for h in [1, 3, 10, 25]:
    train[f'horizon_{h}'] = (train['horizon'] == h).astype(np.int8)
    test[f'horizon_{h}']  = (test['horizon']  == h).astype(np.int8)

# horizon column itself stays in DROP_COLS (raw int not needed, we have one-hots)
DROP_COLS = DROP_COLS + ['horizon']

feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Total features: {len(feat_cols)} (384 v2 + 4 horizon one-hots)")

# ── Train/val split ───────────────────────────────────────────
tr = train[train.ts_index <= VAL_SPLIT]
va = train[train.ts_index >  VAL_SPLIT]

print(f"\n  VAL_SPLIT={VAL_SPLIT} | Train: {len(tr):,} | Val: {len(va):,} | Test: {len(test):,}")
print(f"  Val ts_index range: {va.ts_index.min()} – {va.ts_index.max()}")
print(f"  Val horizons: {sorted(va.horizon.unique())}")

X_tr, y_tr, w_tr = tr[feat_cols].values.astype(np.float32), tr['y_target'].values, tr['weight'].values
X_va, y_va, w_va = va[feat_cols].values.astype(np.float32), va['y_target'].values, va['weight'].values
X_te               = test[feat_cols].values.astype(np.float32)

# ── Train 5 seeds ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Multi-horizon LightGBM | all 4 horizons | {len(SEEDS)} seeds | CPU")
print(f"  VAL_SPLIT={VAL_SPLIT} | n_estimators={LGBM_PARAMS['n_estimators']} | lr={LGBM_PARAMS['learning_rate']}")
print(f"{'='*60}")

val_preds_list  = []
test_preds_list = []
val_scores      = []

dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
dval   = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)

for seed in SEEDS:
    print(f"\n  --- seed={seed} ---")
    params = {**LGBM_PARAMS, 'seed': seed}

    model = lgb.train(
        params, dtrain,
        num_boost_round=LGBM_PARAMS['n_estimators'],
        valid_sets=[dval], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=LOG_PERIOD),
        ],
    )

    val_pred  = model.predict(X_va)
    test_pred = model.predict(X_te)
    score     = skill_score(y_va, val_pred, w_va)
    print(f"  best_iter={model.best_iteration}  combined_val={score:.4f}")

    # Per-horizon breakdown
    for h in [1, 3, 10, 25]:
        mask = va['horizon'] == h
        h_score = skill_score(y_va[mask], val_pred[mask], w_va[mask])
        print(f"    h={h:2d}: {h_score:.4f}")

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)
    val_scores.append(score)

# ── Ensemble ──────────────────────────────────────────────────
val_pred_avg  = np.mean(val_preds_list, axis=0)
test_pred_avg = np.mean(test_preds_list, axis=0)
score_avg     = skill_score(y_va, val_pred_avg, w_va)

print(f"\n{'='*60}")
print(f"RESULTS — Multi-horizon LightGBM (VAL_SPLIT={VAL_SPLIT}, 5 seeds)")
print(f"{'='*60}")
print(f"  Per-seed scores  : {[f'{s:.4f}' for s in val_scores]}")
print(f"  Ensemble val     : {score_avg:.4f}")
print(f"  Baseline combined: 0.2454  ({score_avg - 0.2454:+.4f})")
print(f"  Best LB          : 0.2438")

print(f"\n  Per-horizon ensemble breakdown:")
for h in [1, 3, 10, 25]:
    mask = va['horizon'] == h
    h_score = skill_score(y_va[mask], val_pred_avg[mask], w_va[mask])
    print(f"    h={h:2d}: {h_score:.4f}")

# ── Save ──────────────────────────────────────────────────────
val_out           = va[['id', 'ts_index', 'y_target', 'weight', 'horizon']].copy()
val_out['y_pred'] = val_pred_avg
val_out.to_csv(OUT_DIR / f"val_preds_multihorizon_cv{score_avg:.4f}.csv", index=False)

# Split test preds by horizon for potential per-horizon blending later
test_out               = test[['id', 'horizon']].copy()
test_out['prediction'] = test_pred_avg
test_out.to_csv(OUT_DIR / f"test_preds_multihorizon_cv{score_avg:.4f}.csv", index=False)

# Also save submission-ready file
sub = test_out[['id', 'prediction']].sort_values('id').reset_index(drop=True)
sub.to_csv(OUT_DIR / f"submission_multihorizon_cv{score_avg:.4f}.csv", index=False)

print(f"\n  Saved to multihorizon_results/")
print(f"  submission_multihorizon_cv{score_avg:.4f}.csv ready to submit")
