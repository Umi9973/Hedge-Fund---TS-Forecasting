"""
33_skewkurt_tuned_h25.py

LightGBM GBDT h=25 with v2 features (384) but tighter regularization:
- lambda_l2: 6.81 -> 15.0  (stronger L2)
- min_data_in_leaf: 392 -> 600  (larger leaves)
- feature_fraction: 0.737 -> 0.55  (more selective)
- Only top 10 skew/kurt features (drop bottom 10 IC)

Previous result (loose): val=0.2983, LB hurt (part of 0.2400 submission)
Baseline (no skew/kurt):  val=0.2535
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR     = PROJECT_DIR / "dart_results"
OUT_DIR.mkdir(exist_ok=True)

VAL_SPLIT = 3500
H         = 25
SEEDS     = [42, 2024, 777]

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
             'ts_index', 'y_target', 'weight']

DROP_SKEWKURT = [
    f'{feat}_{stat}'
    for feat in ['feature_bq', 'feature_ag', 'feature_ap', 'feature_br', 'feature_bp',
                 'feature_bs', 'feature_bn', 'feature_bo', 'feature_al', 'feature_an']
    for stat in ['cs_skew', 'cs_kurt']
]

# Tighter regularization for 384-feature set
LGBM_PARAMS = {
    'boosting_type':    'gbdt',
    'objective':        'regression',
    'metric':           'rmse',
    'n_estimators':     3000,
    'learning_rate':    0.00519,
    'max_depth':        9,
    'num_leaves':       120,
    'feature_fraction': 0.55,     # was 0.737
    'bagging_fraction': 0.599,
    'bagging_freq':     5,
    'lambda_l1':        0.5,
    'lambda_l2':        15.0,     # was 6.81
    'min_data_in_leaf': 600,      # was 392
    'verbose':          -1,
    'n_jobs':           -1,
}
EARLY_STOPPING = 192
LOG_PERIOD     = 200


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


print("Loading v2 features...")
train = pd.read_parquet(PROJECT_DIR / "train_features_v2.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features_v2.parquet")
print(f"  Train: {train.shape} | Test: {test.shape}")

train_h = train[train.horizon == H].copy()
test_h  = test[test.horizon == H].copy()
del train, test

train_h = train_h.drop(columns=[c for c in DROP_SKEWKURT if c in train_h.columns])
test_h  = test_h.drop(columns=[c for c in DROP_SKEWKURT if c in test_h.columns])

feat_cols = [c for c in train_h.columns if c not in DROP_COLS]
skew_kurt_kept = [c for c in feat_cols if 'cs_skew' in c or 'cs_kurt' in c]
print(f"  h={H} total features   : {len(feat_cols)}")
print(f"  Skew/kurt kept         : {len(skew_kurt_kept)} (top 10 features only)")

tr = train_h[train_h.ts_index <= VAL_SPLIT]
va = train_h[train_h.ts_index >  VAL_SPLIT]

X_tr, y_tr, w_tr = tr[feat_cols].values.astype(np.float32), tr['y_target'].values, tr['weight'].values
X_va, y_va, w_va = va[feat_cols].values.astype(np.float32), va['y_target'].values, va['weight'].values
X_te               = test_h[feat_cols].values.astype(np.float32)

print(f"  Train: {len(tr):,} | Val: {len(va):,} | Test: {len(test_h):,}")

print(f"\n{'='*60}")
print(f"LightGBM GBDT h={H} | v2 tuned | {len(SEEDS)} seeds | CPU")
print(f"  lambda_l2={LGBM_PARAMS['lambda_l2']} | min_leaf={LGBM_PARAMS['min_data_in_leaf']} | feat_frac={LGBM_PARAMS['feature_fraction']}")
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
    print(f"  best_iter={model.best_iteration}  val_score={score:.4f}")

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)
    val_scores.append(score)

val_pred_avg  = np.mean(val_preds_list, axis=0)
test_pred_avg = np.mean(test_preds_list, axis=0)
score_avg     = skill_score(y_va, val_pred_avg, w_va)

print(f"\n{'='*60}")
print(f"RESULTS — h={H} LGBM tuned + CS skew/kurt top10")
print(f"{'='*60}")
print(f"  Per-seed scores  : {[f'{s:.4f}' for s in val_scores]}")
print(f"  Ensemble val     : {score_avg:.4f}")
print(f"  Loose skew/kurt  : 0.2983  ({score_avg - 0.2983:+.4f})")
print(f"  Baseline LGBM    : 0.2535  ({score_avg - 0.2535:+.4f})")

val_out           = va[['id', 'ts_index', 'y_target', 'weight']].copy()
val_out['y_pred'] = val_pred_avg
val_out.to_csv(OUT_DIR / f"val_preds_h{H}_skewkurt_tuned_cv{score_avg:.4f}.csv", index=False)

test_out               = test_h[['id']].copy()
test_out['prediction'] = test_pred_avg
test_out.to_csv(OUT_DIR / f"test_preds_h{H}_skewkurt_tuned_cv{score_avg:.4f}.csv", index=False)
print(f"  Saved to dart_results/")
