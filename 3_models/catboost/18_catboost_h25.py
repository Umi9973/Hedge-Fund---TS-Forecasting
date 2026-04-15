"""
CatBoost — Horizon 25 (standalone, no blending)
- 3-seed ensemble with early stopping
- Progress: verbose every 200 iters (learn + val RMSE live)
- Overfitting indicators: skill score gap, val/train RMSE ratio plot
- Saves val preds and test preds
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from pathlib import Path

plt.style.use('dark_background')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")

H         = 25
VAL_SPLIT = 3500
SEEDS     = [42, 2024, 777]

CB_PARAMS = {
    'iterations':            3000,
    'learning_rate':         0.01,
    'depth':                 6,
    'l2_leaf_reg':           5.0,
    'min_data_in_leaf':      100,
    'subsample':             0.8,
    'bootstrap_type':        'Bernoulli',
    'early_stopping_rounds': 250,
    'verbose':               200,
    'loss_function':         'RMSE',
    'eval_metric':           'RMSE',
    'task_type':             'GPU',
}

DROP_COLS = ['id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'y_target', 'weight']
LOW_IC_FEATURES = ['feature_b', 'feature_c', 'feature_d', 'feature_e',
                   'feature_f', 'feature_g', 'feature_h', 'feature_i']


def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0:
        return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


print(f"\n{'='*60}")
print(f"CatBoost — Horizon {H}")
print(f"{'='*60}")

print(f"\nLoading data (h={H})...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")

drop_feat = []
for f in LOW_IC_FEATURES:
    drop_feat += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feat = [c for c in drop_feat if c in train.columns]
train = train.drop(columns=drop_feat)
test  = test.drop(columns=[c for c in drop_feat if c in test.columns])

feat_cols = [c for c in train.columns if c not in DROP_COLS]

tr = train[train.horizon == H].copy()
te = test[test.horizon == H].copy()

train_df = tr[tr.ts_index <= VAL_SPLIT]
val_df   = tr[tr.ts_index  > VAL_SPLIT]

X_train, y_train = train_df[feat_cols], train_df['y_target']
w_train = train_df['weight'].values
X_val,   y_val   = val_df[feat_cols],   val_df['y_target']
w_val   = val_df['weight'].values
X_test  = te[feat_cols]
X_full, y_full, w_full = tr[feat_cols], tr['y_target'], tr['weight'].values

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"  Features: {len(feat_cols)}")

print(f"\nTraining CatBoost h={H} (3 seeds)...")
print("  [verbose: Iter | learn RMSE | val RMSE | best | time]")

val_preds   = np.zeros(len(X_val))
train_preds = np.zeros(len(X_train))
test_preds  = np.zeros(len(X_test))
best_iters  = []
lc_train = lc_val = None

for seed in SEEDS:
    print(f"\n  --- seed={seed} ---")
    model = CatBoostRegressor(**{**CB_PARAMS, 'random_seed': seed})
    model.fit(
        Pool(X_train, y_train, weight=w_train),
        eval_set=Pool(X_val, y_val, weight=w_val),
        use_best_model=True,
    )
    best_iter = model.best_iteration_
    best_iters.append(best_iter)
    val_preds   += model.predict(X_val)   / len(SEEDS)
    train_preds += model.predict(X_train) / len(SEEDS)
    test_preds  += model.predict(X_test)  / len(SEEDS)
    print(f"  seed={seed}  best_iter={best_iter}")

    evals   = model.get_evals_result()
    tr_rmse = np.array(evals['learn']['RMSE'])
    v_rmse  = np.array(evals['validation']['RMSE'])
    if lc_train is None:
        lc_train, lc_val = tr_rmse.copy(), v_rmse.copy()
    else:
        l = min(len(tr_rmse), len(lc_train))
        lc_train = (lc_train[:l] + tr_rmse[:l]) / 2
        lc_val   = (lc_val[:l]   + v_rmse[:l])  / 2
    del model; gc.collect()

val_score   = skill_score(y_val.values,   val_preds,   w_val)
train_score = skill_score(y_train.values, train_preds, w_train)
gap = train_score - val_score
avg_iter = int(np.mean(best_iters) * 1.1)

print(f"\n  Avg best_iter: {int(np.mean(best_iters))} -> refit with {avg_iter}")
print(f"  Train score     : {train_score:.4f}")
print(f"  Val score       : {val_score:.4f}")
print(f"  Gap (train-val) : {gap:.4f}  {'! possible overfit' if gap > 0.05 else 'OK'}")
print(f"  Val/Train RMSE  : {lc_val[-1]/lc_train[-1]:.3f}  (>1.3 = likely overfit)")

print(f"\n  Full-data refit ({len(tr):,} rows)...")
final_preds = np.zeros(len(X_test))
for seed in SEEDS:
    params_refit = {**CB_PARAMS, 'random_seed': seed, 'iterations': avg_iter, 'verbose': 0}
    params_refit.pop('early_stopping_rounds', None)
    m = CatBoostRegressor(**params_refit)
    m.fit(Pool(X_full, y_full, weight=w_full))
    final_preds += m.predict(X_test) / len(SEEDS)
    del m; gc.collect()

pd.DataFrame({
    'ts_index': val_df['ts_index'].values,
    'y_true':   y_val.values,
    'y_pred':   val_preds,
    'weight':   w_val,
}).to_csv(PROJECT_DIR / f"val_preds_h{H}_cb_cv{val_score:.4f}.csv", index=False)
print(f"\nVal preds saved: val_preds_h{H}_cb_cv{val_score:.4f}.csv")

test_out = te[['id']].copy()
test_out['prediction'] = final_preds
test_out.to_csv(PROJECT_DIR / f"test_preds_h{H}_cb_cv{val_score:.4f}.csv", index=False)
print(f"Test preds saved: test_preds_h{H}_cb_cv{val_score:.4f}.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
iters = np.arange(1, len(lc_train) + 1)
axes[0].plot(iters, lc_train, color='#2a7fbf', linewidth=1.0, label='Train RMSE')
axes[0].plot(iters, lc_val,   color='#ff6b6b', linewidth=1.0, label='Val RMSE')
axes[0].set_title(f'CatBoost h={H} — Learning Curve', fontweight='bold')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('RMSE')
axes[0].legend(); axes[0].grid(True, alpha=0.2)

ratio = lc_val / np.maximum(lc_train, 1e-12)
axes[1].plot(iters, ratio, color='#ffcc44', linewidth=1.0, label='Val/Train ratio')
axes[1].axhline(1.0, color='white',   linewidth=0.8, linestyle='--', alpha=0.5, label='ratio=1.0')
axes[1].axhline(1.3, color='#ff6b6b', linewidth=0.8, linestyle=':', alpha=0.7, label='ratio=1.3 (overfit)')
axes[1].set_title('Val/Train RMSE Ratio (overfit detector)', fontweight='bold')
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Ratio')
axes[1].legend(); axes[1].grid(True, alpha=0.2)

plt.suptitle(f'h={H} CatBoost | Train={train_score:.4f} Val={val_score:.4f} Gap={gap:.4f}',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_DIR / f"cb_lc_h{H}_cv{val_score:.4f}.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"Plot saved: cb_lc_h{H}_cv{val_score:.4f}.png")
