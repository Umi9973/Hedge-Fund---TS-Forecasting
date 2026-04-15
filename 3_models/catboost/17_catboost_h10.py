"""
CatBoost blend — Horizon 10
- Trains LightGBM + CatBoost on train split (ts_index <= 3500)
- Progress: CatBoost verbose every 200 iters (shows learn + val RMSE live)
- Overfitting indicators: skill score gap, val/train RMSE ratio plot
- Blend search, full-data refit, saves val + test preds
Note: LGB early_stopping updated to 300 (was 145, hit 3000-tree cap)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from pathlib import Path

plt.style.use('dark_background')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")

H         = 10
VAL_SPLIT = 3500
SEEDS     = [42, 2024, 777]

LGB_BASE = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'bagging_freq': 5, 'lambda_l1': 0.5, 'verbosity': -1, 'n_jobs': -1,
}
LGB_HP = {
    'num_leaves': 93, 'min_child_samples': 469, 'lambda_l2': 9.232, 'max_depth': 9,
    'learning_rate': 0.00698, 'feature_fraction': 0.490, 'bagging_fraction': 0.859,
    'early_stopping': 300,  # updated from 145 — was hitting 3000-tree cap
}

CB_PARAMS = {
    'iterations':            3000,
    'learning_rate':         0.03,
    'depth':                 7,
    'l2_leaf_reg':           2.0,
    'subsample':             0.8,
    'bootstrap_type':        'Bernoulli',
    'early_stopping_rounds': 300,
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
print(f"CatBoost Blend — Horizon {H}")
print(f"{'='*60}")

# ── Load data ──────────────────────────────────────────────
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
for col in feat_cols:
    if train[col].dtype == 'object':
        train[col] = train[col].astype('category')
        test[col]  = test[col].astype('category')

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

# ── LightGBM ───────────────────────────────────────────────
print(f"\nTraining LightGBM h={H} (3 seeds)...")
hp = {**LGB_BASE, **LGB_HP}
es = hp.pop('early_stopping')

val_preds_lgb  = np.zeros(len(X_val))
test_preds_lgb = np.zeros(len(X_test))
best_iters     = []

for seed in SEEDS:
    model = lgb.LGBMRegressor(**hp, random_state=seed)
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              eval_sample_weight=[w_train, w_val],
              eval_names=['train', 'val'],
              callbacks=[lgb.early_stopping(es, verbose=False),
                         lgb.log_evaluation(200)])
    best_iters.append(model.best_iteration_)
    val_preds_lgb  += model.predict(X_val)  / len(SEEDS)
    test_preds_lgb += model.predict(X_test) / len(SEEDS)
    print(f"  seed={seed}  best_iter={model.best_iteration_}")
    del model; gc.collect()

lgb_val_score = skill_score(y_val.values, val_preds_lgb, w_val)
avg_iter = int(np.mean(best_iters) * 1.1)
print(f"  Avg best_iter: {int(np.mean(best_iters))} → refit with {avg_iter}")
print(f"  LightGBM val score: {lgb_val_score:.4f}")

final_preds_lgb = np.zeros(len(X_test))
for seed in SEEDS:
    m = lgb.LGBMRegressor(**{**hp, 'n_estimators': avg_iter}, random_state=seed)
    m.fit(X_full, y_full, sample_weight=w_full)
    final_preds_lgb += m.predict(X_test) / len(SEEDS)
    del m; gc.collect()

# ── CatBoost ───────────────────────────────────────────────
print(f"\nTraining CatBoost h={H} (3 seeds)...")
print("  [verbose shows: Iter | learn RMSE | val RMSE | best | time]")

val_preds_cb   = np.zeros(len(X_val))
train_preds_cb = np.zeros(len(X_train))
test_preds_cb  = np.zeros(len(X_test))
cb_best_iters  = []
lc_train = lc_val = None

for seed in SEEDS:
    print(f"\n  --- seed={seed} ---")
    model_cb = CatBoostRegressor(**{**CB_PARAMS, 'random_seed': seed})
    model_cb.fit(
        Pool(X_train, y_train, weight=w_train),
        eval_set=Pool(X_val, y_val, weight=w_val),
        use_best_model=True,
    )
    best_iter = model_cb.best_iteration_
    cb_best_iters.append(best_iter)
    val_preds_cb   += model_cb.predict(X_val)   / len(SEEDS)
    train_preds_cb += model_cb.predict(X_train) / len(SEEDS)
    test_preds_cb  += model_cb.predict(X_test)  / len(SEEDS)
    print(f"  seed={seed}  best_iter={best_iter}")

    evals   = model_cb.get_evals_result()
    tr_rmse = np.array(evals['learn']['RMSE'])
    v_rmse  = np.array(evals['validation']['RMSE'])
    if lc_train is None:
        lc_train, lc_val = tr_rmse.copy(), v_rmse.copy()
    else:
        l = min(len(tr_rmse), len(lc_train))
        lc_train = (lc_train[:l] + tr_rmse[:l]) / 2
        lc_val   = (lc_val[:l]   + v_rmse[:l])  / 2
    del model_cb; gc.collect()

cb_val_score   = skill_score(y_val.values,   val_preds_cb,   w_val)
cb_train_score = skill_score(y_train.values, train_preds_cb, w_train)
gap = cb_train_score - cb_val_score
avg_cb_iter = int(np.mean(cb_best_iters) * 1.1)

print(f"\n  Avg best_iter: {int(np.mean(cb_best_iters))} → refit with {avg_cb_iter}")
print(f"  CatBoost train score : {cb_train_score:.4f}")
print(f"  CatBoost val score   : {cb_val_score:.4f}")
print(f"  Gap (train-val)      : {gap:.4f}  {'! possible overfit' if gap > 0.05 else 'OK'}")
print(f"  Val/Train RMSE ratio : {lc_val[-1]/lc_train[-1]:.3f}  (>1.3 = likely overfit)")

print(f"\n  Full-data refit ({len(tr):,} rows)...")
final_preds_cb = np.zeros(len(X_test))
for seed in SEEDS:
    params_refit = {**CB_PARAMS, 'random_seed': seed, 'iterations': avg_cb_iter, 'verbose': 0}
    params_refit.pop('early_stopping_rounds', None)
    m_cb = CatBoostRegressor(**params_refit)
    m_cb.fit(Pool(X_full, y_full, weight=w_full))
    final_preds_cb += m_cb.predict(X_test) / len(SEEDS)
    del m_cb; gc.collect()

# ── Blend search ───────────────────────────────────────────
print(f"\nBlend search (alpha = CatBoost weight):")
print(f"{'alpha':>8}  {'score':>8}")
print(f"{'------':>8}  {'--------':>8}")

best_alpha, best_score = 0.0, lgb_val_score
for alpha in np.arange(0.0, 1.05, 0.05):
    blend = (1.0 - alpha) * val_preds_lgb + alpha * val_preds_cb
    score = skill_score(y_val.values, blend, w_val)
    print(f"  {alpha:6.2f}    {score:.4f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"\nBest blend: alpha={best_alpha:.2f}  score={best_score:.4f}")
print(f"LightGBM alone: {lgb_val_score:.4f}  CatBoost alone: {cb_val_score:.4f}")
print(f"Delta vs LightGBM: {best_score - lgb_val_score:+.4f}")

# ── Save predictions ───────────────────────────────────────
val_blend = (1.0 - best_alpha) * val_preds_lgb + best_alpha * val_preds_cb
pd.DataFrame({
    'ts_index':     val_df['ts_index'].values,
    'y_true':       y_val.values,
    'y_pred_lgbm':  val_preds_lgb,
    'y_pred_cb':    val_preds_cb,
    'y_pred_blend': val_blend,
    'weight':       w_val,
}).to_csv(PROJECT_DIR / f"val_preds_h{H}_cbblend_cv{best_score:.4f}.csv", index=False)
print(f"\nVal preds saved: val_preds_h{H}_cbblend_cv{best_score:.4f}.csv")

test_blend = (1.0 - best_alpha) * final_preds_lgb + best_alpha * final_preds_cb
test_out = te[['id']].copy()
test_out['prediction'] = test_blend
test_out.to_csv(PROJECT_DIR / f"test_preds_h{H}_cbblend_a{best_alpha:.2f}_cv{best_score:.4f}.csv", index=False)
print(f"Test preds saved: test_preds_h{H}_cbblend_a{best_alpha:.2f}_cv{best_score:.4f}.csv")

# ── Learning curve + overfitting ratio plot ────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

iters = np.arange(1, len(lc_train) + 1)
axes[0].plot(iters, lc_train, color='#2a7fbf', linewidth=1.0, label='Train RMSE')
axes[0].plot(iters, lc_val,   color='#ff6b6b', linewidth=1.0, label='Val RMSE')
axes[0].axvline(len(iters), color='white', linewidth=0.8, linestyle=':', alpha=0.5)
axes[0].set_title(f'CatBoost h={H} — Learning Curve', fontweight='bold')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('RMSE')
axes[0].legend(); axes[0].grid(True, alpha=0.2)

ratio = lc_val / np.maximum(lc_train, 1e-12)
axes[1].plot(iters, ratio, color='#ffcc44', linewidth=1.0, label='Val/Train ratio')
axes[1].axhline(1.0, color='white',   linewidth=0.8, linestyle='--', alpha=0.5, label='ratio=1.0')
axes[1].axhline(1.3, color='#ff6b6b', linewidth=0.8, linestyle=':', alpha=0.7, label='ratio=1.3 (overfit zone)')
axes[1].set_title('Val/Train RMSE Ratio (overfit detector)', fontweight='bold')
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Ratio')
axes[1].legend(); axes[1].grid(True, alpha=0.2)

plt.suptitle(
    f'h={H} CatBoost | Train={cb_train_score:.4f} Val={cb_val_score:.4f} '
    f'Gap={gap:.4f} | Blend={best_score:.4f} (alpha={best_alpha:.2f})',
    fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(PROJECT_DIR / f"cb_lc_h{H}_cv{cb_val_score:.4f}.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"Plot saved: cb_lc_h{H}_cv{cb_val_score:.4f}.png")
