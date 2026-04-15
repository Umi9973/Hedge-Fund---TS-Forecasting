"""
Walk-Forward CV — LightGBM Horizon 3
4 folds, each trains on all data up to cutpoint, validates on next 100 timestamps:
  Fold 1: Train [min–3200] → Val [3201–3300]
  Fold 2: Train [min–3300] → Val [3301–3400]
  Fold 3: Train [min–3400] → Val [3401–3500]
  Fold 4: Train [min–3500] → Val [3501–3601]  ← current single split

Reports per-fold score + average. Full-data refit for test predictions.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import time
from pathlib import Path

plt.style.use('dark_background')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")

H     = 25
SEEDS = [42, 2024, 777]

FOLDS = [
    {'train_end': 3200, 'val_start': 3201, 'val_end': 3300},
    {'train_end': 3300, 'val_start': 3301, 'val_end': 3400},
    {'train_end': 3400, 'val_start': 3401, 'val_end': 3500},
    {'train_end': 3500, 'val_start': 3501, 'val_end': 3601},
]

BASE_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'bagging_freq': 5, 'lambda_l1': 0.5, 'verbosity': -1, 'n_jobs': -1,
}
HORIZON_PARAMS = {
    'num_leaves': 120, 'min_child_samples': 392, 'lambda_l2': 6.81, 'max_depth': 9,
    'learning_rate': 0.00519, 'feature_fraction': 0.737, 'bagging_fraction': 0.599,
    'early_stopping': 192,
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
print(f"Walk-Forward CV — LightGBM Horizon {H}")
print(f"{'='*60}")

# ── Load data ──────────────────────────────────────────────
print(f"\nLoading data...")
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
X_test = te[feat_cols]

print(f"  Full train rows (h={H}): {len(tr):,} | Test: {len(X_test):,}")
print(f"  Features: {len(feat_cols)}")

hp = {**BASE_PARAMS, **HORIZON_PARAMS}
es = hp.pop('early_stopping')

# ── Walk-Forward CV ────────────────────────────────────────
fold_scores  = []
fold_iters   = []
all_val_preds = []
total_start  = time.time()

for fold_idx, fold in enumerate(FOLDS):
    fold_num = fold_idx + 1
    print(f"\n{'─'*60}")
    print(f"Fold {fold_num}/4 | Train [–{fold['train_end']}] → Val [{fold['val_start']}–{fold['val_end']}]")
    print(f"{'─'*60}")

    train_df = tr[tr.ts_index <= fold['train_end']]
    val_df   = tr[(tr.ts_index >= fold['val_start']) & (tr.ts_index <= fold['val_end'])]

    X_train, y_train = train_df[feat_cols], train_df['y_target']
    w_train = train_df['weight'].values
    X_val,   y_val   = val_df[feat_cols],   val_df['y_target']
    w_val   = val_df['weight'].values

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    val_preds   = np.zeros(len(X_val))
    train_preds = np.zeros(len(X_train))
    best_iters  = []

    for seed in SEEDS:
        model = lgb.LGBMRegressor(**hp, random_state=seed)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  eval_sample_weight=[w_train, w_val],
                  eval_names=['train', 'val'],
                  callbacks=[lgb.early_stopping(es, verbose=False),
                             lgb.log_evaluation(200)])
        best_iters.append(model.best_iteration_)
        val_preds   += model.predict(X_val)   / len(SEEDS)
        train_preds += model.predict(X_train) / len(SEEDS)
        print(f"  seed={seed}  best_iter={model.best_iteration_}")
        del model; gc.collect()

    val_score   = skill_score(y_val.values,   val_preds,   w_val)
    train_score = skill_score(y_train.values, train_preds, w_train)
    gap = train_score - val_score
    avg_iter = int(np.mean(best_iters))
    fold_iters.append(avg_iter)
    fold_scores.append(val_score)

    print(f"\n  Train score : {train_score:.4f}")
    print(f"  Val score   : {val_score:.4f}")
    print(f"  Gap         : {gap:.4f}  {'! overfit' if gap > 0.05 else 'OK'}")
    print(f"  Avg iter    : {avg_iter}")

    # Store val records for analysis
    rec = val_df[['ts_index']].copy()
    rec['y_true']  = y_val.values
    rec['y_pred']  = val_preds
    rec['weight']  = w_val
    rec['fold']    = fold_num
    all_val_preds.append(rec)

# ── Summary ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Walk-Forward CV Summary — h={H}")
print(f"{'='*60}")
for i, (score, iters) in enumerate(zip(fold_scores, fold_iters)):
    fold = FOLDS[i]
    print(f"  Fold {i+1} [{fold['val_start']}–{fold['val_end']}]: {score:.4f}  (avg_iter={iters})")

wf_score = np.mean(fold_scores)
wf_std   = np.std(fold_scores)
print(f"\n  WF-CV Mean  : {wf_score:.4f}")
print(f"  WF-CV Std   : {wf_std:.4f}")
print(f"  Fold 4 only : {fold_scores[3]:.4f}  (our current single-split baseline)")
print(f"  Difference  : {fold_scores[3] - wf_score:+.4f}  (+ = fold4 optimistic)")
print(f"\n  Total time  : {(time.time()-total_start)/60:.1f} min")

# ── Full-data refit for test preds ─────────────────────────
X_full = tr[feat_cols]
y_full = tr['y_target']
w_full = tr['weight'].values
avg_final_iter = int(np.mean(fold_iters) * 1.1)

print(f"\n  Full-data refit ({len(tr):,} rows, {avg_final_iter} iters)...")
final_preds = np.zeros(len(X_test))
for seed in SEEDS:
    m = lgb.LGBMRegressor(**{**hp, 'n_estimators': avg_final_iter}, random_state=seed)
    m.fit(X_full, y_full, sample_weight=w_full)
    final_preds += m.predict(X_test) / len(SEEDS)
    del m; gc.collect()

test_out = te[['id']].copy()
test_out['prediction'] = final_preds
test_out.to_csv(PROJECT_DIR / f"test_preds_h{H}_wfcv{wf_score:.4f}.csv", index=False)
print(f"  Test preds saved: test_preds_h{H}_wfcv{wf_score:.4f}.csv")

# ── Plot: per-fold val score ───────────────────────────────
val_all = pd.concat(all_val_preds, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

fold_labels = [f"Fold {i+1}\n[{f['val_start']}–{f['val_end']}]" for i, f in enumerate(FOLDS)]
colors = ['#2a7fbf', '#0e9e8e', '#c46e2a', '#8a3fba']
bars = axes[0].bar(fold_labels, fold_scores, color=colors, alpha=0.85)
axes[0].axhline(wf_score, color='#ffcc44', linewidth=1.5, linestyle='--', label=f'WF Mean={wf_score:.4f}')
axes[0].axhline(fold_scores[3], color='white', linewidth=1.0, linestyle=':', alpha=0.6, label=f'Fold4={fold_scores[3]:.4f}')
for bar, score in zip(bars, fold_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=10)
axes[0].set_title(f'h={H} Walk-Forward CV — Per-Fold Skill Score', fontweight='bold')
axes[0].set_ylabel('Skill Score')
axes[0].legend()
axes[0].grid(True, alpha=0.2, axis='y')

# Per-timestamp skill score across all folds
def skill_per_ts(group):
    return skill_score(group['y_true'], group['y_pred'], group['weight'])

ts_scores = val_all.groupby('ts_index').apply(skill_per_ts).reset_index()
ts_scores.columns = ['ts_index', 'skill_score']
axes[1].plot(ts_scores['ts_index'], ts_scores['skill_score'],
             color='#2a7fbf', linewidth=0.8, alpha=0.8)
axes[1].axhline(wf_score, color='#ffcc44', linewidth=1.2, linestyle='--',
                label=f'WF Mean={wf_score:.4f}')
for fold in FOLDS:
    axes[1].axvline(fold['val_start'], color='white', linewidth=0.6, alpha=0.3, linestyle=':')
axes[1].set_title('Skill Score per Timestamp (all folds)', fontweight='bold')
axes[1].set_xlabel('ts_index')
axes[1].set_ylabel('Skill Score')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.suptitle(f'h={H} LightGBM Walk-Forward CV | Mean={wf_score:.4f} ± {wf_std:.4f}',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_DIR / f"wfcv_h{H}_{wf_score:.4f}.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Plot saved: wfcv_h{H}_{wf_score:.4f}.png")
