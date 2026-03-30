"""
Step 4: LightGBM Model
- Per-horizon models (1, 3, 10, 25)
- Sample weights from weight column (with optional log transform)
- 3-seed ensemble for stability
- Full-data refit for final predictions
- Validation split: ts_index <= 3500
- Overfitting controls: regularization, low-IC feature removal, log weights
- Diagnostics: train/val gap, learning curves, per-timestamp error plot
- Saves submission CSV to submissions/
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
SUB_DIR     = PROJECT_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

VAL_SPLIT = 3500
HORIZONS  = [1, 3, 10, 25]
SEEDS     = [42, 2024, 777]

# ── Overfitting controls ──────────────────────────────────────
# Set True to compress skewed weights — prevents model obsessing
# over a tiny number of extreme-weight rows
USE_LOG_WEIGHT = False

# Features with near-zero IC (Spearman ~0.0002) — pure noise
# Dropping them removes dimensions the model could overfit on
LOW_IC_FEATURES = [
    'feature_b', 'feature_c', 'feature_d', 'feature_e',
    'feature_f', 'feature_g', 'feature_h', 'feature_i',
]
# ─────────────────────────────────────────────────────────────

DROP_COLS = [
    'id', 'code', 'sub_code', 'sub_category', 'horizon',
    'ts_index', 'y_target', 'weight'
]

# Tighter regularization than baseline to reduce overfitting
# Short horizons are noisier → more regularization
# Long horizons have more structure → slightly looser
# Optuna-tuned params (30 trials per horizon, 2026-03-29)
HORIZON_PARAMS = {
    1:  {'num_leaves': 20,  'min_child_samples': 59,  'lambda_l2': 1.058,  'max_depth': 9,  'learning_rate': 0.04407, 'feature_fraction': 0.622, 'bagging_fraction': 0.309, 'early_stopping': 295},
    3:  {'num_leaves': 33,  'min_child_samples': 300, 'lambda_l2': 21.435, 'max_depth': 10, 'learning_rate': 0.03794, 'feature_fraction': 0.718, 'bagging_fraction': 0.745, 'early_stopping': 237},
    10: {'num_leaves': 93,  'min_child_samples': 469, 'lambda_l2': 9.232,  'max_depth': 9,  'learning_rate': 0.00698, 'feature_fraction': 0.490, 'bagging_fraction': 0.859, 'early_stopping': 145},
    25: {'num_leaves': 120, 'min_child_samples': 392, 'lambda_l2': 6.810,  'max_depth': 9,  'learning_rate': 0.00519, 'feature_fraction': 0.737, 'bagging_fraction': 0.599, 'early_stopping': 192},
}

BASE_PARAMS = {
    'objective':        'regression',
    'metric':           'rmse',
    'n_estimators':     3000,
    'bagging_freq':     5,
    'lambda_l1':        0.5,
    'verbosity':        -1,
    'n_jobs':           -1,
}

# ============================================================
# METRIC
# ============================================================
def skill_score(y_true, y_pred, w):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    w      = np.array(w)
    denom  = np.sum(w * y_true ** 2)
    if denom <= 0:
        return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))


# ============================================================
# LOAD DATA
# ============================================================
print("Loading feature data...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")
print(f"  Train: {train.shape} | Test: {test.shape}")

# Drop low-IC noise features (including all derived columns from them)
drop_feature_cols = []
for f in LOW_IC_FEATURES:
    drop_feature_cols += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feature_cols = [c for c in drop_feature_cols if c in train.columns]
train = train.drop(columns=drop_feature_cols)
test  = test.drop(columns=[c for c in drop_feature_cols if c in test.columns])
print(f"  Dropped {len(drop_feature_cols)} low-IC feature columns")

feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Feature columns: {len(feat_cols)}")

# Cast object columns to category for LightGBM
for col in feat_cols:
    if train[col].dtype == 'object':
        train[col] = train[col].astype('category')
        test[col]  = test[col].astype('category')

# ============================================================
# TRAINING LOOP
# ============================================================
test_preds     = []
cv_y, cv_p, cv_w = [], [], []
horizon_scores = {}
val_records    = []
learning_curves = {}   # {horizon: {'train': [...], 'val': [...]}}

total_start = time.time()

for horizon in HORIZONS:
    print(f"\n{'='*60}")
    print(f"HORIZON {horizon}")
    print(f"{'='*60}")
    h_start = time.time()

    tr = train[train.horizon == horizon]
    te = test[test.horizon == horizon]

    train_df = tr[tr.ts_index <= VAL_SPLIT]
    val_df   = tr[tr.ts_index  > VAL_SPLIT]

    X_train = train_df[feat_cols]
    y_train = train_df['y_target']
    w_train_raw = train_df['weight'].values

    X_val   = val_df[feat_cols]
    y_val   = val_df['y_target']
    w_val   = val_df['weight'].values

    X_test  = te[feat_cols]

    # Apply log weight transform if enabled
    if USE_LOG_WEIGHT:
        w_train = np.log1p(w_train_raw)
        w_val_es = np.log1p(w_val)   # log weights for early stopping (raw weights cause premature stop)
    else:
        w_train  = w_train_raw
        w_val_es = w_val

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    if USE_LOG_WEIGHT:
        print(f"  Weight range (log): {w_train.min():.2f} – {w_train.max():.2f}")

    hp = {**BASE_PARAMS, **HORIZON_PARAMS[horizon]}
    es = hp.pop('early_stopping')

    # ── Step 1: seed ensemble with early stopping ─────────────
    val_preds    = np.zeros(len(X_val))
    test_preds_h = np.zeros(len(X_test))
    train_preds  = np.zeros(len(X_train))
    best_iters   = []

    # Accumulate learning curves across seeds
    lc_train = None
    lc_val   = None

    for seed in SEEDS:
        evals_result = {}
        model = lgb.LGBMRegressor(**hp, random_state=seed)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_sample_weight=[w_train, w_val_es],
            eval_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(es, verbose=False),
                lgb.log_evaluation(200),
                lgb.record_evaluation(evals_result),
            ],
        )
        best_iters.append(model.best_iteration_)
        val_preds    += model.predict(X_val)   / len(SEEDS)
        test_preds_h += model.predict(X_test)  / len(SEEDS)
        train_preds  += model.predict(X_train) / len(SEEDS)

        # Accumulate learning curves (RMSE at each boosting round)
        n = len(evals_result['train']['rmse'])
        if lc_train is None:
            lc_train = np.array(evals_result['train']['rmse'])
            lc_val   = np.array(evals_result['val']['rmse'])
        else:
            # Pad shorter curve to match length
            l = min(n, len(lc_train))
            lc_train = (lc_train[:l] + np.array(evals_result['train']['rmse'])[:l]) / 2
            lc_val   = (lc_val[:l]   + np.array(evals_result['val']['rmse'])[:l])   / 2

    learning_curves[horizon] = {'train': lc_train, 'val': lc_val}

    # ── Overfitting diagnostic ─────────────────────────────────
    train_score = skill_score(y_train.values, train_preds, w_train_raw)
    val_score   = skill_score(y_val.values,   val_preds,   w_val)
    gap         = train_score - val_score

    horizon_scores[horizon] = val_score
    avg_iter = int(np.mean(best_iters) * 1.1)

    print(f"  Best iters     : {best_iters} → refit with {avg_iter}")
    print(f"  Train Score    : {train_score:.4f}")
    print(f"  Val Score      : {val_score:.4f}")
    print(f"  Gap (train-val): {gap:.4f}  {'! possible overfit' if gap > 0.05 else 'OK healthy'}")

    cv_y.extend(y_val.values)
    cv_p.extend(val_preds)
    cv_w.extend(w_val)

    # Store for per-timestamp plot
    val_rec = val_df[['ts_index']].copy()
    val_rec['y_true']  = y_val.values
    val_rec['y_pred']  = val_preds
    val_rec['weight']  = w_val
    val_rec['horizon'] = horizon
    val_records.append(val_rec)

    # ── Step 2: full-data refit ───────────────────────────────
    print(f"  Full-data refit ({len(tr)} rows)...")
    X_full = tr[feat_cols]
    y_full = tr['y_target']
    w_full_raw = tr['weight'].values
    w_full = np.log1p(w_full_raw) if USE_LOG_WEIGHT else w_full_raw
    final_preds = np.zeros(len(X_test))

    full_hp = {**hp, 'n_estimators': avg_iter}
    for seed in SEEDS:
        model_full = lgb.LGBMRegressor(**full_hp, random_state=seed)
        model_full.fit(X_full, y_full, sample_weight=w_full)
        final_preds += model_full.predict(X_test) / len(SEEDS)

    te = te.copy()
    te['prediction'] = final_preds
    test_preds.append(te[['id', 'prediction']])

    print(f"  Horizon {horizon} done in {time.time()-h_start:.1f}s")

    del tr, te, train_df, val_df, X_train, X_val, X_test, X_full
    gc.collect()

# ============================================================
# OVERALL SCORE
# ============================================================
overall = skill_score(np.array(cv_y), np.array(cv_p), np.array(cv_w))

print(f"\n{'='*60}")
print(f"OVERALL CV Skill Score : {overall:.4f}")
print(f"Per-horizon scores     : {horizon_scores}")
print(f"Log weight transform   : {USE_LOG_WEIGHT}")
print(f"Total time             : {(time.time()-total_start)/60:.1f} min")
print(f"{'='*60}")

# ============================================================
# PLOT 1 — LEARNING CURVES (train vs val RMSE per horizon)
# Shows if model overfits: val curve rises after early stopping point
# ============================================================
HCOLORS = {1: '#2a7fbf', 3: '#0e9e8e', 10: '#c46e2a', 25: '#8a3fba'}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, horizon in enumerate(HORIZONS):
    lc = learning_curves[horizon]
    iters = np.arange(1, len(lc['train']) + 1)
    axes[i].plot(iters, lc['train'], color=HCOLORS[horizon],
                 linewidth=1.0, alpha=0.9, label='Train RMSE')
    axes[i].plot(iters, lc['val'],   color='#ff6b6b',
                 linewidth=1.0, alpha=0.9, label='Val RMSE')
    axes[i].set_title(f'Horizon {horizon} — Learning Curve '
                      f'(CV={horizon_scores[horizon]:.4f})', fontweight='bold')
    axes[i].set_xlabel('Boosting Round')
    axes[i].set_ylabel('RMSE')
    axes[i].legend()
    axes[i].grid(True, alpha=0.2)

    # Mark where training stopped
    stop = len(iters)
    axes[i].axvline(stop, color='white', linewidth=0.8,
                    linestyle=':', alpha=0.5, label=f'Stop @ {stop}')

plt.suptitle(f'Learning Curves — CV={overall:.4f} | LogWeight={USE_LOG_WEIGHT}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
lc_path = PROJECT_DIR / f"learning_curves_cv{overall:.4f}.png"
plt.savefig(lc_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nLearning curves saved: {lc_path}")

# ============================================================
# PLOT 2 — VALIDATION ERROR OVER TIME
# Shows if model degrades toward end of validation window
# ============================================================
val_all = pd.concat(val_records, ignore_index=True)

def skill_per_ts(group):
    return skill_score(group['y_true'], group['y_pred'], group['weight'])

ts_scores = val_all.groupby('ts_index').apply(skill_per_ts).reset_index()
ts_scores.columns = ['ts_index', 'skill_score']

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(ts_scores['ts_index'], ts_scores['skill_score'],
             color='#2a7fbf', linewidth=1.0, alpha=0.8)
axes[0].axhline(overall, color='#ffcc44', linewidth=1.2,
                linestyle='--', label=f'Overall CV={overall:.4f}')
axes[0].axhline(0, color='white', linewidth=0.6, alpha=0.3)
axes[0].set_title('Validation Skill Score per Timestamp (all horizons)', fontweight='bold')
axes[0].set_ylabel('Skill Score')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

for h in HORIZONS:
    sub  = val_all[val_all.horizon == h]
    ts_h = sub.groupby('ts_index').apply(skill_per_ts).reset_index()
    ts_h.columns = ['ts_index', 'skill_score']
    ts_h['smoothed'] = ts_h['skill_score'].rolling(5, min_periods=1).mean()
    axes[1].plot(ts_h['ts_index'], ts_h['smoothed'],
                 color=HCOLORS[h], linewidth=1.2,
                 label=f'h={h} (CV={horizon_scores[h]:.4f})', alpha=0.85)

axes[1].axhline(0, color='white', linewidth=0.6, alpha=0.3)
axes[1].set_title('Validation Skill Score per Timestamp by Horizon (smoothed 5-step)', fontweight='bold')
axes[1].set_xlabel('ts_index (validation window: 3501–3601)')
axes[1].set_ylabel('Skill Score')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.suptitle(f'Validation Error Over Time — CV={overall:.4f}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
vt_path = PROJECT_DIR / f"val_error_over_time_cv{overall:.4f}.png"
plt.savefig(vt_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Validation error plot saved: {vt_path}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.concat(test_preds, ignore_index=True)
submission = submission.sort_values('id').reset_index(drop=True)

assert submission['prediction'].isnull().sum() == 0, "NaN found in predictions!"
assert len(submission) == len(test), f"Row count mismatch: {len(submission)} vs {len(test)}"

out_path = SUB_DIR / f"submission_cv{overall:.4f}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission saved: {out_path}")
print(f"Shape: {submission.shape}")
print(submission.head())
