"""
Step 8: Optuna hyperparameter tuning — all 4 horizons
- 30 trials per horizon (120 total) — designed to run overnight
- 1 seed per trial for speed; final validation uses 3-seed ensemble
- Saves best params per horizon to optuna_results/best_params.json
- Saves full study per horizon to optuna_results/study_h{n}.pkl
- Uses baseline feature set (328 features)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import json
import pickle
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_DIR  = Path("G:/Umi/Python Projects/TS Forecast")
OUT_DIR      = PROJECT_DIR / "optuna_results"
OUT_DIR.mkdir(exist_ok=True)

VAL_SPLIT = 3500
HORIZONS  = [1, 3, 10, 25]
N_TRIALS  = 30      # per horizon — ~6-8 hours total overnight
SEED      = 42      # single seed per trial for speed

DROP_COLS = [
    'id', 'code', 'sub_code', 'sub_category', 'horizon',
    'ts_index', 'y_target', 'weight'
]
LOW_IC_FEATURES = [
    'feature_b', 'feature_c', 'feature_d', 'feature_e',
    'feature_f', 'feature_g', 'feature_h', 'feature_i',
]

# Baseline params (fixed — not tuned)
BASE_PARAMS = {
    'objective':     'regression',
    'metric':        'rmse',
    'bagging_freq':  5,
    'lambda_l1':     0.5,
    'verbosity':     -1,
    'n_jobs':        -1,
}

# Baseline per-horizon params (reference point)
BASELINE_PARAMS = {
    1:  {'num_leaves': 50,  'min_child_samples': 400, 'lambda_l2': 20.0, 'max_depth': 7,  'learning_rate': 0.02, 'feature_fraction': 0.5, 'bagging_fraction': 0.6, 'n_estimators': 3000, 'early_stopping': 150},
    3:  {'num_leaves': 55,  'min_child_samples': 350, 'lambda_l2': 18.0, 'max_depth': 7,  'learning_rate': 0.02, 'feature_fraction': 0.5, 'bagging_fraction': 0.6, 'n_estimators': 3000, 'early_stopping': 140},
    10: {'num_leaves': 63,  'min_child_samples': 280, 'lambda_l2': 14.0, 'max_depth': 8,  'learning_rate': 0.02, 'feature_fraction': 0.5, 'bagging_fraction': 0.6, 'n_estimators': 3000, 'early_stopping': 130},
    25: {'num_leaves': 70,  'min_child_samples': 220, 'lambda_l2': 11.0, 'max_depth': 8,  'learning_rate': 0.02, 'feature_fraction': 0.5, 'bagging_fraction': 0.6, 'n_estimators': 3000, 'early_stopping': 120},
}

def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

# ============================================================
# LOAD DATA ONCE
# ============================================================
print("Loading feature data...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
# test not needed for tuning — skip load to save RAM

drop_feature_cols = []
for f in LOW_IC_FEATURES:
    drop_feature_cols += [c for c in train.columns if c == f or c.startswith(f + '_')]
drop_feature_cols = [c for c in drop_feature_cols if c in train.columns]
train = train.drop(columns=drop_feature_cols)
feat_cols = [c for c in train.columns if c not in DROP_COLS]
print(f"  Feature columns: {len(feat_cols)}")

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def make_objective(X_train, y_train, w_train, X_val, y_val, w_val):

    def objective(trial):
        params = {
            **BASE_PARAMS,
            'n_estimators':      3000,
            'learning_rate':     trial.suggest_float('learning_rate',     0.005, 0.05,  log=True),
            'num_leaves':        trial.suggest_int(  'num_leaves',        20,    120),
            'max_depth':         trial.suggest_int(  'max_depth',         5,     10),
            'min_child_samples': trial.suggest_int(  'min_child_samples', 50,    600),
            'lambda_l2':         trial.suggest_float('lambda_l2',         1.0,   40.0, log=True),
            'feature_fraction':  trial.suggest_float('feature_fraction',  0.3,   0.9),
            'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.3,   0.9),
        }
        es = trial.suggest_int('early_stopping', 80, 300)

        model = lgb.LGBMRegressor(**params, random_state=SEED)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            callbacks=[
                lgb.early_stopping(es, verbose=False),
            ],
        )
        preds = model.predict(X_val)
        return skill_score(y_val, preds, w_val)

    return objective

# ============================================================
# TUNE EACH HORIZON
# ============================================================
all_best_params = {}
total_start = time.time()

for horizon in HORIZONS:
    print(f"\n{'='*60}")
    print(f"HORIZON {horizon}  ({N_TRIALS} trials)")
    print(f"{'='*60}")
    h_start = time.time()

    tr       = train[train.horizon == horizon]
    train_df = tr[tr.ts_index <= VAL_SPLIT]
    val_df   = tr[tr.ts_index  > VAL_SPLIT]

    X_train = train_df[feat_cols]
    y_train = train_df['y_target'].values
    w_train = train_df['weight'].values
    X_val   = val_df[feat_cols]
    y_val   = val_df['y_target'].values
    w_val   = val_df['weight'].values

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # Seed with baseline params as first trial so we always have a reference
    baseline = BASELINE_PARAMS[horizon]
    study.enqueue_trial({
        'learning_rate':     baseline['learning_rate'],
        'num_leaves':        baseline['num_leaves'],
        'max_depth':         baseline['max_depth'],
        'min_child_samples': baseline['min_child_samples'],
        'lambda_l2':         baseline['lambda_l2'],
        'feature_fraction':  baseline['feature_fraction'],
        'bagging_fraction':  baseline['bagging_fraction'],
        'early_stopping':    baseline['early_stopping'],
    })

    objective = make_objective(X_train, y_train, w_train, X_val, y_val, w_val)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    print(f"\n  Best trial #{best.number}: score={best.value:.4f}")
    print(f"  Best params: {best.params}")

    # Save study
    with open(OUT_DIR / f"study_h{horizon}.pkl", 'wb') as f:
        pickle.dump(study, f)

    # Store best params
    all_best_params[horizon] = {
        'val_score':  best.value,
        'params':     best.params,
        'n_trials':   len(study.trials),
        'runtime_min': (time.time() - h_start) / 60,
    }

    # Print all trials sorted by score
    trials_df = study.trials_dataframe().sort_values('value', ascending=False)
    print(f"\n  Top 5 trials:")
    print(trials_df[['number', 'value']].head(5).to_string(index=False))

    print(f"  Horizon {horizon} done in {(time.time()-h_start)/60:.1f} min")

# ============================================================
# SAVE RESULTS
# ============================================================
with open(OUT_DIR / "best_params.json", 'w') as f:
    json.dump(all_best_params, f, indent=2)

print(f"\n{'='*60}")
print(f"OPTUNA TUNING COMPLETE")
print(f"{'='*60}")
for h in HORIZONS:
    r = all_best_params[h]
    print(f"  h={h:2d}: best_score={r['val_score']:.4f}  ({r['n_trials']} trials, {r['runtime_min']:.1f} min)")
print(f"\nTotal runtime: {(time.time()-total_start)/60:.1f} min")
print(f"Results saved to: {OUT_DIR}/best_params.json")
