"""
Step 6: Per-(ts_index, horizon) rank normalization of predictions.

Core idea: rank-normalize predictions within each date×horizon group so
the model outputs a cross-sectional signal rather than absolute values.
Removes date-level systematic bias; aligns with the skill score metric.

Two variants tested:
  - rank_uniform : ranks mapped to [0, 1] uniform
  - rank_gaussian: ranks mapped to standard normal (Gaussianized)

Validation: test on h=3 val predictions first, then apply to full submission.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
SUB_DIR     = PROJECT_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

VAL_SPLIT = 3500

# ============================================================
# HELPERS
# ============================================================
def rank_norm_uniform(pred):
    """Rank within group, map to (0, 1) exclusive."""
    r = rankdata(pred, method='average')
    return r / (len(r) + 1)          # +1 keeps values strictly inside (0,1)

def rank_norm_gaussian(pred):
    """Rank within group, Gaussianize via normal PPF."""
    u = rank_norm_uniform(pred)
    return norm.ppf(u)

def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true**2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred)**2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

def apply_rank_norm(df, pred_col, group_cols, method='gaussian'):
    """Apply rank normalization within each group, return new series."""
    fn = rank_norm_gaussian if method == 'gaussian' else rank_norm_uniform
    result = df.groupby(group_cols)[pred_col].transform(fn)
    return result

# ============================================================
# STEP 1: VALIDATE ON H=3 VAL PREDICTIONS
# ============================================================
print("=" * 60)
print("STEP 1: Validate on h=3 val predictions")
print("=" * 60)

# Find the most recent h=3 val preds file
h3_dir = PROJECT_DIR / "h3_experiment_submission"
val_files = sorted(h3_dir.glob("val_preds_h3_cv*.csv"))

if not val_files:
    print("  No h=3 val predictions found — skipping validation step")
else:
    # Use baseline (no-clipping) val predictions — the last experiment file
    val_file = val_files[-1]
    print(f"  Using: {val_file.name}")

    val = pd.read_csv(val_file)
    print(f"  Rows: {len(val)} | Columns: {list(val.columns)}")

    # Original score
    score_raw = skill_score(val['y_true'], val['y_pred'], val['weight'])
    print(f"\n  Raw h=3 val score        : {score_raw:.4f}")

    # Add a dummy horizon col since this file is h=3 only
    val['horizon'] = 3

    # Rank norm variants
    val['pred_uniform']  = apply_rank_norm(val, 'y_pred', ['ts_index'], method='uniform')
    val['pred_gaussian'] = apply_rank_norm(val, 'y_pred', ['ts_index'], method='gaussian')

    score_uniform  = skill_score(val['y_true'], val['pred_uniform'],  val['weight'])
    score_gaussian = skill_score(val['y_true'], val['pred_gaussian'], val['weight'])

    print(f"  Rank-uniform  val score  : {score_uniform:.4f}  ({score_uniform - score_raw:+.4f})")
    print(f"  Rank-gaussian val score  : {score_gaussian:.4f}  ({score_gaussian - score_raw:+.4f})")

    best_method = 'gaussian' if score_gaussian >= score_uniform else 'uniform'
    best_score  = max(score_gaussian, score_uniform)
    print(f"\n  Best method: {best_method}  ({best_score:.4f})")

# ============================================================
# STEP 2: APPLY TO FULL TEST SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Apply to full test submission")
print("=" * 60)

# Load submission
sub_path = SUB_DIR / "submission_cv0.2403.csv"
if not sub_path.exists():
    # Try project root fallback
    sub_path = PROJECT_DIR / "submission_cv0.2403.csv"
print(f"  Loading submission: {sub_path}")
sub = pd.read_csv(sub_path)
print(f"  Rows: {len(sub)}")

# Load test features to get ts_index + horizon per id
print("  Loading test features for ts_index/horizon mapping...")
test_meta = pd.read_parquet(
    PROJECT_DIR / "test_features.parquet",
    columns=['id', 'ts_index', 'horizon']
)

# Merge
sub = sub.merge(test_meta, on='id', how='left')
assert sub['ts_index'].isnull().sum() == 0, "Missing ts_index after merge!"
print(f"  ts_index range: {sub['ts_index'].min()} – {sub['ts_index'].max()}")
print(f"  Horizons: {sorted(sub['horizon'].unique())}")

# Apply rank norm (gaussian — typically best in financial competitions)
print("\n  Applying rank-gaussian normalization per (ts_index, horizon)...")
sub['prediction_ranknorm'] = apply_rank_norm(
    sub, 'prediction', ['ts_index', 'horizon'], method='gaussian'
)

# Sanity checks
print(f"\n  Raw pred stats      : mean={sub['prediction'].mean():.6f}  std={sub['prediction'].std():.6f}")
print(f"  Rank-norm pred stats: mean={sub['prediction_ranknorm'].mean():.6f}  std={sub['prediction_ranknorm'].std():.6f}")

# Save both variants
out_raw  = sub[['id', 'prediction']].sort_values('id').reset_index(drop=True)
out_norm = sub[['id', 'prediction_ranknorm']].rename(
    columns={'prediction_ranknorm': 'prediction'}
).sort_values('id').reset_index(drop=True)

# Also save uniform variant
sub['prediction_uniform'] = apply_rank_norm(
    sub, 'prediction', ['ts_index', 'horizon'], method='uniform'
)
out_uniform = sub[['id', 'prediction_uniform']].rename(
    columns={'prediction_uniform': 'prediction'}
).sort_values('id').reset_index(drop=True)

path_gaussian = SUB_DIR / "submission_cv0.2403_ranknorm_gaussian.csv"
path_uniform  = SUB_DIR / "submission_cv0.2403_ranknorm_uniform.csv"

out_norm.to_csv(path_gaussian, index=False)
out_uniform.to_csv(path_uniform, index=False)

assert out_norm['prediction'].isnull().sum() == 0
assert out_uniform['prediction'].isnull().sum() == 0

print(f"\n  Saved: {path_gaussian.name}")
print(f"  Saved: {path_uniform.name}")
print(f"\nDone. Submit gaussian variant first (recommended by competition research).")
