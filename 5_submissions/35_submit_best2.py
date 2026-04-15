"""
35_submit_best2.py

Best per-horizon LGBM + skew/kurt submission.

  h=1  : LGBM + skew/kurt (v2)         val=0.0498  dart_results/test_preds_h1_skewkurt_cv0.0498.csv
  h=3  : LGBM + skew/kurt tuned (v2)   val=0.1211  dart_results/test_preds_h3_skewkurt_tuned_cv0.1211.csv
  h=10 : LGBM + skew/kurt 5k (v2)      val=0.2267  dart_results/test_preds_h10_skewkurt_5k_cv0.2267.csv
  h=25 : LGBM + skew/kurt tuned (v2)   val=0.2903  dart_results/test_preds_h25_skewkurt_tuned_cv0.2903.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
SUB_DIR     = PROJECT_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

FILES = {
    1:  PROJECT_DIR / "dart_results/test_preds_h1_skewkurt_cv0.0498.csv",
    3:  PROJECT_DIR / "dart_results/test_preds_h3_skewkurt_tuned_cv0.1211.csv",
    10: PROJECT_DIR / "dart_results/test_preds_h10_skewkurt_5k_cv0.2267.csv",
    25: PROJECT_DIR / "dart_results/test_preds_h25_skewkurt_tuned_cv0.2903.csv",
}

VAL_SCORES = {1: 0.0498, 3: 0.1211, 10: 0.2267, 25: 0.2903}

print("Loading per-horizon test preds...")
parts = []
for h, path in FILES.items():
    df = pd.read_csv(path)[['id', 'prediction']]
    print(f"  h={h:2d}: {len(df):,} rows | val={VAL_SCORES[h]:.4f} | {path.name}")
    parts.append(df)

sub = pd.concat(parts, ignore_index=True)
sub = sub.sort_values('id').reset_index(drop=True)

assert sub['prediction'].isnull().sum() == 0
print(f"\nTotal rows : {len(sub):,}")
print(f"Pred stats : mean={sub['prediction'].mean():.6f}  std={sub['prediction'].std():.6f}")

# True combined CV
val_files = {
    1:  PROJECT_DIR / "dart_results/val_preds_h1_skewkurt_cv0.0498.csv",
    3:  PROJECT_DIR / "dart_results/val_preds_h3_skewkurt_tuned_cv0.1211.csv",
    10: PROJECT_DIR / "dart_results/val_preds_h10_skewkurt_5k_cv0.2267.csv",
    25: PROJECT_DIR / "dart_results/val_preds_h25_skewkurt_tuned_cv0.2903.csv",
}

def skill_score(y_true, y_pred, w):
    y_true, y_pred, w = np.array(y_true), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

val_parts = []
for h, path in val_files.items():
    df = pd.read_csv(path)
    val_parts.append(df)

all_val = pd.concat(val_parts, ignore_index=True)
combined_cv = skill_score(all_val['y_target'], all_val['y_pred'], all_val['weight'])
print(f"\nTrue combined CV : {combined_cv:.4f}")
print(f"Previous best CV : 0.2497  ({combined_cv - 0.2497:+.4f})")
print(f"Optuna baseline  : 0.2474  ({combined_cv - 0.2474:+.4f})")

avg_val  = np.mean(list(VAL_SCORES.values()))
out_path = SUB_DIR / f"submission_skewkurt_v2_cv{combined_cv:.4f}.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path.name}")
