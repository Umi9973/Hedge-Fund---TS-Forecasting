"""
31_submit_best.py

Combines best test preds per horizon into a single submission.
All models are LGBM GBDT — no CatBoost.

  h=1  : LGBM + CS skew/kurt (v2)  val=0.0498  dart_results/test_preds_h1_skewkurt_cv0.0498.csv
  h=3  : LGBM + CS skew/kurt (v2)  val=0.1216  dart_results/test_preds_h3_skewkurt_cv0.1216.csv
  h=10 : LGBM + CS skew/kurt (v2)  val=0.2245  dart_results/test_preds_h10_skewkurt_cv0.2245.csv
  h=25 : LGBM + CS skew/kurt (v2)  val=0.2983  dart_results/test_preds_h25_skewkurt_cv0.2983.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
SUB_DIR     = PROJECT_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

FILES = {
    1:  PROJECT_DIR / "dart_results/test_preds_h1_skewkurt_cv0.0498.csv",
    3:  PROJECT_DIR / "dart_results/test_preds_h3_skewkurt_cv0.1216.csv",
    10: PROJECT_DIR / "dart_results/test_preds_h10_skewkurt_cv0.2245.csv",
    25: PROJECT_DIR / "dart_results/test_preds_h25_skewkurt_cv0.2983.csv",
}

VAL_SCORES = {1: 0.0498, 3: 0.1216, 10: 0.2245, 25: 0.2983}

print("Loading per-horizon test preds...")
parts = []
for h, path in FILES.items():
    df = pd.read_csv(path)[['id', 'prediction']]
    print(f"  h={h:2d}: {len(df):,} rows | val={VAL_SCORES[h]:.4f} | {path.name}")
    parts.append(df)

sub = pd.concat(parts, ignore_index=True)
sub = sub.sort_values('id').reset_index(drop=True)

assert sub['prediction'].isnull().sum() == 0, "NaN predictions found!"
print(f"\nTotal rows : {len(sub):,}")
print(f"Pred stats : mean={sub['prediction'].mean():.6f}  std={sub['prediction'].std():.6f}")

avg_val  = np.mean(list(VAL_SCORES.values()))
out_path = SUB_DIR / f"submission_skewkurt_avg{avg_val:.4f}.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path.name}")
print(f"Avg val score: {avg_val:.4f}")
