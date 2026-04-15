"""
Compute true combined skill score across all 4 horizons for the skewkurt submission.
Skill score = sqrt(1 - clip(sum(w*(y-pred)^2) / sum(w*y^2), 0, 1))
computed over ALL rows together, not averaged per horizon.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")

VAL_FILES = {
    1:  PROJECT_DIR / "dart_results/val_preds_h1_skewkurt_cv0.0498.csv",
    3:  PROJECT_DIR / "dart_results/val_preds_h3_skewkurt_cv0.1216.csv",
    10: PROJECT_DIR / "dart_results/val_preds_h10_skewkurt_cv0.2245.csv",
    25: PROJECT_DIR / "dart_results/val_preds_h25_skewkurt_cv0.2983.csv",
}

def skill_score(y_target, y_pred, w):
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * y_target ** 2)
    if denom <= 0: return 0.0
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

parts = []
for h, path in VAL_FILES.items():
    df = pd.read_csv(path)
    df['horizon'] = h
    h_score = skill_score(df['y_target'], df['y_pred'], df['weight'])
    print(f"  h={h:2d}: {len(df):,} rows | per-horizon score={h_score:.4f}")
    parts.append(df)

all_val = pd.concat(parts, ignore_index=True)

combined = skill_score(all_val['y_target'], all_val['y_pred'], all_val['weight'])
naive_avg = np.mean([skill_score(
    g['y_target'], g['y_pred'], g['weight']
) for _, g in all_val.groupby('horizon')])

print(f"\n  Naive avg of per-horizon scores : {naive_avg:.4f}")
print(f"  True combined skill score       : {combined:.4f}  <-- this is CV")
print(f"\n  Previous best (WF submission)   : 0.2438 (LB)")
print(f"  Previous best (optuna)          : 0.2474 (CV)")
