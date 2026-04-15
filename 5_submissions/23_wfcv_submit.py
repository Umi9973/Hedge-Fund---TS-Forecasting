"""
Build submission from walk-forward CV test predictions (all 4 horizons).
Loads test_preds_h{H}_wfcv*.csv for each horizon, concatenates, sorts, saves.
"""

import pandas as pd
import glob
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")
SUB_DIR     = PROJECT_DIR / "submissions"
SUB_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 3, 10, 25]
all_preds = []
scores = {}

for H in HORIZONS:
    files = sorted(glob.glob(str(PROJECT_DIR / f"test_preds_h{H}_wfcv*.csv")))
    if not files:
        raise FileNotFoundError(f"No wfcv test preds found for h={H}")
    path = files[-1]  # most recent
    score = float(path.split('wfcv')[1].replace('.csv', ''))
    scores[H] = score
    df = pd.read_csv(path)
    all_preds.append(df[['id', 'prediction']])
    print(f"  h={H}: loaded {path.split('/')[-1]} ({len(df):,} rows, score={score:.4f})")

submission = pd.concat(all_preds, ignore_index=True)
submission = submission.sort_values('id').reset_index(drop=True)

assert submission['prediction'].isnull().sum() == 0, "NaN in predictions!"
print(f"\n  Total rows: {len(submission):,}")

avg_score = sum(scores.values()) / len(scores)
out_path = SUB_DIR / f"submission_wfcv_avg{avg_score:.4f}.csv"
submission.to_csv(out_path, index=False)
print(f"  Submission saved: {out_path}")
print(f"  Per-horizon WF scores: {scores}")
