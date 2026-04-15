"""
26_add_skewkurt.py

Adds cross-sectional skewness + kurtosis features to existing parquets.
Saves as NEW files (does not overwrite originals):
  train_features_v2.parquet
  test_features_v2.parquet

New features: per (ts_index, horizon), for top 20 IC features:
  {feat}_cs_skew  — skewness of feature across all instruments at that time step
  {feat}_cs_kurt  — kurtosis of feature across all instruments at that time step

These capture distribution SHAPE (fat-tail days, skewed days) — info not
in our existing cs_z / cs_rank features which only use mean and std.
"""

import pandas as pd
import numpy as np
import gc
import time
from pathlib import Path

PROJECT_DIR = Path("G:/Umi/Python Projects/TS Forecast")

TOP_FEATURES = [
    'feature_bz', 'feature_ca', 'feature_by', 'feature_am', 'feature_u',
    'feature_ao', 'feature_cd', 'feature_cc', 'feature_cb', 'feature_az',
    'feature_bq', 'feature_ag', 'feature_ap', 'feature_br', 'feature_bp',
    'feature_bs', 'feature_bn', 'feature_bo', 'feature_al', 'feature_an',
]

# ── Load ─────────────────────────────────────────────────────
print("Loading train + test features...")
train = pd.read_parquet(PROJECT_DIR / "train_features.parquet")
test  = pd.read_parquet(PROJECT_DIR / "test_features.parquet")
print(f"  Train: {train.shape} | Test: {test.shape}")

# ── Combine for cross-sectional groupby ──────────────────────
# Need both splits so groupby covers all ts_index batches in test too
train['_split'] = 0
test['_split']  = 1
test['y_target'] = np.nan
test['weight']   = np.nan

df = pd.concat([train, test], ignore_index=True)
del train, test
gc.collect()
print(f"  Combined: {df.shape}")

# ── Compute skew + kurt per (ts_index, horizon) ───────────────
print("\nComputing CS skewness + kurtosis...")
print("  (per ts_index × horizon, across all instruments)")

# Pre-allocate all new columns at once to avoid DataFrame fragmentation
new_cols = {}
t0 = time.time()

for i, feat in enumerate(TOP_FEATURES):
    if feat not in df.columns:
        continue
    grp = df.groupby(['ts_index', 'horizon'])[feat]
    new_cols[f'{feat}_cs_skew'] = grp.transform('skew').fillna(0).values
    new_cols[f'{feat}_cs_kurt'] = grp.transform(lambda x: x.kurt()).fillna(0).values
    print(f"  [{i+1}/{len(TOP_FEATURES)}] {feat}  ({time.time()-t0:.0f}s elapsed)")

# Add all new columns at once (avoids fragmentation warning)
new_df = pd.DataFrame(new_cols, index=df.index)
df = pd.concat([df, new_df], axis=1)
del new_cols, new_df
gc.collect()

print(f"\n  Added {len(TOP_FEATURES) * 2} new features in {time.time()-t0:.1f}s")
print(f"  Combined shape now: {df.shape}")

# ── Split back ────────────────────────────────────────────────
print("\nSplitting back into train / test...")
train_v2 = df[df._split == 0].drop(columns=['_split']).reset_index(drop=True)
test_v2  = df[df._split == 1].drop(columns=['_split', 'y_target', 'weight']).reset_index(drop=True)
del df
gc.collect()

print(f"  train_v2: {train_v2.shape}")
print(f"  test_v2:  {test_v2.shape}")

# ── Save as new files ─────────────────────────────────────────
print("\nSaving...")
t1 = time.time()
train_v2.to_parquet(PROJECT_DIR / "train_features_v2.parquet", index=False)
print(f"  train_features_v2.parquet saved ({time.time()-t1:.1f}s)")
t1 = time.time()
test_v2.to_parquet(PROJECT_DIR / "test_features_v2.parquet", index=False)
print(f"  test_features_v2.parquet  saved ({time.time()-t1:.1f}s)")

print(f"\nDone. New feature count: {train_v2.shape[1]} (was {train_v2.shape[1] - len(TOP_FEATURES)*2})")
print("Use train_features_v2.parquet / test_features_v2.parquet in future scripts.")
