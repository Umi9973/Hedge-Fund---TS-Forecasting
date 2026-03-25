"""
Step 1: Data Preprocessing
- Compute NaN fill values from training split (ts_index <= 3500) only
- Fill NaN grouped by sub_category, fallback to global median
- Downcast dtypes to save memory
- Save cleaned files as parquet
"""

import duckdb
import pandas as pd
import numpy as np

PROJECT_DIR = "G:/Umi/Python Projects/TS Forecast"
TRAIN_PATH  = f"{PROJECT_DIR}/train.parquet"
TEST_PATH   = f"{PROJECT_DIR}/test.parquet"
VAL_SPLIT   = 3500

# ============================================================
# LOAD
# ============================================================
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  Train: {train.shape} | Test: {test.shape}")

feature_cols = [c for c in train.columns if c.startswith("feature_")]
print(f"  Feature columns: {len(feature_cols)}")

# ============================================================
# NULL ANALYSIS
# ============================================================
print("\n--- Null Analysis ---")
null_counts = train[feature_cols].isnull().sum()
null_features = null_counts[null_counts > 0].sort_values(ascending=False)
print(f"  Features with nulls: {len(null_features)} / {len(feature_cols)}")
print(null_features.to_string())

# Check if nulls cluster in specific sub_categories
print("\n  Null rate by sub_category (top features):")
for feat in null_features.index[:5]:
    rates = train.groupby("sub_category")[feat].apply(lambda x: x.isnull().mean())
    print(f"    {feat}: {rates.to_dict()}")

# ============================================================
# COMPUTE FILL VALUES (training split only — no leakage)
# ============================================================
print("\n--- Computing fill values from ts_index <= 3500 only ---")

train_only = train[train.ts_index <= VAL_SPLIT]

# Group median by sub_category
group_medians = (
    train_only[["sub_category"] + list(null_features.index)]
    .groupby("sub_category")
    .median()
)
print(f"  Group medians computed for {len(null_features)} features × {len(group_medians)} sub_categories")

# Global median fallback
global_medians = train_only[list(null_features.index)].median()
print(f"  Global medians computed as fallback")

# ============================================================
# FILL NaN
# ============================================================
def fill_nulls(df, group_medians, global_medians, null_features):
    df = df.copy()
    for feat in null_features.index:
        if feat not in df.columns:
            continue
        for subcat, grp_idx in df.groupby("sub_category").groups.items():
            mask = df.index[grp_idx][df.loc[grp_idx, feat].isnull()]
            if len(mask) == 0:
                continue
            # Use group median if available, else global
            if subcat in group_medians.index and not pd.isna(group_medians.loc[subcat, feat]):
                fill_val = group_medians.loc[subcat, feat]
            else:
                fill_val = global_medians[feat]
            df.loc[mask, feat] = fill_val
    return df

print("\n--- Filling NaN ---")
print("  Filling train...")
train = fill_nulls(train, group_medians, global_medians, null_features)
print("  Filling test...")
test  = fill_nulls(test,  group_medians, global_medians, null_features)

# Final safety net — fill anything still null with global median from training split
# (catches features that had nulls in test but not in train)
extra_nulls = test[feature_cols].isnull().sum()
extra_null_feats = extra_nulls[extra_nulls > 0].index.tolist()
if extra_null_feats:
    fallback = train_only[extra_null_feats].median()
    test[extra_null_feats] = test[extra_null_feats].fillna(fallback)
    print(f"  Safety net filled {len(extra_null_feats)} features in test using global median")

# Verify
remaining_train = train[feature_cols].isnull().sum().sum()
remaining_test  = test[feature_cols].isnull().sum().sum()
print(f"  Remaining nulls — Train: {remaining_train} | Test: {remaining_test}")

# ============================================================
# MEMORY OPTIMIZATION — downcast dtypes
# ============================================================
print("\n--- Memory Optimization ---")

def downcast(df):
    before = df.memory_usage(deep=True).sum() / 1e6
    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="int64").columns:
        if col not in ["ts_index", "horizon"]:  # keep these readable
            df[col] = pd.to_numeric(df[col], downcast="integer")
    after = df.memory_usage(deep=True).sum() / 1e6
    print(f"  Memory: {before:.1f} MB → {after:.1f} MB ({(1-after/before)*100:.1f}% reduction)")
    return df

train = downcast(train)
test  = downcast(test)

# ============================================================
# SAVE
# ============================================================
print("\n--- Saving cleaned files ---")
train.to_parquet(f"{PROJECT_DIR}/train_clean.parquet", index=False)
test.to_parquet(f"{PROJECT_DIR}/test_clean.parquet",   index=False)
print("  Saved: train_clean.parquet")
print("  Saved: test_clean.parquet")

print("\nPreprocessing complete.")
