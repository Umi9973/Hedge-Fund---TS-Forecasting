"""
Step 3: Feature Engineering
- Temporal features: EWM, rolling mean/std, momentum (on top IC features only)
- Cross-sectional features: z-score, group z-score, rank percentile
- Spread & ratio features between high-IC feature pairs
- Categorical encoding: sub_category one-hot, sub_code frequency, feature_ch as-is
- All temporal features are strictly backward-looking (no future leakage)
- Train + test concatenated for feature computation, then split back
"""

import pandas as pd
import numpy as np
import gc
import time
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = "G:/Umi/Python Projects/TS Forecast"
VAL_SPLIT   = 3500

# Top features by IC from EDA (these get temporal engineering)
TOP_FEATURES = [
    'feature_bz', 'feature_ca', 'feature_by', 'feature_am', 'feature_u',
    'feature_ao', 'feature_cd', 'feature_cc', 'feature_cb', 'feature_az',
    'feature_bq', 'feature_ag', 'feature_ap', 'feature_br', 'feature_bp',
    'feature_bs', 'feature_bn', 'feature_bo', 'feature_al', 'feature_an',
]

# ============================================================
# LOAD
# ============================================================
print("Loading data...")
train = pd.read_parquet(f"{PROJECT_DIR}/train_clean.parquet")
test  = pd.read_parquet(f"{PROJECT_DIR}/test_clean.parquet")
print(f"  Train: {train.shape} | Test: {test.shape}")

feature_cols = [c for c in train.columns if c.startswith("feature_")]

# Tag rows so we can split back later
train['_is_test'] = False
test['_is_test']  = True

# test has no y_target/weight — add placeholders to allow concat
test['y_target'] = np.nan
test['weight']   = np.nan

# Combine and sort by instrument × horizon × time
df = pd.concat([train, test], ignore_index=True)
df = df.sort_values(['sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
print(f"  Combined: {df.shape}")

del train, test
gc.collect()

# ============================================================
# 1. CATEGORICAL ENCODING
# ============================================================
print("\n--- 1. Categorical Encoding ---")

# One-hot sub_category (5 values — low cardinality, safe to one-hot)
dummies = pd.get_dummies(df['sub_category'], prefix='subcat', dtype=np.int8)
df = pd.concat([df, dummies], axis=1)
print(f"  sub_category one-hot: {dummies.shape[1]} columns")

# Frequency encoding for sub_code (180 values — too many for one-hot)
# Computed from training split only to avoid leakage
train_split  = df[(df._is_test == False) & (df.ts_index <= VAL_SPLIT)]
freq_counts  = train_split['sub_code'].value_counts()
total_freq   = freq_counts.sum()

df['subcode_freq']      = df['sub_code'].map(freq_counts).fillna(1)
df['subcode_log_freq']  = np.log1p(df['subcode_freq'])
df['subcode_freq_rank'] = df['sub_code'].map(
    freq_counts.rank(ascending=False, method='dense')
).fillna(freq_counts.nunique() + 1)
print(f"  sub_code frequency encoding: 3 columns")

# feature_ch is int (0-10) — keep as-is, model handles it natively
print(f"  feature_ch: kept as int (0-10)")

del train_split
gc.collect()

# ============================================================
# 2. SPREAD & RATIO FEATURES
# ============================================================
print("\n--- 2. Spread & Ratio Features ---")

# Pairs chosen based on high IC and likely relatedness
spread_pairs = [
    ('feature_bz', 'feature_ca'),
    ('feature_by', 'feature_am'),
    ('feature_cd', 'feature_cc'),
    ('feature_az', 'feature_bq'),
    ('feature_ao', 'feature_u'),
    ('feature_al', 'feature_am'),
    ('feature_br', 'feature_bp'),
    ('feature_bs', 'feature_bn'),
]

count = 0
for f1, f2 in spread_pairs:
    if f1 in df.columns and f2 in df.columns:
        df[f'spread_{f1}_{f2}'] = df[f1] - df[f2]
        df[f'ratio_{f1}_{f2}']  = df[f1] / (df[f2].abs() + 1e-7)
        count += 2

print(f"  Created {count} spread/ratio columns")

# ============================================================
# 3. TEMPORAL FEATURES
# Per instrument × horizon, strictly backward-looking
# We group by sub_code + horizon so each instrument-horizon pair
# has its own independent time series history
# ============================================================
print("\n--- 3. Temporal Features ---")
print("  Computing EWM, rolling mean/std, momentum on top IC features...")
t0 = time.time()

grp = df.groupby(['sub_code', 'horizon'])

for i, feat in enumerate(TOP_FEATURES):
    if feat not in df.columns:
        continue

    print(f"  [{i+1}/{len(TOP_FEATURES)}] {feat}...")

    # EWM with 3 spans — short, medium, long
    # Recent timestamps weighted more heavily than older ones
    df[f'{feat}_ewm3']  = grp[feat].transform(
        lambda x: x.ewm(span=3,  adjust=False).mean()
    )
    df[f'{feat}_ewm7']  = grp[feat].transform(
        lambda x: x.ewm(span=7,  adjust=False).mean()
    )
    df[f'{feat}_ewm14'] = grp[feat].transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    )

    # Rolling mean — equal weight over window
    # Captures general trend over that window
    df[f'{feat}_rollmean5']  = grp[feat].transform(
        lambda x: x.rolling(5,  min_periods=1).mean()
    )
    df[f'{feat}_rollmean10'] = grp[feat].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    # Rolling std — volatility over window
    # High std = signal has been unstable recently
    df[f'{feat}_rollstd5']  = grp[feat].transform(
        lambda x: x.rolling(5,  min_periods=1).std().fillna(0)
    )
    df[f'{feat}_rollstd10'] = grp[feat].transform(
        lambda x: x.rolling(10, min_periods=1).std().fillna(0)
    )

    # Momentum — how much did the signal change since last timestamp?
    # Positive = rising, negative = falling
    df[f'{feat}_diff1'] = grp[feat].transform(
        lambda x: x.diff(1).fillna(0)
    )

print(f"  Temporal features done in {time.time()-t0:.1f}s")
print(f"  Created {len(TOP_FEATURES) * 8} temporal columns")

gc.collect()

# ============================================================
# 4. CROSS-SECTIONAL FEATURES
# Per timestamp — compares each instrument against all others
# at the same point in time. Safe because all instruments at
# the same ts_index are available simultaneously.
# ============================================================
print("\n--- 4. Cross-Sectional Features ---")
print("  Computing z-scores, group z-scores, rank percentiles...")
t0 = time.time()

# Use top 10 features for cross-sectional (most impactful)
cs_features = TOP_FEATURES[:10]

for feat in cs_features:
    if feat not in df.columns:
        continue

    # Z-score: how does this instrument compare to ALL others right now?
    # Removes the market-wide level effect
    ts_grp = df.groupby('ts_index')[feat]
    cs_mean = ts_grp.transform('mean')
    cs_std  = ts_grp.transform('std').fillna(1).replace(0, 1)
    df[f'{feat}_cs_z'] = (df[feat] - cs_mean) / cs_std

    # Rank percentile: outlier-robust version of z-score
    # 0.0 = lowest instrument, 1.0 = highest instrument at this timestamp
    df[f'{feat}_cs_rank'] = df.groupby('ts_index')[feat].rank(pct=True)

    # Group z-score: compare within (ts_index, sub_category) peer group
    # An instrument may look extreme vs market but normal within its category
    g_grp  = df.groupby(['ts_index', 'sub_category'])[feat]
    g_mean = g_grp.transform('mean')
    g_std  = g_grp.transform('std').fillna(1).replace(0, 1)
    df[f'{feat}_grp_z'] = (df[feat] - g_mean) / g_std

print(f"  Cross-sectional features done in {time.time()-t0:.1f}s")
print(f"  Created {len(cs_features) * 3} cross-sectional columns")

gc.collect()

# ============================================================
# 5. LAG FEATURES
# Exact value from N steps ago for each instrument × horizon
# Gives model direct access to recent history, not just smoothed averages
# ============================================================
print("\n--- 5. Lag Features ---")
t0 = time.time()

LAG_STEPS   = [1, 3, 5]
LAG_FEATURES = TOP_FEATURES[:10]  # top 10 IC features

for feat in LAG_FEATURES:
    if feat not in df.columns:
        continue
    for lag in LAG_STEPS:
        df[f'{feat}_lag{lag}'] = grp[feat].transform(
            lambda x: x.shift(lag)
        )

print(f"  Lag features done in {time.time()-t0:.1f}s")
print(f"  Created {len(LAG_FEATURES) * len(LAG_STEPS)} lag columns")

gc.collect()

# ============================================================
# 6. I2QQ2C5C FLAG
# This single instrument carries 17.4% of total weight — the model
# should know when it's dealing with this specific instrument
# ============================================================
print("\n--- 6. I2QQ2C5C Flag ---")

# Binary flag: 1 if this row is I2QQ2C5C, 0 otherwise
df['is_top_instrument'] = (df['sub_code'] == 'I2QQ2C5C').astype(np.int8)

# Interaction: flag × top 5 features
# Lets model learn completely different feature relationships for this instrument
for feat in TOP_FEATURES[:5]:
    if feat in df.columns:
        df[f'top_inst_{feat}'] = df['is_top_instrument'] * df[feat]

print(f"  Created 1 flag + {len(TOP_FEATURES[:5])} interaction columns")

gc.collect()

# ============================================================
# SPLIT BACK INTO TRAIN AND TEST
# ============================================================
print("\n--- Splitting back into train / test ---")

# Identify all newly created columns
original_cols = feature_cols + [
    'id', 'code', 'sub_code', 'sub_category', 'horizon',
    'ts_index', 'y_target', 'weight', '_is_test'
]
new_cols = [c for c in df.columns if c not in original_cols]
print(f"  New engineered columns: {len(new_cols)}")

# Fill any NaN from rolling/EWM at start of series with 0
df[new_cols] = df[new_cols].fillna(0)

# Split
train_out = df[df._is_test == False].drop(columns=['_is_test']).reset_index(drop=True)
test_out  = df[df._is_test == True].drop(
    columns=['_is_test', 'y_target', 'weight']
).reset_index(drop=True)

print(f"  Train: {train_out.shape}")
print(f"  Test:  {test_out.shape}")

# ============================================================
# SAVE
# ============================================================
print("\n--- Saving ---")
train_out.to_parquet(f"{PROJECT_DIR}/train_features.parquet", index=False)
test_out.to_parquet(f"{PROJECT_DIR}/test_features.parquet",   index=False)
print("  Saved: train_features.parquet")
print("  Saved: test_features.parquet")

print(f"\nFeature engineering complete.")
print(f"Total columns: {train_out.shape[1]} (was {len(feature_cols) + 6} before engineering)")
