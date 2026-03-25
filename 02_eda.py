"""
Step 2: Exploratory Data Analysis
- Feature signal strength (Spearman IC vs y_target)
- y_target distribution per horizon
- Weight distribution and concentration
- Null patterns
- feature_ch deep dive
- Temporal stability
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file, no window needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = "G:/Umi/Python Projects/TS Forecast"
TRAIN_PATH  = f"{PROJECT_DIR}/train_clean.parquet"
VAL_SPLIT   = 3500

plt.style.use('dark_background')
COLORS = ['#2a7fbf', '#0e9e8e', '#c46e2a', '#8a3fba', '#c43a3a']

print("Loading cleaned training data...")
train = pd.read_parquet(TRAIN_PATH)
print(f"  Shape: {train.shape}")

feature_cols = [c for c in train.columns if c.startswith("feature_")]
train_split  = train[train.ts_index <= VAL_SPLIT]  # training split only

# ============================================================
# 1. FEATURE SIGNAL STRENGTH (Spearman IC)
# ============================================================
print("\n--- 1. Feature Signal Strength (Spearman IC) ---")

ics = {}
for feat in feature_cols:
    ic, _ = spearmanr(train_split[feat], train_split['y_target'])
    ics[feat] = ic

ics_sorted = dict(sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True))
ic_df = pd.DataFrame({'feature': list(ics_sorted.keys()), 'IC': list(ics_sorted.values())})
ic_df['abs_IC'] = ic_df['IC'].abs()

print(f"\n  Top 15 features by |IC|:")
print(ic_df.head(15).to_string(index=False))
print(f"\n  Bottom 10 features (weakest signal):")
print(ic_df.tail(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 8))
top30 = ic_df.head(30)
bar_colors = [COLORS[0] if v > 0 else COLORS[4] for v in top30['IC']]
ax.barh(top30['feature'][::-1], top30['IC'][::-1], color=bar_colors[::-1], edgecolor='none')
ax.axvline(0,     color='white', linewidth=0.8, alpha=0.5)
ax.axvline(0.02,  color='#ffcc44', linewidth=1.0, linestyle='--', alpha=0.7, label='IC=±0.02')
ax.axvline(-0.02, color='#ffcc44', linewidth=1.0, linestyle='--', alpha=0.7)
ax.set_title('Top 30 Features by Spearman IC vs y_target', fontsize=14, fontweight='bold')
ax.set_xlabel('Spearman IC')
ax.legend()
ax.grid(True, axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/eda_feature_ic.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: eda_feature_ic.png")

# ============================================================
# 2. y_target DISTRIBUTION PER HORIZON
# ============================================================
print("\n--- 2. y_target Distribution per Horizon ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, horizon in enumerate([1, 3, 10, 25]):
    yt = train_split[train_split.horizon == horizon]['y_target']
    clipped = yt.clip(yt.quantile(0.01), yt.quantile(0.99))

    print(f"  Horizon {horizon:2d} | mean={yt.mean():.4f} | std={yt.std():.4f} | "
          f"skew={yt.skew():.3f} | min={yt.min():.2f} | max={yt.max():.2f}")

    axes[i].hist(clipped, bins=80, color=COLORS[i], alpha=0.85, edgecolor='none')
    axes[i].axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[i].set_title(f'Horizon {horizon} — y_target (clipped 1–99%)', fontweight='bold')
    axes[i].set_xlabel('y_target')
    axes[i].set_ylabel('Count')
    axes[i].text(0.97, 0.95, f'std={yt.std():.4f}\nskew={yt.skew():.3f}',
                 transform=axes[i].transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='#1a3050', alpha=0.6))

plt.suptitle('y_target Distribution by Horizon', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/eda_target_dist.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: eda_target_dist.png")

# ============================================================
# 3. WEIGHT DISTRIBUTION & CONCENTRATION
# ============================================================
print("\n--- 3. Weight Distribution ---")

w = train_split['weight']
print(f"  Mean:   {w.mean():.2f}")
print(f"  Median: {w.median():.2f}")
print(f"  Max:    {w.max():.2e}")
print(f"  % rows with weight=0: {(w==0).mean()*100:.2f}%")

# Weight concentration by sub_code
wt_by_subcode = train_split.groupby('sub_code')['weight'].sum()
wt_by_subcode = wt_by_subcode.sort_values(ascending=False)
total_weight  = wt_by_subcode.sum()
top5_pct      = wt_by_subcode.head(5).sum() / total_weight * 100
top10_pct     = wt_by_subcode.head(10).sum() / total_weight * 100
print(f"\n  Top 5 instruments  = {top5_pct:.1f}% of total weight")
print(f"  Top 10 instruments = {top10_pct:.1f}% of total weight")
print(f"\n  Top 10 instruments by weight:")
print((wt_by_subcode.head(10) / total_weight * 100).round(2).to_string())

# Weight by horizon
wt_by_horizon = train_split.groupby('horizon')['weight'].sum() / total_weight * 100
print(f"\n  Weight share by horizon:")
print(wt_by_horizon.round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Lorenz curve (weight concentration)
w_sorted  = w.sort_values(ascending=False).reset_index(drop=True)
cumsum_w  = w_sorted.cumsum() / w_sorted.sum()
pct_rows  = np.linspace(0, 100, len(cumsum_w))
axes[0].plot(pct_rows, cumsum_w.values * 100, color=COLORS[0], linewidth=2)
axes[0].plot([0, 100], [0, 100], 'w--', linewidth=0.8, alpha=0.4)
axes[0].set_title('Weight Concentration (Lorenz Curve)', fontweight='bold')
axes[0].set_xlabel('% of rows (sorted by weight desc)')
axes[0].set_ylabel('% cumulative weight')
axes[0].grid(True, alpha=0.2)

# Top 20 instruments by weight
top20 = wt_by_subcode.head(20) / total_weight * 100
axes[1].barh(range(20), top20.values[::-1], color=COLORS[1], edgecolor='none')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top20.index[::-1], fontsize=7)
axes[1].set_title('Top 20 Instruments by Weight Share (%)', fontweight='bold')
axes[1].set_xlabel('% of total weight')
axes[1].grid(True, axis='x', alpha=0.2)

# Weight by horizon
axes[2].bar(wt_by_horizon.index.astype(str), wt_by_horizon.values, color=COLORS[2], edgecolor='none')
axes[2].set_title('Weight Share by Horizon (%)', fontweight='bold')
axes[2].set_xlabel('Horizon')
axes[2].set_ylabel('% of total weight')
axes[2].grid(True, axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/eda_weight.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: eda_weight.png")

# ============================================================
# 4. NULL PATTERNS
# ============================================================
print("\n--- 4. Null Patterns ---")

# Are null rows the same rows across features?
raw_train = pd.read_parquet(f"{PROJECT_DIR}/train.parquet")
null_mask  = raw_train[feature_cols].isnull()
null_counts_per_row = null_mask.sum(axis=1)

print(f"  Rows with 0 nulls:   {(null_counts_per_row == 0).sum():,} ({(null_counts_per_row == 0).mean()*100:.1f}%)")
print(f"  Rows with 1-5 nulls: {((null_counts_per_row >= 1) & (null_counts_per_row <= 5)).sum():,}")
print(f"  Rows with >5 nulls:  {(null_counts_per_row > 5).sum():,}")
print(f"  Max nulls in one row: {null_counts_per_row.max()}")

del raw_train

# ============================================================
# 5. feature_ch DEEP DIVE
# ============================================================
print("\n--- 5. feature_ch Deep Dive ---")

print(f"  Unique values: {sorted(train['feature_ch'].unique())}")
print(f"\n  Value counts:")
print(train['feature_ch'].value_counts().sort_index().to_string())

# Relationship with sub_category and code
print(f"\n  feature_ch vs sub_category:")
print(pd.crosstab(train['feature_ch'], train['sub_category'], normalize='index').round(3).to_string())

# IC of feature_ch with y_target per horizon
print(f"\n  Spearman IC of feature_ch vs y_target per horizon:")
for h in [1, 3, 10, 25]:
    sub = train_split[train_split.horizon == h]
    ic, _ = spearmanr(sub['feature_ch'], sub['y_target'])
    print(f"    Horizon {h:2d}: IC = {ic:.5f}")

# ============================================================
# 6. TEMPORAL STABILITY
# ============================================================
print("\n--- 6. Temporal Stability ---")

# Rolling mean and std of y_target over time (horizon=1 only for clarity)
h1 = train[train.horizon == 1].groupby('ts_index')['y_target'].agg(['mean', 'std']).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(h1['ts_index'], h1['mean'], color=COLORS[0], linewidth=1.0, alpha=0.8)
axes[0].axvline(VAL_SPLIT, color='#ff6b6b', linewidth=1.5, linestyle='--', label='Val split (3500)')
axes[0].set_title('y_target Mean over Time (Horizon=1)', fontweight='bold')
axes[0].set_ylabel('Mean y_target')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

axes[1].plot(h1['ts_index'], h1['std'], color=COLORS[1], linewidth=1.0, alpha=0.8)
axes[1].axvline(VAL_SPLIT, color='#ff6b6b', linewidth=1.5, linestyle='--', label='Val split (3500)')
axes[1].set_title('y_target Std over Time (Horizon=1)', fontweight='bold')
axes[1].set_xlabel('ts_index')
axes[1].set_ylabel('Std y_target')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.suptitle('Temporal Stability — Does distribution shift over time?', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/eda_temporal.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: eda_temporal.png")

# ============================================================
# SAVE EDA OUTPUTS TO CSV
# ============================================================
print("\n--- Saving EDA outputs to CSV ---")

EDA_DIR = f"{PROJECT_DIR}/eda_outputs"
import os
os.makedirs(EDA_DIR, exist_ok=True)

# 1. Feature IC ranking
ic_df.to_csv(f"{EDA_DIR}/eda_feature_ic.csv", index=False)
print("  Saved: eda_feature_ic.csv")

# 2. Weight by instrument
wt_by_subcode_df = pd.DataFrame({
    'sub_code':       wt_by_subcode.index,
    'total_weight':   wt_by_subcode.values,
    'weight_share_%': (wt_by_subcode / total_weight * 100).round(4).values,
    'cumulative_%':   (wt_by_subcode / total_weight * 100).cumsum().round(4).values,
})
wt_by_subcode_df.to_csv(f"{EDA_DIR}/eda_weight_by_instrument.csv", index=False)
print("  Saved: eda_weight_by_instrument.csv")

# 3. Weight by horizon
wt_by_horizon.reset_index().rename(columns={'weight': 'weight_share_%'}).to_csv(
    f"{EDA_DIR}/eda_weight_by_horizon.csv", index=False
)
print("  Saved: eda_weight_by_horizon.csv")

# 4. Target stats by horizon
target_stats = []
for h in [1, 3, 10, 25]:
    yt = train_split[train_split.horizon == h]['y_target']
    target_stats.append({
        'horizon': h,
        'mean':    round(yt.mean(), 6),
        'std':     round(yt.std(), 6),
        'skew':    round(yt.skew(), 4),
        'min':     round(yt.min(), 4),
        'max':     round(yt.max(), 4),
        'p1':      round(yt.quantile(0.01), 4),
        'p99':     round(yt.quantile(0.99), 4),
    })
pd.DataFrame(target_stats).to_csv(f"{EDA_DIR}/eda_target_stats_by_horizon.csv", index=False)
print("  Saved: eda_target_stats_by_horizon.csv")

# 5. Null counts per feature (from raw train)
raw_null = pd.read_parquet(f"{PROJECT_DIR}/train.parquet")[feature_cols].isnull().sum()
null_df = pd.DataFrame({
    'feature':     raw_null.index,
    'null_count':  raw_null.values,
    'null_rate_%': (raw_null / len(train) * 100).round(4).values,
}).sort_values('null_count', ascending=False)
null_df.to_csv(f"{EDA_DIR}/eda_null_counts.csv", index=False)
print("  Saved: eda_null_counts.csv")

print("\nEDA complete.")
