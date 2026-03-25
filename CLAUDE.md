# TS Forecast — Project Context

## Competition
- **Name**: Hedge Fund - Time Series Forecasting (Kaggle: `ts-forecasting`)
- **Deadline**: April 7, 2026 | **Goal**: Top 100 / 1035 teams
- **Submissions**: 1 per day max — every slot matters
- **Metric**: `Skill Score = sqrt(1 - clip(sum(w*(y-pred)^2) / sum(w*y^2), 0, 1))` — higher is better
- **Public LB = 25% of test, Private LB = 75%** — don't chase public LB rank
- **Notebook runtime limit**: 6 hours (for prize eligibility reproducibility)

## Data
- **Location**: `G:/Umi/Python Projects/TS Forecast/`
- **Train**: 5.34M rows, `ts_index` 1–3601 | **Test**: 1.45M rows, `ts_index` 3602–4376
- **Target**: `y_target` (normalized financial return)
- **Key cols**: `id`, `code` (23), `sub_code` (180 instruments), `sub_category` (5), `horizon` (1/3/10/25), `ts_index`, `weight`
- **Features**: `feature_a` to `feature_ch` (82 anonymized signals; `feature_ch` is int 0–10, treat as categorical)
- **Weights**: extremely skewed — `I2QQ2C5C` alone = 17.4% of total weight, top 28 instruments = 50%
- **DO NOT use `weight` as a model feature** — only as sample_weight in training and metric

## Key EDA Findings
- Top IC features (Spearman ~0.09): `feature_bz`, `feature_ca`, `feature_by`, `feature_am`, `feature_u`
- Near-zero IC (noise): `feature_b`, `feature_c`, `feature_d`, `feature_e`, `feature_f`, `feature_g`
- 48 features have NaN — filled by sub_category group median (training split only), global median fallback
- Horizon 3 carries the most weight (~37.7%) — most important horizon
- Target variance grows with horizon (close to random walk variance scaling)
- Temporal distribution is not fully stable — recent timestamps may matter more

## Rules
- No future leakage — all temporal features strictly backward-looking
- Compute all fill values / encodings from `ts_index <= 3500` only
- Sequential prediction: prediction at ts=t must use only data from ts 1..t
- Submission: `id`, `prediction` columns only, no NaN

## Validation Split
- Train: `ts_index <= 3500` | Validate: `ts_index 3501–3601`
- Local CV ≈ Public LB (0.2439 vs 0.2437) — validation is trustworthy, improve CV = improve LB

## Pipeline Files
- `01_preprocessing.py` — NaN fill, dtype downcast → `train_clean.parquet`, `test_clean.parquet`
- `02_eda.py` — EDA plots + CSV outputs → `eda_outputs/`
- `03_feature_engineering.py` — 308 total features → `train_features.parquet`, `test_features.parquet`
- `04_model.py` — LightGBM per horizon, 3-seed ensemble, full refit → `submissions/`

## Engineered Features (308 total)
- Raw (86): original features + feature_ch
- Categorical (8): sub_category one-hot (5), sub_code freq encoding (3)
- Spread/ratio (16): differences and ratios between high-IC feature pairs
- Temporal (160): EWM (3/7/14), rolling mean/std (5/10), diff1 — on top 20 IC features
- Cross-sectional (30): cs_z, cs_rank, grp_z — on top 10 IC features
- Lag (30): lag 1/3/5 on top 10 IC features
- I2QQ2C5C flag (6): binary flag + 5 interaction terms with top features

## Submissions
| Date | CV Score | Public LB | Rank | Notes |
|------|----------|-----------|------|-------|
| 2026-03-25 | 0.2439 | 0.2437 | 326/1035 | Baseline LightGBM, 3-seed ensemble |

## Next Steps
- Hyperparameter tuning with Optuna (start with horizon=3, highest weight)
- Ensemble with XGBoost / CatBoost for diversity
- Weight transformation: try log(1+w) as sample_weight
- Dedicated sub-model for I2QQ2C5C
