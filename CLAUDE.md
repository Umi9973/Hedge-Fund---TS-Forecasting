# TS Forecast — Project Context

## Git / GitHub
- **Repo**: https://github.com/Umi9973/Hedge-Fund---TS-Forecasting (main branch)
- **ALWAYS ask user explicitly before any `git commit` or `git push`** — do not do either automatically

## Output Rules
- **ALL outputs must be saved inside `G:/Umi/Python Projects/TS Forecast/`** — never write to temp folders or anywhere outside this project directory
- Submissions go to `submissions/`
- Experiment outputs go to their own subfolder (e.g. `h3_experiment_submission/`)
- Every experiment script must save: val predictions, test submission, and a results summary text file

## Competition
- **Name**: Hedge Fund - Time Series Forecasting (Kaggle: `ts-forecasting`)
- **Deadline**: April 7, 2026 | **Goal**: Top 100 / 1035 teams
- **Submissions**: 1 per day max — every slot matters
- **Metric**: `Skill Score = sqrt(1 - clip(sum(w*(y-pred)^2) / sum(w*y^2), 0, 1))` — higher is better
- **Public LB = 25% of test, Private LB = 75%** — don't chase public LB rank
- **Notebook runtime limit**: 6 hours (for prize eligibility reproducibility)

## Data
- **Raw data location**: `G:/Umi/Python Projects/TS Forecast/data/raw/train.parquet` and `test.parquet`
- **Train**: 5.34M rows, `ts_index` 1–3601 | **Test**: 1.45M rows, `ts_index` 3602–4376
- **Target**: `y_target` (normalized financial return)
- **Key cols**: `id`, `code` (23), `sub_code` (180 instruments), `sub_category` (5), `horizon` (1/3/10/25), `ts_index`, `weight`
- **Features**: `feature_a` to `feature_ch` (82 anonymized signals; `feature_ch` is int 0–10, treat as categorical)
- **Weights**: extremely skewed — `I2QQ2C5C` alone = 17.4% of total training weight, top 28 instruments = 50%
- **DO NOT use `weight` as a model feature** — only as sample_weight in training and metric

## Critical Weight Insight
- **I2QQ2C5C only exists in ts_index 456–671** — it is completely absent from the validation period (3501–3601) and likely the test period
- **High-weight rows in the val period have y_target = 0** — any non-zero prediction on these rows explodes sum(w*(y-pred)^2) while sum(w*y^2) stays tiny → ratio >> 1 → score = 0
- **USE_LOG_WEIGHT must be False** — log-transforming weights causes the model to predict non-zero for high-weight y=0 rows, making CV = 0. Raw weights force the model to predict ≈0 where y=0
- **Do NOT use log weights for eval/early stopping either** — same problem applies

## Key EDA Findings
- Top IC features (Spearman ~0.09): `feature_bz`, `feature_ca`, `feature_by`, `feature_am`, `feature_u`
- Near-zero IC (noise): `feature_b`, `feature_c`, `feature_d`, `feature_e`, `feature_f`, `feature_g`, `feature_h`, `feature_i`
- 48 features have NaN — filled by sub_category group median (training split only), global median fallback
- Horizon 3 carries the most weight (~37.7%) — most important horizon
- Target variance grows with horizon (close to random walk variance scaling)
- Temporal distribution is not fully stable — recent timestamps may matter more

## Rules
- No future leakage — all temporal features strictly backward-looking
- Compute all fill values / encodings from `ts_index <= 3500` only
- Sequential prediction: prediction at ts=t must use only data from ts 1..t
- Submission: `id`, `prediction` columns only, no NaN
- Submission CSVs go to `submissions/` subfolder

## Validation Split
- Train: `ts_index <= 3500` | Validate: `ts_index 3501–3601`
- Local CV ≈ Public LB (0.2439 vs 0.2437) — validation is trustworthy, improve CV = improve LB

## Pipeline Files
- `01_preprocessing.py` — NaN fill, dtype downcast → `train_clean.parquet`, `test_clean.parquet`
- `02_eda.py` — EDA plots + CSV outputs → `eda_outputs/`
- `03_feature_engineering.py` — 344 total columns → `train_features.parquet`, `test_features.parquet`
- `04_model.py` — LightGBM per horizon, 3-seed ensemble, full refit → `submissions/`

## Engineered Features (328 model features after dropping low-IC)
- Raw (78): original features minus 8 low-IC dropped ones
- Categorical (8): sub_category one-hot (5), sub_code freq encoding (3)
- Spread/ratio (16): differences and ratios between high-IC feature pairs
- Temporal (160): EWM (3/7/14), rolling mean/std (5/10), diff1 — on top 20 IC features
- Cross-sectional (30): cs_z, cs_rank, grp_z — per (ts_index, horizon) — on top 10 IC features
- Lag (30): lag 1/3/5 on top 10 IC features
- I2QQ2C5C flag (6): binary flag + 5 interaction terms with top features

## Known Bugs Fixed
- `w_val[:len(y_train)]` shape mismatch in train_score diagnostic — fixed to use `w_train_raw`
- Unicode chars (⚠ ✓) crash Windows GBK console — replaced with ASCII
- CS features were grouped by `ts_index` only (mixing horizons) — fixed to `(ts_index, horizon)`
- Raw data paths pointed to project root — fixed to `data/raw/`

## Submissions
| Date | CV Score | Public LB | Rank | Notes |
|------|----------|-----------|------|-------|
| 2026-03-25 | 0.2439 | 0.2437 | 326/1035 | Baseline LightGBM, 3-seed ensemble |
| 2026-03-28 | 0.2403 | pending | — | Fixed pipeline, CS per-horizon grouping |

## Next Steps
- Submit 2026-03-28 run (CV 0.2403) and check LB
- Hyperparameter tuning with Optuna (start with horizon=3, highest weight)
- Ensemble with XGBoost / CatBoost for diversity
- Dedicated sub-model for high-weight val instruments
