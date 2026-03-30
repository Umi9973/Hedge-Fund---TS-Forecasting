# Work Log — Hedge Fund TS Forecasting

---

## 2026-03-25

### Baseline Submission
- Built full ML pipeline from scratch: preprocessing → EDA → feature engineering → LightGBM model
- **328 engineered features**: raw, spread/ratio, temporal (EWM/rolling), cross-sectional (z-score/rank), lag, I2QQ2C5C flag
- **3-seed LightGBM ensemble** per horizon (h=1/3/10/25), full refit on all data
- **CV: 0.2439 | Public LB: 0.2437 | Rank: 326/1035**

### Critical Bugs Fixed
- CS features grouped by `ts_index` only (mixing horizons) → fixed to `(ts_index, horizon)`
- Raw data paths pointed to project root → fixed to `data/raw/`
- Unicode chars (⚠ ✓) crash Windows GBK console → replaced with ASCII

---

## 2026-03-26 — 2026-03-27

### EDA & Feature IC Analysis
- Computed Spearman IC for all 82 raw features
- Top IC: `feature_bz`, `feature_ca`, `feature_by`, `feature_am`, `feature_u` (~0.09)
- Near-zero IC (dropped): `feature_b/c/d/e/f/g/h/i`
- Key insight: high-weight val rows have `y_target=0` — any non-zero prediction on these rows destroys the skill score metric

### Failed Experiments: h=3 clipping
- Tried clipping y_target extreme values before training
- No improvement — h=3 has weak signal (kurtosis=244), model naturally learns near-zero predictions

### Failed Experiment: Rank Normalization (06_rank_normalize.py)
- Post-process predictions by rank-normalizing to gaussian/uniform
- **Result: CV 0.1051 → 0.0000** — rank norm spreads predictions away from zero, destroying score on high-weight y=0 val rows
- Fundamental incompatibility with metric; crossed off permanently

---

## 2026-03-27 — Feature Engineering Experiments

### Failed Experiment: Market-Wide Aggregates
- Added per `(ts_index, horizon)` mean/std for top-10 IC features + idiosyncratic deviation
- **h=3: -0.0098, h=10: -0.0027** — redundant with existing CS z-scores which already capture market-level info
- Reverted; saved baseline features to `features_baseline_cv0.2403/`

### Failed Experiment: Feature Group Statistics
- Row-level mean/std/skew across IC-ranked feature groups
- Pure noise, hurt scores — reverted

### Failed Experiment: Sub_code as Categorical
- LightGBM native categorical for 180 instrument IDs
- **h=3: -0.0147, h=25: -0.0037** — instrument-specific splits overfit to training period behavior
- Crossed off permanently

---

## 2026-03-28 — Optuna Hyperparameter Tuning

### 08_optuna_tune.py
- 30 Bayesian TPE trials per horizon, single seed per trial for speed
- Runtime: **391.6 min (~6.5 hours)** overnight
- Single-seed results: h=1: 0.0789, h=3: 0.1273, h=10: 0.2297, h=25: 0.2940
- Applied best params to `04_model.py` with 3-seed ensemble
- **CV: 0.2474 | Public LB: 0.2438 | ~Rank 300/1035**
- Note: h=10 hit 3000 tree limit (best_iter=2996) — needs n_estimators=5000 next run

### Failed Experiment: Era Boosting (05_h3_experiment.py)
- Iteratively retrain on worst-performing dates (bottom 50% by score)
- **h=3: 0.1103 → 0.0923** — worst dates are dates with high-weight y=0 rows; retraining makes predictions MORE non-zero
- Same fundamental incompatibility as rank norm; crossed off permanently

---

## 2026-03-29 — Neural Network Experiments

### Disk Space Issue (Resolved)
- PyTorch 2.6 (2.5GB download) failed on C drive (OSError: no space left)
- Fixed: redirected pip cache + TEMP to G drive, then moved torch/torchvision to `G:/PyPackages`
- Added `.pth` file at `C:/Users/17464/AppData/Local/Programs/Python/Python313/Lib/site-packages/torch_g_drive.pth`

### Supervised Autoencoder v1 (09_autoencoder_h3.py — Run 1)
- Architecture: Input(328) → Encoder(256→128→32 bottleneck) → Decoder(128→256→328) + concat(Bottleneck, Input) → MLP(256→128→1)
- Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
- **Result: 0.0944** — cosine LR warm restarts caused score to collapse to 0.0000 at restart epochs
- Same issue: LR spikes push predictions away from zero on high-weight y=0 rows

### Supervised Autoencoder v2 (09_autoencoder_h3.py — Run 2)
- Fixes: ReduceLROnPlateau (no restarts) + L2 prediction penalty (1e-4 * mean(pred²)) + patience=15
- **Result: 0.1118 at epoch 35** — beats LightGBM baseline of 0.1103
- Runtime: 31.9 min on RTX 3060 (cuda:1)
- Output saved to `autoencoder_results/`

### Blend Search (10_blend_h3.py)
- Retrained h=3 LightGBM (3 seeds, Optuna params) → val score: 0.1095
- Tested 21 blend ratios (alpha=0.0 to 1.0, step 0.05) — alpha = NN weight
- **Best: alpha=0.55 (55% NN + 45% LightGBM) → val score 0.1192 (+0.0097 vs LightGBM)**
- Full blend table:

| Alpha (NN%) | Score  |
|-------------|--------|
| 0.00        | 0.1095 |
| 0.10        | 0.1129 |
| 0.20        | 0.1155 |
| 0.30        | 0.1174 |
| 0.40        | 0.1186 |
| 0.45        | 0.1189 |
| **0.55**    | **0.1192** |
| 0.60        | 0.1190 |
| 0.70        | 0.1182 |
| 1.00        | 0.1118 |

---

## Score Progression

| Step | CV Score | Public LB | Notes |
|------|----------|-----------|-------|
| Baseline LightGBM | 0.2439 | 0.2437 | 326/1035 |
| Pipeline fix | 0.2403 | — | CS per-horizon grouping |
| Optuna HPO | 0.2474 | 0.2438 | ~300/1035 |
| h=3 NN blend (pending submission) | ~0.251+ | — | 55% NN + 45% LGBM on h=3 |

---

## Key Learnings

1. **Metric incompatibility**: rank norm, era boosting, log weights, cosine LR restarts all fail because they push predictions away from zero — destroyed by high-weight y=0 val rows
2. **CS z-scores already capture market-wide info** — don't add redundant market aggregates
3. **Sub_code categorical overfits** — 180 instrument IDs are too granular for LightGBM splits
4. **NN + LightGBM are complementary** — blending at 55/45 gives +0.0097 over LightGBM alone
5. **ReduceLROnPlateau > CosineAnnealing** for this metric — no surprise LR spikes
6. **h=10 needs n_estimators=5000** — still improving at 3000 tree limit
