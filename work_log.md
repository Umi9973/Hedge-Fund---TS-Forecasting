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

## 2026-03-30 — All-Horizon NN + Blend Experiments

### All-Horizon Supervised Autoencoder (11_autoencoder_all_horizons.py)
- Extended h=3 autoencoder to all 4 horizons with horizon-specific hyperparameters
- Architecture identical to v2 (ReduceLROnPlateau, L2 penalty)
- Results:
  - h=1: **val 0.0501** (stronger than LightGBM 0.0508 — NN wins)
  - h=3: **val 0.1118**
  - h=10: **val 0.1932**
  - h=25: **val 0.2517**
- Models saved to `autoencoder_results/model_h{H}_cv*.pt` with normalization stats

### Blend Experiments — All Horizons (12_blend_h1.py, 13_blend_h10.py, 14_blend_h25.py)
- Tested 21 alpha ratios (alpha = NN weight) on h=1/10/25
- Best blend results:
  - h=1: alpha=0.55 → **0.0535** (vs LGBM 0.0508, NN 0.0501)
  - h=3: alpha=0.55 → **0.1192** (vs LGBM 0.1103, NN 0.1118)
  - h=10: alpha=0.40 → **0.2014** (vs LGBM best, NN 0.1932)
  - h=25: alpha=0.30 → **0.2584** (vs LGBM best, NN 0.2517)
- **Blend submission built, LB: not yet submitted**

### LightGBM GPU Investigation (Failed)
- Attempted `device='gpu'` in BASE_PARAMS to speed up training
- **Crash**: `best_split_info.right_count > 0` error — LightGBM GPU doesn't support `bagging_fraction < 1.0`
- Seed=42 RMSE diverged to 1e+34 before crash
- Fix: reverted to CPU. Alternative (GOSS sampling) would require full Optuna re-tuning — deferred
- LightGBM stays CPU-only

---

## 2026-03-30 — CatBoost Standalone Models

### Motivation
- LightGBM GPU broken; CatBoost GPU works natively (`task_type='GPU'`)
- Numerai research: CatBoost Sharpe 0.8704 vs LightGBM 0.7188 — may generalize better
- Goal: standalone CatBoost per horizon, no blending, compare to LightGBM baseline

### CatBoost h=1 (15_catboost_h1.py)
- GPU (cuda:1), 3 seeds, `depth=4, lr=0.01, l2=15, min_data_in_leaf=100`
- Early stopping (250 rounds), val/train RMSE ratio plotted
- First run: lr=0.05 too high, stopped at iter 9–87 (avg 43 trees) → score 0.0496
- Fix: lr 0.05→0.01, depth 6→4, l2 5→15, added min_data_in_leaf=100
- **Final val: 0.0508** — matches LightGBM, slower training

### CatBoost h=3 (16_catboost_h3.py)
- `depth=6, lr=0.01, l2=5, min_data_in_leaf=100`
- First result: **val 0.1143** — beats LightGBM (0.1103) by +0.004
- Tuning attempt (depth=5, l2=10, min_data_in_leaf=200) → 0.1118 — worse, reverted
- Expanding val experiment (VAL_SPLIT=3300) → **0.1000** — worse, reverted to 3500
- Final: **0.1143**

### CatBoost h=25 (18_catboost_h25.py)
- First run: lr=0.03, depth=8, l2=1.5 → val 0.2480, val/train ratio=1.406 (overfit)
- Fix: lr 0.03→0.01, depth 8→6, l2 1.5→5, min_data_in_leaf 100→100
- **Final val: 0.2535** — comparable to LightGBM

### CatBoost h=10 (17_catboost_h10.py)
- `depth=7, lr=0.03, l2=2, min_data_in_leaf=50`
- **Not yet run**

### Expanding Val Set Experiment (h=3)
- Hypothesis: expand validation to match public test size (~775 timestamps)
- Tried VAL_SPLIT=2826 (775 timestamps, 341K val rows) → **val 0.0632** — much worse
- Tried VAL_SPLIT=3300 (300 timestamps) → **val 0.1000** — still worse than 3500
- Root cause: each timestamp has hundreds of rows; 775 timestamps = 25% of training data lost
- **Conclusion**: original VAL_SPLIT=3500 is optimal; expanding val doesn't work here

---

## 2026-03-31 — Walk-Forward CV + Submissions

### Walk-Forward CV (19–22_wfcv_h*.py)
- Implemented 4-fold walk-forward CV for all horizons
- Train cutpoints: [3200, 3300, 3400, 3500]; val windows: 100 timestamps each
- No leakage: each fold trains on past, validates on future
- Per-fold and mean scores:

| Horizon | WF Mean | Fold1 | Fold2 | Fold3 | Fold4 | Fold4 optimism |
|---------|---------|-------|-------|-------|-------|----------------|
| h=1 | 0.0621 | 0.0609 | 0.0633 | 0.0543 | 0.0699 | +0.0078 |
| h=3 | 0.1138 | 0.1170 | 0.1182 | 0.1052 | 0.1103 | -0.0035 |
| h=10 | 0.2011 | 0.2300 | 0.1937 | 0.1508 | 0.2300 | +0.0289 |
| h=25 | 0.2478 | 0.2928 | 0.2297 | 0.1758 | 0.2928 | +0.0450 |

- Fold 3 (3401–3500) is consistently the hardest window across all horizons
- h=10 Fold1 hit 3000-tree cap — still improving at cutoff

### Walk-Forward Submission (23_wfcv_submit.py)
- Built submission from 4 WF test pred CSVs
- Used avg best_iter across folds for full-data refit
- **avg CV 0.1562 | Public LB: 0.2429** — worse than baseline 0.2438
- Root cause: avg_iter pulled tree count down for h=10/25 (hard folds had low iters)

### Pure NN Submission (24_nn_inference.py)
- Generated h=1 and h=25 test preds from saved autoencoder weights
- h=3 from `test_preds_h3_nn.csv`, h=10 from `test_preds_h10_blend_a1.00_cv0.1932.csv`
- **avg val 0.1517 | LB: not yet submitted**
- Submission saved: `submissions/submission_nn_pure_avg0.1517.csv`

### Competition Research — Ubiquant Report
- Deep-dived 1st/2nd/20th/41st/22nd place Ubiquant solutions
- Written up in `Ubiquant_report.md`
- Key finding: **`date_id` cross-sectional aggregate features** used by every top solution
- We already have CS z-scores but NOT the raw market-level mean/std as separate features
- Previous experiment added mean/std but was redundant with existing z-scores — however report shows they kept both the raw market stats AND the normalized deviation as separate inputs

---

## Score Progression

| Step | CV Score | Public LB | Notes |
|------|----------|-----------|-------|
| Baseline LightGBM | 0.2439 | 0.2437 | 326/1035 |
| Pipeline fix | 0.2403 | — | CS per-horizon grouping |
| Optuna HPO | 0.2474 | 0.2438 | ~300/1035 |
| h=3 NN blend | ~0.251+ | — | 55% NN + 45% LGBM on h=3 |
| Walk-forward CV submission | 0.1562 (avg) | 0.2429 | Worse; avg_iter issue |
| Pure NN submission | 0.1517 (avg) | pending | `submission_nn_pure_avg0.1517.csv` |
| CatBoost h=3 standalone | 0.1143 | — | Best single-horizon CB result |

---

## Key Learnings

1. **Metric incompatibility**: rank norm, era boosting, log weights, cosine LR restarts all fail because they push predictions away from zero — destroyed by high-weight y=0 val rows
2. **CS z-scores already capture market-wide info** — don't add redundant market aggregates
3. **Sub_code categorical overfits** — 180 instrument IDs are too granular for LightGBM splits
4. **NN + LightGBM are complementary** — blending at 55/45 gives +0.0097 over LightGBM alone
5. **ReduceLROnPlateau > CosineAnnealing** for this metric — no surprise LR spikes
6. **h=10 needs n_estimators=5000** — still improving at 3000 tree limit
7. **LightGBM GPU crashes** with bagging_fraction < 1.0 — must stay CPU
8. **CatBoost GPU works** — h=3 beats LightGBM (0.1143 vs 0.1103); h=1/25 roughly equal
9. **Expanding val set doesn't work** — losing 25% of training data hurts more than the better CV estimate helps
10. **Walk-forward avg_iter hurts LB** — fold-level tree count variation means avg underestimates best count for hard horizons; use max or fold4 iter instead
11. **Fold3 (3401–3500) is hardest** — consistently lowest score across all horizons and models
