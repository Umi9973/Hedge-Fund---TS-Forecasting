# Hedge Fund Time Series Forecasting

Kaggle competition: predict financial instrument returns across 4 horizons (h=1, 3, 10, 25).

**Best Public LB: 0.2438** (~rank 300/1035)

**Metric:** `Skill Score = sqrt(1 - clip(Σw(y-ŷ)² / Σwy², 0, 1))`
> High-weight rows have `y_target=0` — any non-zero prediction on these rows destroys the score.

---

## Score Progression

| Experiment | CV | Public LB | Notes |
|---|---|---|---|
| Baseline LightGBM | 0.2439 | 0.2437 | 326/1035 |
| Optuna HPO | 0.2474 | **0.2438** | ~300/1035, best LB |
| Walk-forward CV submission | 0.1562 (avg) | 0.2429 | avg_iter issue |
| Skew/kurt features + blend | 0.2454 | 0.2400 | v2 features, overfit |

---

## Project Structure

```
1_preprocessing/    Raw data → features (v1: 344 feat, v2: +40 skew/kurt = 384)
2_eda_outputs/      IC analysis, target distribution, weight stats
3_models/
    lgbm/           LightGBM (GBDT, DART, Optuna, skew/kurt variants)
    catboost/       CatBoost per horizon (GPU)
    wfcv/           Walk-forward cross-validation
    nn/             Supervised Autoencoder + MLP
4_results/          Optuna study outputs
5_submissions/      Blend scripts and submission builders
work_log.md         Full experiment history with scores
```

---

## Models & Results

### LightGBM (best overall)
- Optuna-tuned per horizon (30 TPE trials, 6.5 hrs), 3-seed ensemble
- **CV: 0.2474 | LB: 0.2438**
- h=10 hits 3000-tree cap — needs `n_estimators=5000`

### CatBoost (GPU)
- Runs on GPU natively (LightGBM GPU crashes with `bagging_fraction < 1.0`)
- h=3: **0.1143** (beats LightGBM 0.1103)
- h=1: 0.0508, h=25: 0.2535

### Supervised Autoencoder + MLP
- Architecture: `Input → Encoder(256→128→32) → Decoder` + skip connection to predictor
- `ReduceLROnPlateau` + L2 prediction penalty (prevents score collapse)
- h=3 single seed: **0.1118**, 3-seed ensemble: **0.1163**

### NN + LightGBM Blend
- h=3: alpha=0.55 (55% NN) → **0.1192** (+0.009 vs LightGBM)
- h=3 multiseed blend: **0.1216**

### Walk-Forward CV (4 folds)
| Horizon | WF Mean | Fold 3 (hardest) | Fold 4 |
|---------|---------|-----------------|--------|
| h=1  | 0.0621 | 0.0543 | 0.0699 |
| h=3  | 0.1138 | 0.1052 | 0.1103 |
| h=10 | 0.2011 | 0.1508 | 0.2300 |
| h=25 | 0.2478 | 0.1758 | 0.2928 |

---

## Features

- **v1 (344)**: raw features, spread/ratio, EWM/rolling temporal, CS z-score/rank, lag, flag features
- **v2 (384)**: v1 + CS skewness & kurtosis for top-20 IC features

Top IC features: `feature_bz`, `feature_ca`, `feature_by`, `feature_am`, `feature_u` (~IC 0.09)

---

## Key Learnings

1. **Metric trap**: rank normalization, era boosting, cosine LR restarts all fail — they push predictions away from zero and destroy score on high-weight `y=0` rows
2. **CS z-scores already encode market-wide info** — redundant to add raw market mean/std separately
3. **NN + LightGBM are complementary** — blending at 55/45 gives +0.009 over LightGBM alone
4. **Walk-forward avg_iter underestimates** — use fold4 or max iter for full refit, not avg
5. **Fold 3 (ts_index 3401–3500) is consistently hardest** across all horizons and models
6. **Expanding val doesn't help** — losing 25% of training data hurts more than the better CV estimate

---

See [work_log.md](work_log.md) for full experiment details.
