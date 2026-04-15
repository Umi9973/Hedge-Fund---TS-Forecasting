# Hedge Fund Time Series Forecasting

Kaggle competition: predict returns across 4 horizons (h=1, 3, 10, 25) for financial instruments.

**Best public LB: 0.2438** | Metric: Skill Score = `sqrt(1 - clip(Σw(y-ŷ)²/Σwy², 0, 1))`

---

## Structure

```
├── preprocessing/   Feature engineering, EDA, rank normalization, skew/kurt features
├── lgbm/            LightGBM experiments (GBDT, DART, Optuna tuning, skew/kurt variants)
├── catboost/        CatBoost experiments per horizon
├── wfcv/            Walk-forward cross-validation experiments
├── nn/              Supervised Autoencoder + MLP experiments
├── submissions/     Blend and submission scripts
└── work_log.md      Experiment log with all results
```

---

## Models

| Model | Best CV | Notes |
|-------|---------|-------|
| LightGBM (GBDT) | 0.2454 | Optuna-tuned per horizon, 3 seeds |
| CatBoost | — | Per horizon, learning curve saved |
| NN Autoencoder | 0.1163 (h=3) | Supervised autoencoder + skip connection, 3 seeds |
| NN + LGBM Blend | 0.1216 (h=3) | Best blend alpha=0.60 |

---

## Features

- **v1 (344 features)**: CS z-scores, rolling stats, rank normalization
- **v2 (384 features)**: v1 + CS skewness/kurtosis for top-20 IC features

---

## Key Files

| File | Purpose |
|------|---------|
| `preprocessing/01_preprocessing.py` | Raw data → train/test parquet |
| `preprocessing/03_feature_engineering.py` | Build v1 features |
| `preprocessing/26_add_skewkurt.py` | Build v2 features |
| `lgbm/08_optuna_tune.py` | Optuna hyperparameter search |
| `lgbm/32_skewkurt_tuned_h3.py` | Best LGBM for h=3 (CV=0.1211) |
| `nn/39_ae_h3_multiseed.py` | NN baseline with full indicators |
| `submissions/41_blend_h3.py` | NN + LGBM blend for h=3 |

---

## Results Log

See [work_log.md](work_log.md) for full experiment history.
