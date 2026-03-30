# Competition Research Report: High-Ranking Solutions for Financial Time Series Forecasting

**Research Date:** 2026-03-29
**Our Competition:** Hedge Fund - Time Series Forecasting (`ts-forecasting`)
**Our Current Rank:** 326/1035 | CV: 0.2403
**Sources Reviewed:** 25+ web sources, 8+ GitHub repositories, multiple discussion forums

---

## 1. Ubiquant Market Prediction (Kaggle 2022)

### Similarity to our project: 9/10
Most similar competition found. 3,000+ investment IDs, `time_id`, 300 **anonymous** features, predict return-like `target`. Metric = mean Pearson correlation per time_id. Even the feature structure (f_0 to f_299) mirrors ours (feature_a to feature_ch).

### Resources
- [1st Place Solution "Our Betting Strategy"](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy)
- [2nd Place Solution "Robust CV and LGBM"](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/davide-stenner-2nd-place-solution-robust-cv-and-lg)
- [GitHub: Top 1% Solution (20th/2893) - pinouche](https://github.com/pinouche/ubiquant-kaggle-competition) — LightGBM
- [GitHub: DNN Ensemble Solution (top 2%, 41st/2893)](https://github.com/user-jiyichen/Ubiquant-Market-Prediction-Competition)
- [Ubiquant Feature Exploration (marketneutral)](https://www.kaggle.com/code/marketneutral/ubiquant-feature-exploration)
- [Ubiquant Target EDA PCA Magic](https://www.kaggle.com/code/marketneutral/ubiquant-target-eda-pca-magic)

### Key Techniques

**Feature Engineering:**
- The 300 anonymous features were used mostly as-is by LGBM-based solutions
- **Investment ID embedding**: Top DNN solutions used a learned 32-dimensional embedding for `investment_id` (categorical lookup layer), concatenated with the 300 numerical features
- Cross-sectional features (grouping by `time_id`): competitors computed per-time_id statistics to create market-neutral features
- PCA exploration of the target structure revealed hidden patterns — EDA suggested features had low mutual correlation, so aggressive dimensionality reduction was counterproductive
- Standard preprocessing: float16 for memory, standardization of 300 features

**Model:**
- **1st place**: LightGBM + TabNet ensemble with heavy feature engineering (betting strategy based on cross-sectional ranking of predictions)
- **2nd place**: Robust CV + LightGBM, focusing on time-series-safe CV over feature complexity
- **Top 2% DNN ensemble**: 25+ model ensemble combining:
  - Base DNN: `investment_id` embedding (32d) + dense(256→256→256) → dense(512→128→32→1)
  - Dropout-regularized variants (rates 0.1–0.75)
  - Conv1D models: reshape 300 features to 1D, apply Conv1D filters (16→16→64→64), then dense
  - Conv2D spatial models: reshape to 64×64×1, apply 2D convolutions
  - **Correlation loss** variants: optimize Pearson correlation directly instead of MSE
  - Gaussian noise injection (variance=0.035) for augmentation
- Metric was mean Pearson correlation per time_id — note our metric is weighted MSE-based, but cross-sectional ranking is still relevant

**Weights Handling:**
- No explicit sample weights in this competition (uniform weighting)
- Evaluation was per-time_id correlation mean, so the "betting strategy" was about allocating predictions cross-sectionally per time period

**Target Engineering:**
- Target used as-is; PCA exploration suggested the target had latent structure (sub-components)

**Ensembling:**
- Mean averaging across 25+ DNN variants
- LGBM + deep learning ensemble

### What We Should Steal
1. **Investment/sub_code ID embedding** as an additional feature (32-dim learned embedding)
2. **Correlation loss function** — optimize Pearson correlation directly instead of MSE when possible
3. **Conv1D over feature vector** — treat the feature vector as a 1D signal, apply Conv1D
4. **Large ensembles** — 25+ model variants with mean averaging
5. **Per time_id cross-sectional post-processing** of predictions (rank-normalize within each time step)

---

## 2. Jane Street Market Prediction (Kaggle 2021)

### Similarity to our project: 8/10
Very similar structure: anonymized features (130+ continuous), `weight` column (highly skewed), multiple response targets (`resp`, `resp_1`, `resp_2`, `resp_3`, `resp_4`), `date_id` for time. Metric used weighted utility function. Competition ran live on future data.

### Resources
- [1st Place Solution — Yirun's Supervised Autoencoder + MLP](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
- [Jane Street Supervised Autoencoder MLP (gogo827jz)](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp)
- [GitHub: scaomath/kaggle-jane-street](https://github.com/scaomath/kaggle-jane-street)
- [GitHub: Leo1998-Lu Silver Medal (173rd/4245)](https://github.com/Leo1998-Lu/Kaggle-Jane-Street-Market-Prediction-Silver-Medal-solution)
- [Numerai Forum: AutoEncoder + multitask MLP discussion](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338)

### Key Techniques

**Feature Engineering:**
- **Missing value imputation**: Fill with past-day mean computed across all rows (including weight=0 rows) for each feature
- **Filtering**: Drop first 85 days of training data — they exhibited high noise inconsistent with later data patterns
- **Categorical features disguised as continuous**: Features 3, 4, 6, 19, 20, 22, 38, 71, 85, 87, 92, 97, 105, 116, 127, 129 had extremely high frequency of common values → treated with categorical embeddings
- **Feature interactions** (from codefluence solution): Multiply feature pairs together; use TabNet to identify top-N most significant interaction pairs; add these pairwise products as new features (6 pairs selected)
- **Market regime features**: Use `feature_64` (gradient approximation) as a volatility indicator; previous-day trade count as market activity classifier
- **Z-scores**: Per-opportunity Z-scores compared to means of recent opportunities — provides regime context for networks
- **Rejected** (tested but did not help): EWM, rolling means, lag features, denoised targets via covariance eigenvalue removal, feature neutralization (too slow at inference)

**Model Architecture (1st Place — Supervised Autoencoder + MLP):**
- **Architecture**: Autoencoder encoder + small MLP with skip connection in first layer
- Inspired by "Deep Bottleneck Classifiers in Supervised Dimension Reduction" paper
- The autoencoder learns a compressed representation; MLP predicts on the bottleneck features + skip connection
- Multiple response columns used as targets simultaneously (`resp`, `resp_3`, `resp_4` for volatile models; all 5 for smoother models)
- **Training**: RAdam/Adam with cosine annealing scheduler (warm restarts)
- **Fine-tuning**: Every 10 epochs, apply utility function regularizer (optimize `sigmoid(output)` for utility maximization)
- **Denoised targets**: 1–2 additional denoised target columns added during training

**Model Architecture (scaomath team — 3rd place range):**
- PyTorch skip-connection model (~400k params)
- TF Autoencoder + MLP (~300k params, best CV)
- Residual MLP with high-dropout filtering layer
- Categorical embedding model for disguised-categorical features (~300k params, best single LB score)

**Weights Handling (CRITICAL for our project):**
- Trained ONLY on `weight > 0` rows (zero-weight rows excluded)
- No weighted loss functions in final pipeline (tested `ln(1+weight)` weighting — did not help)
- Heavy trades with high weights are crucial to get right — they can outweigh thousands of small trades
- **Alternative tested**: Assign weight=1e-7 to 40% of zero-weight rows — did NOT improve
- The key insight: the metric is dominated by high-weight samples, so model must implicitly learn to predict those well

**Multi-Target Learning:**
- Trained on `resp`, `resp_1`, `resp_2`, `resp_3`, `resp_4` simultaneously (multi-task)
- Different model variants used different subsets: "volatile" models used `resp`, `resp_3`, `resp_4`; "smooth" models used all 5
- Using multiple horizon targets as auxiliary tasks regularizes the model and improves generalization

**Validation:**
- Grouped 3-fold CV with **10-day gap** (embargo) between train and validation folds
- Separate treatment of "volatile" vs. "regular" trading days using `feature_64` as regime indicator
- Utility function fine-tuning provided ~70% CV-LB correlation (vs. near-zero without it)

**Ensembling:**
- Final: `3(spike-aware) + 3(PyTorch baseline) + 3(autoencoder) + 1(TF residual)` = 10 models
- Blending via concatenation + middle 60% averaging (trims extreme predictions)
- Busy trading days used 50% trimming
- 3-seed ensembles for each architecture

### Code Snippets

**Supervised Autoencoder concept:**
```python
# Encoder learns compressed representation
encoder_output = encoder(input_features)
# MLP uses bottleneck + skip connection from input
mlp_input = concatenate([encoder_output, input_features])
output = mlp_layers(mlp_input)
# Train on multiple response columns simultaneously
loss = weighted_mse(outputs, [resp, resp_1, resp_2, resp_3, resp_4])
```

**Middle-K ensemble blending:**
```python
# Middle 60% averaging (trims extreme predictions)
preds_sorted = np.sort(all_model_preds, axis=0)
n = len(all_model_preds)
trim_lo, trim_hi = int(n * 0.2), int(n * 0.8)
final_pred = preds_sorted[trim_lo:trim_hi].mean(axis=0)
```

### What We Should Steal
1. **Multi-task training** across all 4 horizons simultaneously (our 1/3/10/25-day targets as multi-output)
2. **Supervised autoencoder**: encoder learns feature compression, MLP predicts on bottleneck + skip
3. **Filter low-weight rows** from training (train only on weight > threshold)
4. **10-day embargo gap** in cross-validation to prevent leakage
5. **Market regime features**: compute per-date mean/std as "market context" features
6. **Feature interaction pairs**: multiply top feature pairs selected by TabNet or SHAP importance
7. **Middle-K ensemble trimming** (trim 20% extremes from ensemble predictions)
8. **Cosine annealing with warm restarts** for neural network training

---

## 3. Jane Street Real-Time Market Data Forecasting (Kaggle 2024)

### Similarity to our project: 8/10
Upgraded version of 2021 competition. 130 anonymized continuous features + 1 binary, `weight` column, `responder_6` as primary target, **lagged responders** available for online learning. Live evaluation.

### Resources
- [Competition Page](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- [GitHub: evgeniavolkova solution](https://github.com/evgeniavolkova/kagglejanestreet)
- [Jane Street 2024 LSTM with auxiliary features](https://www.kaggle.com/code/ccyhui/jane-street-2024-lstm-with-auxiliary-features)
- [Medium: Transformer approach](https://medium.com/@jsemrau/forecasting-real-time-market-data-with-transformer-fc8f96bd6b8e)

### Key Techniques

**New Features vs. 2021:**
- Bigger dataset, more sophisticated features
- Multiple auxiliary responders (`responder_0` through `responder_8`) available as inputs
- **Lagged responder** provided explicitly for online learning experiments
- 130 continuous features + 1 binary feature

**Online Learning (key differentiator):**
- The competition explicitly provided lagged responders to enable test-time adaptation
- Models could use recently-observed responses to update predictions
- Approaches explored: updating model weights at test time, using lagged responses as additional input features

**Feature Engineering:**
- Z-scores of original features vs. recent opportunity means (provides market regime context)
- Market direction and volatility estimates for the current day as additional inputs
- Feature interactions: pairwise products of top-N features (selected by TabNet or SHAP)
- LSTM approach: treating sequences of opportunities per symbol as time series with auxiliary responders as inputs

**Models Used by Competitors:**
- MLP with skip connections (baseline adapted from 2021 solution)
- LGBM with lagged responders as features
- LSTM with auxiliary responder features
- Transformer architectures (Autoformer, Informer, TimeXer — explored but less dominant)
- AE+MLP: autoencoder for feature compression + MLP prediction (adapted from 2021 winner)

**System Requirements for Top Solutions:**
- ~100GB RAM, 12GB GPU RAM (indicates very large ensembles or complex models)
- Separate CV scripts, full training scripts, ensemble scripts

### What We Should Steal
1. **Lagged target as input**: Use previously observed `y_target` values for past horizons as new features
2. **Auxiliary targets as shared learning signal**: Train jointly on all 4 horizons
3. **Online learning / model recency weighting**: Weight more recent training data more heavily

---

## 4. Numerai Tournament (Ongoing)

### Similarity to our project: 7/10
Hedge fund-style competition. Anonymized features, financial returns target (20-day ahead), multiple targets, era-based structure. Key differentiator: **Numerai has explicit feature group metadata** (intelligence, wisdom, charisma, dexterity, strength, constitution, agility, serenity) — similar to our subcategory structure.

### Resources
- [Numerai Docs](https://docs.numer.ai/)
- [Feature Neutralization Notebook](https://github.com/numerai/example-scripts/blob/master/feature_neutralization.ipynb)
- [Target Ensemble Notebook](https://github.com/numerai/example-scripts/blob/master/target_ensemble.ipynb)
- [Era Boosting Forum Thread](https://forum.numer.ai/t/era-boosted-models/189)
- [Comprehensive Guide to Numerai](https://tit-btcqash.medium.com/a-comprehensive-guide-to-competing-at-numerai-70b356edbe07)
- [Feature Neutralization Forum](https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955)
- [NumerBlox Pipeline Library](https://github.com/crowdcent/numerblox)

### Key Techniques

**Feature Engineering:**
- Feature group statistics: compute `mean`, `std`, `skewness` across each of the 8 feature groups per row → creates 24 meta-features capturing group-level information
- This is directly analogous to computing statistics across our sub-categories or feature families
- Low inter-feature correlation → PCA hurts (destroys signal); keep all features
- Feature expansion by increasing count while reducing correlation between features

**Feature Neutralization:**
- Neutralize predictions against features to reduce "feature exposure" (over-reliance on a few features)
- Uses Moore-Penrose pseudoinverse: `predictions_neutral = predictions - exposures @ pinv(exposures) @ predictions`
- **Proportion parameter**: 0.0 = no neutralization, 1.0 = fully orthogonal to features; optimal ~0.3–0.5
- Improves Sharpe ratio (reduces variance across eras) at the cost of lower mean correlation
- Direct application to our problem: neutralize predictions against the strongest individual features

**Era Boosting Algorithm:**
```python
def era_boost_train(X, y, era_col, proportion=0.5,
                    trees_per_step=10, num_iters=200):
    """
    Iteratively adds trees trained on worst-performing time periods.
    Dramatically reduces per-era variance (Sharpe from 2.28 → 21.99 in-sample).
    """
    model = XGBRegressor(max_depth=5, learning_rate=0.01,
                         n_estimators=trees_per_step,
                         colsample_bytree=0.1)
    model.fit(X, y)

    for i in range(num_iters - 1):
        preds = model.predict(X)

        # Score each era (time period)
        era_scores = {}
        for era in era_col.unique():
            mask = era_col == era
            era_scores[era] = spearmanr(preds[mask], y[mask]).correlation

        # Select worst-performing eras (bottom 50%)
        worst_eras = [e for e, s in era_scores.items()
                      if s <= np.quantile(list(era_scores.values()), proportion)]
        worst_mask = era_col.isin(worst_eras)

        # Add trees trained only on worst eras
        model.n_estimators += trees_per_step
        model.fit(X[worst_mask], y[worst_mask],
                  xgb_model=model.get_booster())

    return model
```
- In our context: era = `date_id` (or time period buckets); worst eras = dates where predictions are most wrong
- **Caveat**: Strong in-sample Sharpe improvement may not fully generalize OOS

**Target Ensemble:**
- Train separate models for each target (e.g., Numerai trains on `target_nomi_v4_20`, `target_cyrusd_20`, etc.)
- Average predictions across models trained on different targets
- Produces more robust, less overfit predictions

**Model Recommendations:**
- LGBM hyperparameters: `n_estimators=20000, lr=0.001, max_depth=6, num_leaves=64, colsample_bytree=0.1`
- Deep parameters: `n_estimators=30000, max_depth=10, num_leaves=1024, min_child_samples=10000`
- Walk-forward CV: train on eras 1..N, predict era N+1..N+8, leave 8-era gap (embargo)

**Ensembling Canonical Approach:**
1. Rank-normalize predictions per era (period)
2. Gaussianize (apply normal quantile transform)
3. Standardize to unit variance
4. Weighted dot product across models
5. Final Gaussianization + optional neutralization

**CatBoost Superiority:**
- CatBoost (vanilla, no tuning): Sharpe 0.8704 vs. LightGBM: 0.7188 — CatBoost may generalize better on financial data
- Stacked ensemble (CatBoost + XGBoost) + feature neutralization: Sharpe 0.9314

### What We Should Steal
1. **Era boosting**: iteratively train on worst-performing dates — reduces variance, improves Sharpe
2. **Feature neutralization**: remove linear correlation with individual features from predictions
3. **Feature group statistics**: compute mean/std/skew per sub-category or feature family per row
4. **Target ensemble**: train separate models per horizon, average — may outperform single multi-output model
5. **Rank-normalize predictions per time period** before ensembling
6. **Try CatBoost** as alternative to LightGBM — may generalize better

---

## 5. Optiver Realized Volatility Prediction (Kaggle 2021)

### Similarity to our project: 6/10
Financial time series, per-instrument models, temporal features. Key difference: raw order book data available (bid/ask). But cross-instrument features and SuperLearner ensemble are directly applicable.

### Resources
- [1st Place Solution Discussion](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- [GitHub: 7th Place Solution - michaelpoluektov](https://github.com/michaelpoluektov/orvp)
- [Medium: Issam Sebri's Approach](https://koeusiss.medium.com/optiver-realized-volatility-prediction-cb7da76fbd3f)
- [EDA Notebook](https://www.kaggle.com/code/gunesevitan/optiver-realized-volatility-prediction-eda)

### Key Techniques

**Feature Engineering (Two-Level Approach):**
- **Level 1 (Cross-sectional)**: Mathematical relationships between existing variables at each time point (e.g., spread = ask_price - bid_price, bid-ask ratio)
- **Level 2 (Temporal)**: Windowed aggregation — compute mean, variance, log returns over different time windows to create lag features

**Cross-Instrument Features:**
- Compute aggregate statistics (mean volatility, correlation with market average) across ALL instruments at each time_id
- The realized volatility of instrument X is correlated with the average of all instruments — capturing this "market factor" dramatically improved scores
- This is analogous to our cross-sectional z-score features but at the prediction level too

**SuperLearner Ensemble (Stacking):**
- 8 base models: Linear, ElasticNet, Decision Tree, MLP, LightGBM, Bagging, RandomForest, ExtraTrees
- Out-of-fold predictions as meta-features
- Meta-learner (linear regression) combines base learner outputs
- Feature sampling variation: replicate models with randomly selected 60% feature subsets → expands library to ~40 models

**1st Place (Nearest Neighbors approach):**
- Reconstructed chronological ordering of obfuscated time IDs using stock price distances
- Build directed graph with time_IDs as nodes, L2 distance between price vectors as edge weights
- Approximate Hamiltonian path via KNN → revealed temporal structure
- This "data archaeology" recovered information not normally available

**7th Place BorutaSHAP feature selection:**
- Used BorutaSHAP to select important features from large engineered feature set
- LightGBM as primary model

### What We Should Steal
1. **Cross-instrument aggregate features**: for each date, compute mean/std of each feature across all instruments → captures market-wide factors
2. **SuperLearner stacking**: use out-of-fold predictions from diverse models as meta-features
3. **Feature sampling diversity in ensemble**: each model sees 60% random feature subset
4. **Two-level feature engineering**: (a) within-row mathematical transforms, (b) temporal aggregation windows

---

## 6. Two Sigma Financial Modeling / News Prediction (Kaggle 2019)

### Similarity to our project: 6/10
Anonymous features, return prediction, time series structure.

### Resources
- [Two Sigma Financial News Competition](https://www.kaggle.com/c/two-sigma-financial-news)
- [What we learned from the competition (EP Chan)](https://epchan.com/img/links/What-we-learned-from-Kaggle-Two-Sigma-News-Sentiment-competition.pdf)

### Key Techniques
- Extreme value clipping: clip features at 2nd/98th percentiles
- Remove highly correlated features (e.g., wordCount and sentenceCount had near-perfect correlation — drop one)
- LightGBM with ~60% accuracy on direction; XGBoost comparison showed LGBM slightly better
- 5-day moving average of sentiment × relevance as a naive baseline
- Data quality >> model sophistication: finding useful features mattered more than tuning algorithms

### What We Should Steal
1. **Aggressive feature clipping** at 2nd/98th percentiles for all continuous features
2. **Correlation-based deduplication**: remove features with correlation > 0.95 with another feature

---

## 7. Numerai-Inspired Techniques: Era Splitting / Invariant Learning

### Similarity to our project: 7/10
Addresses the core challenge of financial models that overfit to specific time periods.

### Resources
- [Era Splitting: Invariant Learning for Decision Trees](https://arxiv.org/html/2309.14496)

### Key Techniques

**Era Splitting:**
- Alternative to era boosting; modifies tree-splitting criterion to find splits that are consistently good across multiple eras
- Rather than minimizing average loss, minimize maximum per-era loss (minimax)
- Forces trees to find patterns that work across different market regimes

### What We Should Steal
1. **Era-invariant evaluation**: track per-date (or per-week) metric variance, not just mean CV
2. **Worst-period monitoring**: regularly check which dates model fails most on

---

## 8. General Kaggle Competition Insights: Sample Weights and Skewed Distributions

### Similarity to our project: 8/10 (directly applicable)

### Key Techniques

**Handling Skewed Sample Weights:**
- LightGBM weight parameter: gradient and Hessian are multiplied by sample weights
- Custom objective: `objective(y_true, y_pred, weight) -> grad, hess` — manually multiply grad/hess by weights
- Experiment with `ln(1 + weight)` transformation to soften extreme weights
- Alternative: train without weights, then evaluate on weighted metric (sometimes less overfitting to outliers)
- **Critical insight from Jane Street**: zero-weight rows still contain signal for feature learning — don't drop them from features, but weight them appropriately in loss

**Target Winsorization:**
- Clip target at 5th/95th percentile before training → reduces pull of extreme returns
- Improves training stability without losing most of the data
- Apply inverse transform at inference time (if needed)

**Time-Series Cross-Validation:**
- **Purged k-fold** (López de Prado): remove training samples whose labels overlap in time with the test fold
- **Embargo**: after each test fold, skip the next N periods (e.g., 10 days) in training
- Prevents look-ahead bias through overlapping rolling features (EWM, rolling mean)

---

## 9. Purged Cross-Validation (Financial ML Best Practice)

### Similarity to our project: 10/10 (must implement)

### Key Techniques

**Purging:**
- Remove training samples whose label windows overlap with the test fold's label window
- Relevant for us: horizon-25 predictions computed at time T use data from T to T+25 → training samples too close to test fold boundaries must be dropped

**Embargo:**
- After each test fold, skip a buffer period (typically 5-10% of fold size)
- For us: with a 25-day horizon, embargo at least 25 trading days after each fold boundary

**Walk-Forward CV:**
- Train on periods 1..N, validate on N+1..N+k, move forward
- Repeat with expanding or rolling window
- Captures temporal model degradation

**Combinatorial Purged CV (CPCV):**
- Multiple train-test splits using combinations of fold groups
- Provides distribution of OOS performance estimates rather than single estimate
- More statistically robust than single walk-forward split

---

## Summary: Top Techniques to Try (Ranked by Expected Impact)

### Tier 1: High Impact, Relatively Easy to Implement

**1. Multi-task training across all 4 horizons**
- Train a single model outputting all 4 horizon predictions simultaneously
- Use weighted multi-target loss: `loss = sum(horizon_weight_i * MSE_i)`
- Regularizes via shared representation; typically +5–15% improvement over per-horizon models
- Source: Jane Street 2021 (multi-resp training), general competition wisdom

**2. Per-date cross-sectional rank normalization of predictions**
- After generating raw predictions, rank-normalize within each `date_id` × `horizon`
- Formula: `pred_norm = rankdata(pred) / len(pred)` per date group
- Removes systematic date-level biases, aligns with skill score which measures relative ranking
- Source: Ubiquant (metric = Pearson correlation per time_id), Numerai canonical ensemble

**3. Embargo/purged cross-validation (10-25 day gap)**
- Current CV may be leaking via overlapping rolling features
- Add 25-day gap (horizon-25 is longest) between train and validation
- Will likely lower raw CV score but make it more reliable
- Source: Jane Street 2021 (10-day gap), López de Prado financial ML

**4. Era boosting (iterative worst-period training)**
- After training base model, iteratively train 10 trees on worst-performing dates
- 20 iterations typically sufficient; reduces per-date variance dramatically
- Can be applied on top of existing LightGBM pipeline without major refactoring
- Source: Numerai Forum, empirical Sharpe improvement from 2.28 → 21.99 in-sample
```python
# Pseudocode
for iter in range(20):
    per_date_scores = compute_metric_by_date(model, X_train, y_train)
    worst_dates = per_date_scores.nsmallest(50%)
    model.fit(X_train[worst_dates], y_train[worst_dates],
              init_model=model)  # warm start
```

**5. Sub_code / instrument embedding as a feature**
- Map each instrument (180 values) to a learned 16–32 dimensional embedding
- Captures instrument-specific behavior (analogous to investment_id embedding in Ubiquant)
- Easiest to add via LightGBM by keeping `sub_code` as a categorical feature
- Or train a neural layer to learn embeddings jointly
- Source: Ubiquant top DNN solution (32-dim embedding)

### Tier 2: Moderate Impact, Moderate Effort

**6. Feature neutralization (reduce feature exposure)**
- Compute predictions, then regress out linear contribution of individual features
- `pred_neutral = pred - features @ pinv(features) @ pred * proportion`
- Proportion ~0.3–0.5 typically optimal
- Reduces over-reliance on a few features → better generalization across market regimes
- Source: Numerai (Moore-Penrose neutralization), NumerBlox library implementation

**7. Supervised autoencoder for feature compression**
- Train autoencoder on features, use bottleneck representations as additional features
- Or: AE + MLP with skip connection (1st place Jane Street 2021)
- Learns a denoised, compressed feature representation
- Particularly effective for anonymous features where domain knowledge is absent
```python
# Architecture concept
encoder = Dense(128, activation='relu')(input)  # Compress
bottleneck = Dense(32)(encoder)  # Bottleneck
decoder = Dense(128)(bottleneck)  # Reconstruct
output = Dense(1)(concatenate([bottleneck, input]))  # Skip connection
# Loss = reconstruction_loss + prediction_loss
```

**8. Feature group statistics (per sub-category aggregation)**
- For each row, compute mean/std/skew across all features within each sub-category group
- Our data has 5 sub_categories × multiple features → creates ~15 new "meta-features"
- Captures how this instrument's features relate to the group average
- Source: Numerai (intelligence/wisdom/charisma group stats)
```python
# Example implementation
for group in feature_groups:
    group_cols = [f for f in features if feature_to_group[f] == group]
    df[f'{group}_mean'] = df[group_cols].mean(axis=1)
    df[f'{group}_std'] = df[group_cols].std(axis=1)
    df[f'{group}_skew'] = df[group_cols].skew(axis=1)
```

**9. Market-wide features (cross-instrument aggregation)**
- For each `date_id`, compute mean/std of each feature across ALL instruments
- Add as new features: `feature_X_market_mean`, `feature_X_market_std`
- Also: instrument's deviation from market: `feature_X - market_mean_feature_X` (already partially done with z-score)
- Captures market factor; residual is idiosyncratic signal
- Source: Optiver (cross-instrument volatility was most predictive feature)

**10. Feature interaction pairs**
- Use SHAP importance to identify top-20 features
- Create pairwise products: `feature_i * feature_j` for top pairs
- Or use TabNet to discover which pairs improve predictions most
- Source: Jane Street 2021 (6 interaction pairs added); Jane Street 2024 solutions

**11. Pairwise feature ratios and spreads**
- For related features: `f_i / (f_j + epsilon)`, `f_i - f_j`
- Especially useful if features within a sub-category have known relationships
- Source: Optiver (bid-ask spread = key derived feature), general Kaggle practice

### Tier 3: Higher Effort, Potentially High Impact

**12. Multi-task neural network (replace per-horizon LightGBM)**
- Single MLP/residual network with 4 output heads (one per horizon)
- Shared lower layers capture common patterns; horizon-specific heads capture horizon dynamics
- Add skip connections from input to each output head
- Train with weighted horizon losses (match competition's w_i weights)
- Source: Jane Street 2021 (multi-resp MLP), Ubiquant DNN ensemble

**13. Stacking / SuperLearner**
- Train diverse base models (LightGBM, XGBoost, CatBoost, MLP, ExtraTrees)
- Use OOF predictions as meta-features for a linear meta-learner
- Feature sampling: each base model sees random 60% of features (increases diversity)
- Source: Optiver SuperLearner (8 base models)

**14. Target winsorization**
- Clip `y_target` at 5th/95th percentile before training
- Reduces gradient variance from extreme return observations
- Especially relevant given our highly skewed weight distribution (extreme returns likely have extreme weights)
- Source: general financial ML best practice

**15. Online learning / test-time adaptation**
- Use recent test predictions and (after-the-fact) available labels to update model
- In competition context: use the last N observations to re-fit or fine-tune model
- Source: Jane Street 2024 (lagged responder provided explicitly for this purpose)

**16. CatBoost as LightGBM replacement/ensemble member**
- CatBoost showed Sharpe 0.8704 vs. LightGBM 0.7188 on Numerai (20% better)
- Handles categorical features natively (sub_code, sub_category, horizon as categoricals)
- Training is slower but may generalize better out-of-sample
- Source: Numerai empirical comparison

**17. Convolution over feature vector**
- Treat the 82 features as a 1D "signal", apply 1D convolutions
- Conv1D(16 filters, kernel_size=3) → Conv1D(64 filters, kernel_size=3) → Dense
- Captures local feature relationships (neighboring features may be correlated)
- Source: Ubiquant top DNN (Conv1D/Conv2D hybrid models)

### Tier 4: Research-Grade, Hard to Implement Cleanly

**18. Purged combinatorial cross-validation**
- Full CPCV implementation with multiple scenarios
- Provides distribution of OOS estimates → detect if CV is optimistically biased
- Source: López de Prado (2019), quantinsti blog

**19. Feature neutralization via linear regression**
- Regress predictions on a chosen factor (e.g., sub_category average return)
- Residual = prediction neutral to that factor
- Improves cross-sectional alpha isolation
- Source: Numerai, quantitative finance best practice

**20. Era-invariant / minimax tree splitting**
- Modify LGBM/XGBoost split criterion: use worst-era improvement instead of mean improvement
- Forces model to find universally useful splits
- Source: arxiv.org/html/2309.14496 (Era Splitting paper)

---

## Implementation Priority Queue for Our Project

Given our current state (CV=0.2403, rank 326/1035, LightGBM per-horizon, 3-seed ensemble, 328 features):

### Quick Wins (next 1-2 experiments)
1. **Multi-task LightGBM**: combine all 4 horizons in one model with horizon as a feature → tests multi-task signal sharing
2. **Per-date rank normalization of predictions**: post-processing step, no retraining
3. **Add market-wide aggregate features**: group by date_id, compute mean/std per feature across all sub_codes

### Medium-term (next 3-5 experiments)
4. **Embargo in CV**: add 25-day gap → more reliable CV
5. **Era boosting**: implement iterative worst-date retraining
6. **Feature group statistics**: compute per-sub_category mean/std/skew per row
7. **Sub_code as LightGBM categorical** (if not already)
8. **Try CatBoost** on same feature set

### Longer-term (if above yield improvement)
9. **Supervised autoencoder + MLP** as neural alternative
10. **SuperLearner stacking** with diverse base models
11. **Feature neutralization** of final predictions

---

## Key Overarching Insights Across All Competitions

1. **Cross-sectional normalization dominates**: All financial competitions find that normalizing predictions within each time period (rank, z-score) is essential — the metric rewards relative, not absolute, prediction accuracy

2. **Anonymous features resist manual engineering**: When features are anonymized, standard domain-knowledge engineering fails. Successful approaches: (a) feature interaction products, (b) group statistics, (c) neural embeddings that learn structure automatically

3. **Weights require careful handling**: In Jane Street, training without weights but monitoring weighted metric worked better than weighted loss. The signal in zero-weight rows is still useful for the model to learn from.

4. **CV-LB correlation is the hardest problem**: Jane Street team noted "impossible" CV-LB correlation. The fix was utility-function-based regularization. For us: make CV closely mimic the competition's skill score formula.

5. **Multi-task learning almost always helps**: Training on multiple related targets (multiple horizons, multiple response types) consistently outperforms single-target models in all financial competitions reviewed.

6. **Temporal consistency matters more than raw performance**: The Numerai insight that reducing per-era variance (Sharpe) matters more than maximizing mean correlation is directly applicable. A model that is consistently good beats one that is occasionally great but often bad.

7. **Ensemble diversity > ensemble size**: A 4-model ensemble of architecturally diverse models (LGBM + CatBoost + MLP + autoencoder) typically beats 20 nearly-identical LightGBM seeds.

8. **Data leakage via temporal features is the #1 CV trap**: EWM and rolling features computed on the full dataset contaminate the training/validation split. Always compute temporal features within the training fold only, and embargo at least as long as the longest feature lookback window.

---

*Sources consulted: Kaggle competition pages and writeups for Ubiquant, Jane Street (2021 and 2024), Optiver Realized Volatility, Two Sigma Financial News; Numerai documentation, forum, and example scripts; multiple GitHub solution repositories; academic papers on purged CV and era splitting; Medium articles by competition participants.*
