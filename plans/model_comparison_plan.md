# Model Comparison Plan

## Goal
Compare the current forecasting models in a fair and reproducible way, even though they come from different model families. The comparison should answer two questions:

1. Which model predicts Gujarat hourly demand more accurately?
2. Which model is more practical to use in production?

This plan is designed for comparing models such as XGBoost and neural-network-based models like LSTM/RNN.

## Core Principle
Do not compare raw parameter count across different model families as the main fairness criterion.

Parameter count is useful only within the same family, for example:
- LSTM vs LSTM
- XGBoost vs XGBoost

For cross-family comparison, fairness should come from using the same data, same target, same horizon, same split, same preprocessing rules, and the same evaluation metrics.

## What Must Be Held Constant
To make the comparison valid, all models should use:
- The same target variable: `demand_mw`
- The same chronological train/validation/test split
- The same forecast horizon
- The same input availability rule
- The same leakage protection rules
- The same evaluation window
- The same tuning budget as much as possible

If a model needs a different input format, convert the data to that format without changing the underlying information content.

## Models to Compare
The comparison should include at least:
- XGBoost pipeline
- LSTM or RNN model
- Any baseline persistence model already used in the notebook

Optional additions if time allows:
- Random Forest
- LightGBM / CatBoost
- Simple linear regression baseline
- STL or seasonal naive baseline

## Comparison Strategy
The comparison should happen in three layers:

### 1. Forecast accuracy
This is the most important layer.

### 2. Robustness and stability
This checks whether a model is reliable across different time periods.

### 3. Operational cost
This checks whether the model is practical to train, tune, and deploy.

## Meaningful Ways to Compare the Models

### A. Overall Error Metrics
Use these on the same test set:
- MAE
- RMSE
- MAPE
- sMAPE
- R2
- nRMSE

Why these matter:
- MAE shows average absolute miss in MW.
- RMSE penalizes large errors more strongly.
- MAPE and sMAPE show percentage-based error.
- R2 shows explained variance.
- nRMSE makes scale-normalized comparison easier.

### B. Peak-Demand Performance
This is especially important for demand forecasting.

Compare:
- Peak MAPE
- Peak MAE
- Error on top 10% of actual demand hours
- Error on top 5% of actual demand hours

Why this matters:
- A model may perform well on average but fail badly during demand peaks.
- For grid operations, peak-hour quality can matter more than global average quality.

### C. Time-Window Performance
Compare metrics separately for:
- Train period
- Validation period
- Test period
- Each year separately if needed
- Each season separately if needed

Useful slices:
- 2021
- 2022
- 2023
- 2024
- 2025
- Summer vs monsoon vs winter
- Weekday vs weekend
- Daytime vs nighttime

Why this matters:
- Models can behave differently across regimes.
- A model that is stable through 2024 and 2025 is more trustworthy than one that only fits one regime.

### D. Rolling Backtest Performance
Use rolling or expanding-window backtests instead of only one split.

Compare:
- Average metric across folds
- Worst-fold metric
- Standard deviation across folds
- Fold-by-fold peak error

Why this matters:
- A single test split can be misleading.
- Rolling backtests measure temporal robustness.

### E. Residual Behavior
Compare residuals, not just point scores.

Check:
- Residual mean
- Residual standard deviation
- Residual histograms
- Residual autocorrelation
- Residuals by hour of day
- Residuals by season

Why this matters:
- A model with smaller average error may still have structured bias.
- Residual autocorrelation suggests the model is missing predictable patterns.

### F. Calibration and Bias
Compare whether predictions are systematically high or low.

Check:
- Mean signed error
- Bias during peak periods
- Bias during low-demand periods
- Underprediction rate on peaks
- Overprediction rate on low-demand hours

Why this matters:
- Some models are conservative, others are aggressive.
- Bias can be more important than raw error in operations.

### G. Ranking Agreement
Compare whether the models make similar decisions about relative demand levels.

Possible checks:
- Correlation between predicted and actual values
- Correlation of hourly ranks
- Agreement on peak-hour ordering

Why this matters:
- Useful when the operational goal is prioritization, not just exact MW prediction.

### H. Statistical Significance
Do not rely only on visual differences.

Use paired tests on the same timestamps:
- Diebold-Mariano test for forecast comparison
- Paired t-test on absolute errors if assumptions are acceptable
- Wilcoxon signed-rank test for nonparametric comparison
- Bootstrap confidence intervals for MAE, RMSE, and MAPE

Why this matters:
- Two models can have close scores but one may not be statistically better.

### I. Training and Inference Cost
Compare practical deployment cost:
- Training time
- Hyperparameter tuning time
- Inference time per hour or per batch
- Memory usage
- Model file size
- Retraining frequency

Why this matters:
- A slightly better model may not be worth a large runtime or maintenance penalty.

### J. Data Efficiency
Compare how well each model behaves when training data is reduced.

Test on:
- Full training set
- 75% of training set
- 50% of training set
- Recent-only training window

Why this matters:
- Some models need a lot of data.
- Others are more sample-efficient.

### K. Robustness to Regime Change
Compare performance on unusual periods such as:
- 2024 if it is a different regime
- Extreme heat days
- Unexpected demand spikes
- Missing-weather-feature simulations

Why this matters:
- Real demand forecasting often fails during unusual conditions.

## Fair Comparison Protocol
Use this protocol for every model:

1. Fit the model only on the training set.
2. Tune hyperparameters only on the validation set or rolling CV folds.
3. Freeze the model configuration.
4. Evaluate once on the test set.
5. Record metrics on all required slices.
6. Repeat if using multiple seeds or CV folds.

Do not tune based on the test set.

## Reporting Template
For each model, create a summary table with:
- Model name
- Train MAE
- Validation MAE
- Test MAE
- Train RMSE
- Validation RMSE
- Test RMSE
- Train MAPE
- Validation MAPE
- Test MAPE
- R2 on test
- Peak MAPE on test
- Training time
- Inference time
- Notes on bias or instability

## Recommended Decision Rule
Choose the best model by following this order:

1. Peak-demand performance on the test set
2. Overall test MAE and RMSE
3. Stability across folds and time windows
4. Bias and residual quality
5. Training/inference cost

If two models are close, prefer the simpler or faster one.

## Suggested Visual Comparisons
Add these plots for each model:
- Actual vs predicted line plot on test set
- Residual distribution
- Residuals over time
- Error by hour of day
- Error by month or season
- Predicted vs actual scatter plot
- Fold-wise backtest score plot

## What Not to Use as the Main Comparison
Avoid using these as primary fairness rules across families:
- Equal parameter count
- Equal number of layers
- Equal number of trees only
- Training loss alone
- One split only

These can be secondary diagnostics, but they should not decide the winner.

## Final Deliverable
The final notebook should include:
- One unified comparison table
- One plot set per model or a shared comparison plot
- A short written conclusion explaining which model is best and why
- A note on whether the winner is best for accuracy, peak handling, or production cost

## Conclusion
For different model families, compare the models by controlling the data and evaluation protocol, then judge them with forecasting metrics, peak performance, robustness, and deployment cost. That is the fairest and most meaningful comparison for this project.
