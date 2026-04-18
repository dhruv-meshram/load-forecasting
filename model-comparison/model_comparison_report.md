# Gujarat Model Comparison Report (From-Scratch Implementation)

## 1. Objective
This report implements the Model Comparison Plan from scratch using only the saved deployed artifacts:
- XGBoost ONNX model: xgboost_model.onnx
- LSTM ONNX model: final_lstm_model.onnx
- LSTM scaler pack: final_lstm_scalers.pkl

Walk-forward reserved dates were explicitly excluded from this comparison run, as requested.

## 2. Data Scope and Protocol
- Source dataset: gujarat_hourly_merged.csv
- Global cleaned range in file: 2020-01-01 01:00:00 to 2025-06-30 23:00:00
- Comparison evaluation range used: 2020-01-01 00:00:00 to 2024-05-31 23:00:00
- Reserved range excluded: 2024-06-01 00:00:00 to 2025-06-30 23:59:59
- Rows before cleaning: 48191
- Rows after feature engineering + dropna constraints: 38375
- Final synchronized rows used for model-to-model comparison: 38374
- Chronological split (train/validation/test): 70%/15%/15%

## 3. Fairness Controls Enforced
- Same target variable for all models: demand_mw
- Same timestamp rows for scoring all models after alignment
- Same chronological split boundaries for all models
- No shuffling
- No use of reserved walk-forward period
- Same metrics and slicing rules across models

## 4. Model Input Construction
### 4.1 XGBoost
- ONNX input shape: (n_samples, 50)
- Feature set reconstructed from training pipeline (50 ordered features)
- Prediction: direct point forecast demand_mw

### 4.2 LSTM
- ONNX input shape: (batch_size, 168, 30)
- Sequence lookback: 168 hours
- Features scaled using provided x_scaler
- Model output inverse-transformed by y_scaler
- Output interpreted as next-step demand delta, then reconstructed as:
  pred_demand_(t+1) = demand_t + predicted_delta_(t+1)

### 4.3 Baseline
- Persistence baseline using lag_1h

## 5. Main Accuracy Tables
### 5.1 Test Set (primary decision layer)
| split   | model             |   n_samples |     mae |    rmse |   mape_pct |   smape_pct |       r2 |     nrmse |   peak_mape_pct_top10 |   peak_mae_top10 |   top5_mape_pct |   residual_mean |   residual_std |   mean_signed_error |    bias_pct |   peak_underprediction_rate_pct |   low_overprediction_rate_pct |   pearson_corr |   spearman_rank_corr |   peak_count_top10 |   peak_count_top5 |
|:--------|:------------------|------------:|--------:|--------:|-----------:|------------:|---------:|----------:|----------------------:|-----------------:|----------------:|----------------:|---------------:|--------------------:|------------:|--------------------------------:|------------------------------:|---------------:|---------------------:|-------------------:|------------------:|
| test    | XGBoost_ONNX      |        5757 | 311.349 | 428.848 |    1.79624 |     1.78887 | 0.971797 | 0.0270346 |               1.66557 |          364.338 |         2.01347 |      -66.3247   |        423.688 |           66.3247   |  0.381275   |                         71.4038 |                       72.5694 |       0.986142 |             0.986506 |                577 |               288 |
| test    | LSTM_ONNX         |        5757 | 299.151 | 422.775 |    1.68234 |     1.68735 | 0.97259  | 0.0266518 |               1.99403 |          433.368 |         2.32053 |        9.21679  |        422.675 |           -9.21679  | -0.0529838  |                         61.6984 |                       73.7847 |       0.986222 |             0.985237 |                577 |               288 |
| test    | Persistence_lag1h |        5757 | 557.397 | 726.027 |    3.16874 |     3.18587 | 0.919167 | 0.0457688 |               3.11779 |          676.504 |         3.49982 |        0.438126 |        726.027 |           -0.438126 | -0.00251862 |                         59.2721 |                       72.4826 |       0.95958  |             0.955799 |                577 |               288 |

### 5.2 Validation Set
| split      | model             |   n_samples |     mae |    rmse |   mape_pct |   smape_pct |       r2 |     nrmse |   peak_mape_pct_top10 |   peak_mae_top10 |   top5_mape_pct |   residual_mean |   residual_std |   mean_signed_error |    bias_pct |   peak_underprediction_rate_pct |   low_overprediction_rate_pct |   pearson_corr |   spearman_rank_corr |   peak_count_top10 |   peak_count_top5 |
|:-----------|:------------------|------------:|--------:|--------:|-----------:|------------:|---------:|----------:|----------------------:|-----------------:|----------------:|----------------:|---------------:|--------------------:|------------:|--------------------------------:|------------------------------:|---------------:|---------------------:|-------------------:|------------------:|
| validation | LSTM_ONNX         |        5756 | 197.741 | 320.091 |    1.15537 |     1.15504 | 0.976073 | 0.0237403 |              1.27507  |          265.663 |        1.20099  |       13.0432   |        319.825 |          -13.0432   | -0.0765964  |                         60.9375 |                       55.4688 |       0.987987 |             0.988123 |                576 |               288 |
| validation | XGBoost_ONNX      |        5756 | 218.266 | 323.134 |    1.31376 |     1.30661 | 0.975615 | 0.023966  |              0.667475 |          139.254 |        0.654377 |      -49.905    |        319.257 |           49.905    |  0.293069   |                         61.6319 |                       52.691  |       0.988283 |             0.988902 |                576 |               288 |
| validation | Persistence_lag1h |        5756 | 395.316 | 514.148 |    2.29485 |     2.29717 | 0.938266 | 0.038133  |              2.578    |          536.996 |        2.46202  |        0.480966 |        514.148 |           -0.480966 | -0.00282449 |                         55.7292 |                       61.1979 |       0.969137 |             0.967244 |                576 |               288 |

### 5.3 Train Set
| split   | model             |   n_samples |     mae |    rmse |   mape_pct |   smape_pct |       r2 |     nrmse |   peak_mape_pct_top10 |   peak_mae_top10 |   top5_mape_pct |   residual_mean |   residual_std |   mean_signed_error |    bias_pct |   peak_underprediction_rate_pct |   low_overprediction_rate_pct |   pearson_corr |   spearman_rank_corr |   peak_count_top10 |   peak_count_top5 |
|:--------|:------------------|------------:|--------:|--------:|-----------:|------------:|---------:|----------:|----------------------:|-----------------:|----------------:|----------------:|---------------:|--------------------:|------------:|--------------------------------:|------------------------------:|---------------:|---------------------:|-------------------:|------------------:|
| train   | LSTM_ONNX         |       26861 | 132.435 | 191.63  |   0.911032 |    0.910444 | 0.99304  | 0.0123502 |              0.907518 |          167.868 |         0.92254 |        2.89138  |        191.608 |           -2.89138  | -0.0199466  |                         59.9181 |                       52.2241 |       0.996516 |             0.996406 |               2687 |              1344 |
| train   | XGBoost_ONNX      |       26861 | 205.809 | 277.12  |   1.45733  |    1.4527   | 0.985444 | 0.0178599 |              1.07644  |          196.6   |         0.68849 |      -24.7002   |        276.017 |           24.7002   |  0.170397   |                         35.9137 |                       60.9529 |       0.992856 |             0.99363  |               2687 |              1344 |
| train   | Persistence_lag1h |       26861 | 340.055 | 449.827 |   2.33524  |    2.33745  | 0.961647 | 0.0289906 |              2.29195  |          423.717 |         2.27007 |        0.196113 |        449.826 |           -0.196113 | -0.00135291 |                         55.7499 |                       60.5248 |       0.980826 |             0.979079 |               2687 |              1344 |

## 6. Winner Ranking (Test Priority Rule)
Ranking keys: lowest Peak MAPE (top-10%), then MAPE, then RMSE.

| model             |   peak_mape_pct_top10 |   mape_pct |    rmse |       r2 |
|:------------------|----------------------:|-----------:|--------:|---------:|
| XGBoost_ONNX      |               1.66557 |    1.79624 | 428.848 | 0.971797 |
| LSTM_ONNX         |               1.99403 |    1.68234 | 422.775 | 0.97259  |
| Persistence_lag1h |               3.11779 |    3.16874 | 726.027 | 0.919167 |

Selected winner by defined rule: **XGBoost_ONNX**

## 7. Robustness and Stability
### 7.1 Rolling Backtest (TimeSeriesSplit)
|   fold | model             | start               | end                 |     mae |    rmse |   mape_pct |   peak_mape_pct_top10 |       r2 |
|-------:|:------------------|:--------------------|:--------------------|--------:|--------:|-----------:|----------------------:|---------:|
|      1 | LSTM_ONNX         | 2020-10-07 16:00:00 | 2021-07-01 02:00:00 | 123.124 | 173.347 |   0.842436 |              0.849555 | 0.990456 |
|      2 | LSTM_ONNX         | 2021-07-01 03:00:00 | 2022-03-24 13:00:00 | 155.851 | 230.025 |   1.05759  |              1.13851  | 0.981809 |
|      3 | LSTM_ONNX         | 2022-03-24 14:00:00 | 2022-12-16 00:00:00 | 144.867 | 203.085 |   0.901567 |              0.915364 | 0.991884 |
|      4 | LSTM_ONNX         | 2022-12-16 01:00:00 | 2023-09-08 11:00:00 | 186.863 | 305.131 |   1.10841  |              1.2577   | 0.979073 |
|      5 | LSTM_ONNX         | 2023-09-08 12:00:00 | 2024-05-31 22:00:00 | 290.568 | 410.498 |   1.63607  |              1.90754  | 0.97318  |
|      1 | Persistence_lag1h | 2020-10-07 16:00:00 | 2021-07-01 02:00:00 | 351.606 | 459.738 |   2.40416  |              2.10129  | 0.932867 |
|      2 | Persistence_lag1h | 2021-07-01 03:00:00 | 2022-03-24 13:00:00 | 326.049 | 434.213 |   2.23267  |              1.933    | 0.935178 |
|      3 | Persistence_lag1h | 2022-03-24 14:00:00 | 2022-12-16 00:00:00 | 367.803 | 481.169 |   2.26367  |              2.29787  | 0.954441 |
|      4 | Persistence_lag1h | 2022-12-16 01:00:00 | 2023-09-08 11:00:00 | 418.823 | 549.993 |   2.47754  |              2.6371   | 0.932009 |
|      5 | Persistence_lag1h | 2023-09-08 12:00:00 | 2024-05-31 22:00:00 | 537.622 | 702.335 |   3.05851  |              3.03446  | 0.921489 |
|      1 | XGBoost_ONNX      | 2020-10-07 16:00:00 | 2021-07-01 02:00:00 | 189.762 | 253.481 |   1.33329  |              1.29519  | 0.979592 |
|      2 | XGBoost_ONNX      | 2021-07-01 03:00:00 | 2022-03-24 13:00:00 | 211.935 | 291.359 |   1.44612  |              1.586    | 0.970814 |
|      3 | XGBoost_ONNX      | 2022-03-24 14:00:00 | 2022-12-16 00:00:00 | 219.098 | 294.428 |   1.37005  |              0.683563 | 0.982942 |
|      4 | XGBoost_ONNX      | 2022-12-16 01:00:00 | 2023-09-08 11:00:00 | 224.462 | 329.035 |   1.36792  |              0.672962 | 0.975666 |
|      5 | XGBoost_ONNX      | 2023-09-08 12:00:00 | 2024-05-31 22:00:00 | 299.566 | 414.441 |   1.73056  |              1.57473  | 0.972662 |

### 7.2 Time-window Slices (Test)
#### By Year
| split   | model             | slice_type   |   slice_value |     mae |    rmse |   mape_pct |
|:--------|:------------------|:-------------|--------------:|--------:|--------:|-----------:|
| test    | LSTM_ONNX         | year         |          2023 | 233.328 | 314.415 |    1.39069 |
| test    | Persistence_lag1h | year         |          2023 | 520.409 | 676.837 |    3.09701 |
| test    | XGBoost_ONNX      | year         |          2023 | 261.57  | 358.888 |    1.60001 |
| test    | LSTM_ONNX         | year         |          2024 | 337.234 | 474.295 |    1.85107 |
| test    | Persistence_lag1h | year         |          2024 | 578.797 | 753.02  |    3.21024 |
| test    | XGBoost_ONNX      | year         |          2024 | 340.149 | 464.537 |    1.90977 |

#### By Season
| split   | model             | slice_type   | slice_value   |     mae |    rmse |   mape_pct |
|:--------|:------------------|:-------------|:--------------|--------:|--------:|-----------:|
| test    | LSTM_ONNX         | season       | post_monsoon  | 230.2   | 299.084 |    1.25008 |
| test    | Persistence_lag1h | season       | post_monsoon  | 464.966 | 568.815 |    2.52158 |
| test    | XGBoost_ONNX      | season       | post_monsoon  | 193.911 | 253.27  |    1.07145 |
| test    | LSTM_ONNX         | season       | summer        | 373.61  | 524.971 |    2.00325 |
| test    | Persistence_lag1h | season       | summer        | 587.396 | 752.505 |    3.15023 |
| test    | XGBoost_ONNX      | season       | summer        | 372.326 | 513.401 |    2.03635 |
| test    | LSTM_ONNX         | season       | winter        | 257.901 | 353.542 |    1.5346  |
| test    | Persistence_lag1h | season       | winter        | 555.16  | 736.645 |    3.32677 |
| test    | XGBoost_ONNX      | season       | winter        | 291.131 | 387.303 |    1.77499 |

#### Day vs Night
| split   | model             | slice_type   | slice_value   |     mae |    rmse |   mape_pct |
|:--------|:------------------|:-------------|:--------------|--------:|--------:|-----------:|
| test    | LSTM_ONNX         | day_night    | daytime       | 359.602 | 489.179 |    1.93419 |
| test    | Persistence_lag1h | day_night    | daytime       | 712.021 | 875.095 |    3.89176 |
| test    | XGBoost_ONNX      | day_night    | daytime       | 362.522 | 475.795 |    2.01155 |
| test    | LSTM_ONNX         | day_night    | nighttime     | 214.414 | 306.354 |    1.3293  |
| test    | Persistence_lag1h | day_night    | nighttime     | 340.653 | 438.804 |    2.15524 |
| test    | XGBoost_ONNX      | day_night    | nighttime     | 239.616 | 352.671 |    1.49442 |

## 8. Residual and Bias Diagnostics
| split      | model             |   residual_mean |   residual_std |   mean_signed_error |     bias_pct |
|:-----------|:------------------|----------------:|---------------:|--------------------:|-------------:|
| test       | LSTM_ONNX         |        9.21679  |        422.675 |           -9.21679  |  0.0148481   |
| test       | Persistence_lag1h |        0.438126 |        726.027 |           -0.438126 |  0.0818936   |
| test       | XGBoost_ONNX      |      -66.3247   |        423.688 |           66.3247   |  0.442841    |
| train      | LSTM_ONNX         |        2.89138  |        191.608 |           -2.89138  |  1.60447e-05 |
| train      | Persistence_lag1h |        0.196113 |        449.826 |           -0.196113 |  0.0449665   |
| train      | XGBoost_ONNX      |      -24.7002   |        276.017 |           24.7002   |  0.186702    |
| validation | LSTM_ONNX         |       13.0432   |        319.825 |          -13.0432   | -0.0454386   |
| validation | Persistence_lag1h |        0.480966 |        514.148 |           -0.480966 |  0.0414785   |
| validation | XGBoost_ONNX      |      -49.905    |        319.257 |           49.905    |  0.29571     |

## 9. Statistical Significance
Paired tests were computed on identical timestamps of the test set.

| model_a      | model_b           |   mean_abs_error_a |   mean_abs_error_b |   paired_t_stat |   paired_t_p |   wilcoxon_stat |   wilcoxon_p |   dm_stat_abs_error |   dm_p_abs_error |
|:-------------|:------------------|-------------------:|-------------------:|----------------:|-------------:|----------------:|-------------:|--------------------:|-----------------:|
| XGBoost_ONNX | LSTM_ONNX         |            311.349 |            299.151 |         3.54972 |  0.000388714 |     7.80345e+06 |  0.000125117 |             2.59087 |       0.00957329 |
| XGBoost_ONNX | Persistence_lag1h |            311.349 |            557.397 |       -46.0295  |  0           |     3.13173e+06 |  0           |           -35.9253  |       0          |
| LSTM_ONNX    | Persistence_lag1h |            299.151 |            557.397 |       -55.3732  |  0           |     2.0827e+06  |  0           |           -52.109   |       0          |

Interpretation guidance:
- p < 0.05 suggests the error difference is statistically significant.
- Diebold-Mariano values use absolute-error loss differential.

## 10. Bootstrap Confidence Intervals
95% bootstrap confidence intervals on test metrics.

| model             |   mae_ci_low |   mae_ci_high |   rmse_ci_low |   rmse_ci_high |   mape_ci_low |   mape_ci_high |
|:------------------|-------------:|--------------:|--------------:|---------------:|--------------:|---------------:|
| XGBoost_ONNX      |      303.461 |       318.682 |       409.183 |        451.304 |       1.75386 |        1.83748 |
| LSTM_ONNX         |      290.998 |       306.812 |       401.849 |        447.266 |       1.64116 |        1.72213 |
| Persistence_lag1h |      545.537 |       569.205 |       708.953 |        743.554 |       3.10221 |        3.23163 |

## 11. Operational Cost Layer
Training/tuning times are not directly inferable from static pre-trained artifacts, but inference cost and model size are measured.

| model        |   avg_runtime_sec |   n_samples |   runtime_per_1k_samples_sec |   model_file_mb |   training_time_sec |   tuning_time_sec |
|:-------------|------------------:|------------:|-----------------------------:|----------------:|--------------------:|------------------:|
| XGBoost_ONNX |          0.126829 |        5757 |                    0.0220304 |        0.920596 |                 nan |               nan |
| LSTM_ONNX    |          2.36142  |        4000 |                    0.590354  |        1.09276  |                 nan |               nan |

## 12. Graphs
- ![01_actual_vs_predicted_test.png](plots/01_actual_vs_predicted_test.png)
- ![02_residual_distribution_test.png](plots/02_residual_distribution_test.png)
- ![03_residuals_over_time_test.png](plots/03_residuals_over_time_test.png)
- ![04_mae_by_hour_test.png](plots/04_mae_by_hour_test.png)
- ![05_mape_by_month_test.png](plots/05_mape_by_month_test.png)
- ![06_predicted_vs_actual_scatter_test.png](plots/06_predicted_vs_actual_scatter_test.png)
- ![07_foldwise_backtest_mape.png](plots/07_foldwise_backtest_mape.png)
- ![08_peak_mape_test_bar.png](plots/08_peak_mape_test_bar.png)
- ![09_runtime_vs_accuracy.png](plots/09_runtime_vs_accuracy.png)

## 13. Decision Summary
- Best peak-demand performer (primary criterion): **XGBoost_ONNX**
- Accuracy-vs-cost should be judged using both Section 5 and Section 11.
- If two models are close on peak MAPE, lower runtime and simpler operations should be preferred.

## 14. Notes and Constraints
- This is intentionally a from-scratch implementation and does not reuse prior comparison outputs.
- Reserved period June 2024 to June 2025 is left untouched for later walk-forward experiments.
- Report tables were generated directly from this run and saved under model-comparison/tables.

---
Generated by: model-comparison/run_model_comparison.py
