# Gujarat Electricity Demand Forecasting — Retraining Benchmark Pipeline Plan

**Project:** Walk-Forward Retraining Strategy Evaluation  
**Region:** Gujarat, India  
**Model:** XGBoost with Optuna Bayesian Optimization  
**Author:** _(your name)_  
**Date:** April 2026  

---

## Table of Contents

1. [Objective](#1-objective)
2. [Dataset Split Design](#2-dataset-split-design)
3. [Pipeline Architecture Overview](#3-pipeline-architecture-overview)
4. [Phase 1 — Baseline Model](#4-phase-1--baseline-model)
5. [Phase 2 — Walk-Forward Benchmark Loop](#5-phase-2--walk-forward-benchmark-loop)
6. [Three Retraining Strategies](#6-three-retraining-strategies)
7. [Feature Engineering Protocol](#7-feature-engineering-protocol)
8. [Logging and Tracking Schema](#8-logging-and-tracking-schema)
9. [Phase 3 — Results Analysis and Visualization](#9-phase-3--results-analysis-and-visualization)
10. [Decision Framework](#10-decision-framework)
11. [File and Folder Structure](#11-file-and-folder-structure)
12. [Module-wise Implementation Plan](#12-module-wise-implementation-plan)
13. [Risk and Edge Cases](#13-risk-and-edge-cases)
14. [Expected Outcomes and Hypothesis](#14-expected-outcomes-and-hypothesis)
15. [Execution Checklist](#15-execution-checklist)

---

## 1. Objective

This pipeline empirically answers one question:

> **Which retraining strategy — Full Optuna, Warm-Start Refinement, or Fixed-Param Refit — delivers the best accuracy-versus-compute tradeoff for Gujarat hourly electricity demand forecasting?**

### Why this matters

- Electricity demand is non-stationary. A model trained on 2021–2024 data will drift as industrial load, climate patterns, and behavioral factors evolve.
- Three retraining strategies exist on a spectrum of compute cost vs. adaptability.
- Rather than assuming which strategy is best, this benchmark **measures** it on real Gujarat data across four quarterly forecast windows.

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Test R² | > 0.80 |
| Test MAPE | < 5% |
| Peak MAPE | < 5% |
| Strategy winner declared | Yes, with statistical confidence |

---

## 2. Dataset Split Design

### Data Availability

```
Full Dataset:   Jan 2021 ─────────────────────────── Jun 2025
                │                                         │
                ▼                                         ▼
           Start of data                          Last available row
```

### Fixed Splits

```
┌─────────────────────────────────────────────────────────────────┐
│  ZONE A — Model Training Base         │  ZONE B — Benchmark     │
│  Jan 2021 ──────────── Jun 2024       │  Jul 2024 ── Jun 2025   │
│                                       │                         │
│  Used for Phase 1 baseline model      │  Never touched until    │
│  training, validation, and test.      │  walk-forward reveals   │
│                                       │  it quarter by quarter. │
└───────────────────────────────────────┴─────────────────────────┘
```

### Within Zone A — Initial Chronological Split

| Split | Date Range | Proportion | Purpose |
|-------|-----------|------------|---------|
| Train | Jan 2021 – ~Dec 2023 | 70% | Model learning |
| Validation | ~Jan 2024 – ~Apr 2024 | 15% | Hyperparameter tuning, early stopping |
| Test (Zone A) | ~Apr 2024 – Jun 2024 | 15% | Baseline model final evaluation |

> **Strict rule:** No shuffling at any stage. All splits are purely chronological.

### Zone B — Walk-Forward Quarterly Windows

```
Quarter    Predict Window          Retrain Uses Data Up To
────────────────────────────────────────────────────────────
Q3-2024    Jul 01 – Sep 30, 2024   Jun 30, 2024
Q4-2024    Oct 01 – Dec 31, 2024   Sep 30, 2024
Q1-2025    Jan 01 – Mar 31, 2025   Dec 31, 2024
Q2-2025    Apr 01 – Jun 30, 2025   Mar 31, 2025
```

> Each quarter: retrain first → predict next → evaluate → advance window.

---

## 3. Pipeline Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  Raw CSV (Jan 2021 – Jun 2025)                                       │
│  → datetime parse → sort → dedup check → schema validation           │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                          │
│  Weather interactions, lag features, rolling statistics,             │
│  thermal markers, momentum terms — all leakage-safe                  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│               PHASE 1 — BASELINE MODEL                               │
│  Train on Zone A (Jan 2021 – Jun 2024)                               │
│  Full Optuna Run (200 trials, rolling TimeSeriesSplit CV)            │
│  Save: best_params, best_iteration, model artifact, metrics          │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│            PHASE 2 — WALK-FORWARD BENCHMARK LOOP                     │
│                                                                      │
│  For each quarter Q in [Q3-2024, Q4-2024, Q1-2025, Q2-2025]:        │
│                                                                      │
│    ┌──────────────────────────────────────────────────────────┐      │
│    │  Strategy A — Full Optuna (200 trials, fresh search)     │      │
│    │  Strategy B — Warm-Start  (50 trials, seeded from last)  │      │
│    │  Strategy C — Fixed-Param (0 trials, use saved params)   │      │
│    └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│    Evaluate each strategy independently on quarter actuals           │
│    Log all metrics + runtime to results_log.csv                      │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│              PHASE 3 — RESULTS ANALYSIS                              │
│  Comparison tables, MAPE-over-time plots, runtime vs accuracy,       │
│  peak MAPE heatmap, final recommendation                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1 — Baseline Model

### 4.1 Purpose

Establish a fixed reference point against which all walk-forward results are compared. This model represents the "day-zero deploy" scenario — trained on all available history up to June 2024.

### 4.2 Steps

```
Step 1:  Load full dataset, apply feature engineering
Step 2:  Chronological split: Train / Validation / Test (70/15/15 on Zone A)
Step 3:  Run Full Optuna (200 trials)
          - Rolling TimeSeriesSplit CV (5 folds) on Train + Validation
          - log1p target transform
          - Peak-weighted samples (PEAK_WEIGHT = 60.0, top-90th percentile)
          - Blended objective: 0.65 × nRMSE% + 0.35 × Peak MAPE%
Step 4:  Two-stage final training
          - Stage 1: Fit on Train, early stop on Validation → get best_iteration
          - Stage 2: Refit on Train + Validation with calibrated tree count
Step 5:  Evaluate on Zone A Test (Apr–Jun 2024)
Step 6:  Save artifacts
          - baseline_model.json        (XGBoost model)
          - baseline_params.json       (best hyperparameters)
          - baseline_metrics.json      (R², MAE, RMSE, MAPE, Peak MAPE)
          - baseline_best_iteration    (integer, for fixed-param strategy)
```

### 4.3 Artifacts Saved

```python
{
  "best_params": {
    "n_estimators": ...,
    "max_depth": ...,
    "learning_rate": ...,
    "subsample": ...,
    "colsample_bytree": ...,
    "min_child_weight": ...,
    "gamma": ...,
    "reg_alpha": ...,
    "reg_lambda": ...
  },
  "best_iteration": ...,
  "optuna_study": "saved as baseline_study.pkl",
  "zone_a_test_metrics": {
    "r2": ..., "mae": ..., "rmse": ..., "mape": ..., "peak_mape": ...
  }
}
```

---

## 5. Phase 2 — Walk-Forward Benchmark Loop

### 5.1 Loop Logic

```python
quarters = [
    ("Q3-2024", "2024-07-01", "2024-09-30", "2024-06-30"),
    ("Q4-2024", "2024-10-01", "2024-12-31", "2024-09-30"),
    ("Q1-2025", "2025-01-01", "2025-03-31", "2024-12-31"),
    ("Q2-2025", "2025-04-01", "2025-06-30", "2025-03-31"),
]

for quarter_name, pred_start, pred_end, train_cutoff in quarters:
    train_data   = df[df.datetime <= train_cutoff]   # expanding window
    predict_data = df[(df.datetime >= pred_start) &
                      (df.datetime <= pred_end)]

    for strategy in ["full_optuna", "warm_start", "fixed_param"]:
        model, params = retrain(strategy, train_data, prior_best_params)
        predictions   = model.predict(predict_data[feature_cols])
        metrics       = evaluate(predictions, predict_data["demand_mw"])
        log_results(quarter_name, strategy, metrics, runtime)
```

### 5.2 Expanding Window Design

Each retrain uses **all data from Jan 2021 up to the cutoff date**, not just the most recent quarter. This is the recommended default because:

- Gujarat has long-term industrial growth trends that older data encodes
- Seasonal recurrence (2+ annual cycles) improves temperature and holiday feature learning
- Optuna and early stopping will naturally down-weight outdated patterns via CV

> **Optional variant:** A rolling-window version (fixed 24-month lookback) can be added as Strategy D in a future extension.

### 5.3 Within Each Quarter — Chronological Sub-split for Retraining

Even within the retrain phase, no data leakage occurs:

```
Train cutoff = Jun 30, 2024  (for Q3-2024 prediction)

Sub-split for retraining:
  Inner Train:      Jan 2021 – ~Apr 2024   (85% of train_data)
  Inner Validation: ~Apr 2024 – Jun 2024   (15% of train_data)

→ Inner validation used for early stopping only
→ Final model refitted on full train_data before predicting Q3-2024
```

---

## 6. Three Retraining Strategies

### Strategy A — Full Optuna (200 Trials)

```
Description:    Complete Bayesian hyperparameter search from scratch
Trials:         200
Initialization: Random (no prior seeding)
CV:             5-fold rolling TimeSeriesSplit on inner train+validation
Runtime est.:   3–5 hours per quarter
When to use:    Annual deep retrain; baseline establishment
```

**Search space:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `n_estimators` | 200 – 1500 | int |
| `max_depth` | 3 – 10 | int |
| `learning_rate` | 0.005 – 0.3 | log-uniform |
| `subsample` | 0.5 – 1.0 | float |
| `colsample_bytree` | 0.4 – 1.0 | float |
| `min_child_weight` | 1 – 20 | int |
| `gamma` | 0 – 5 | float |
| `reg_alpha` | 1e-8 – 10 | log-uniform |
| `reg_lambda` | 1e-8 – 10 | log-uniform |

**Blended objective (same as baseline):**
```
score = 0.65 × (RMSE / mean_demand × 100) + 0.35 × Peak_MAPE%
```

---

### Strategy B — Warm-Start Refinement (50 Trials)

```
Description:    Optuna study seeded with prior best params, then refines
Trials:         50
Initialization: Prior best_params injected as first trial via add_trial()
CV:             5-fold rolling TimeSeriesSplit (same as full)
Runtime est.:   45–90 minutes per quarter
When to use:    Standard quarterly retrain
```

**Warm-start seeding logic:**

```python
def warm_start_study(prior_best_params, n_trials=50):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    # Inject prior best as the first trial
    study.enqueue_trial(prior_best_params)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

**Key behavior:**
- First trial evaluates the prior best directly
- TPE sampler uses it as a high-quality anchor and explores nearby space
- 50 trials is sufficient to refine but not fully re-explore the space
- `prior_best_params` comes from the last Full Optuna run (carried forward)

---

### Strategy C — Fixed-Param Refit (0 Trials)

```
Description:    No hyperparameter search; retrain with saved params on new data
Trials:         0 (no Optuna)
Initialization: Uses baseline_params.json (or last Full Optuna params)
CV:             None (direct two-stage fit)
Runtime est.:   5–15 minutes per quarter
When to use:    Emergency retrain; compute-constrained environments
```

**Refit logic:**

```python
def fixed_param_retrain(train_data, saved_params, saved_best_iteration):
    # Stage 1: calibrate tree count on inner split
    model_stage1 = XGBRegressor(**saved_params, n_estimators=2000)
    model_stage1.fit(
        X_inner_train, y_inner_train,
        eval_set=[(X_inner_val, y_inner_val)],
        early_stopping_rounds=50,
        sample_weight=peak_weights_inner
    )
    calibrated_iter = model_stage1.best_iteration

    # Stage 2: refit on all train_data with calibrated count
    model_final = XGBRegressor(**saved_params, n_estimators=calibrated_iter)
    model_final.fit(X_full_train, y_full_train, sample_weight=peak_weights_full)
    return model_final
```

> **Design note:** Fixed-Param still re-calibrates tree count via Stage 1 early stopping. Only the hyperparameters are frozen, not the number of boosting rounds. This is more realistic than a purely static refit.

---

### Strategy Comparison Summary

| Property | Full Optuna | Warm-Start | Fixed-Param |
|----------|------------|------------|-------------|
| Hyperparams updated | Yes, fully | Yes, partially | No |
| Tree count updated | Yes | Yes | Yes (via early stop) |
| Optuna trials | 200 | 50 | 0 |
| Compute cost | High | Medium | Low |
| Adapts to new regime | Fully | Partially | No (structure only) |
| Risk of overfitting to noise | Low (broad search) | Low (anchored) | Very low |
| Recommended cadence | Annual | Quarterly | Emergency only |

---

## 7. Feature Engineering Protocol

Feature engineering is identical across all phases and strategies to ensure fair comparison.

### 7.1 Leakage-Safe Rules

- All lag features use strictly past values: `lag_1h`, `lag_24h`, `lag_48h`, `lag_168h`, `lag_336h`
- Rolling and EWM statistics computed on **shifted** target: `target.shift(1)` before any rolling window
- No contemporaneous target values in feature set at inference time
- Test and prediction windows are never seen during feature engineering on training data

### 7.2 Feature Categories

**Temporal:**
```
hour, day_of_week, month, week_of_year,
is_weekend, is_holiday (Gujarat public holidays),
hour_sin, hour_cos, month_sin, month_cos
```

**Weather:**
```
temperature_2m, dewpoint_2m, apparent_temperature,
relative_humidity_2m (or derived),
precipitation (or 0.0 fallback),
cloud_cover, shortwave_radiation, wind_speed_10m
```

**Weather Interactions:**
```
temp_x_peak, solar_x_peak, temp_x_hour_sin,
weather_balance, temp_x_humidity, precip_x_peak,
cloud_x_temp, apparent_minus_temp, rad_x_temp
```

**Demand Memory:**
```
lag_1h, lag_24h, lag_48h, lag_168h, lag_336h,
lag_momentum_1_24, lag_momentum_24_168,
rolling_mean_6h_from_target,
rolling_max_24h_from_target,
ewm_24h_from_target
```

**Thermal Regime:**
```
cdd_24 (cooling degree days, base 24°C),
hdd_18 (heating degree days, base 18°C),
hot_flag (top quartile temperature binary),
hot_hour_interaction
```

### 7.3 Missing Value Policy

- Drop rows with any null in feature_cols or target
- Log how many rows are dropped per retraining run
- If > 2% of rows dropped in any window, raise a data quality warning

---

## 8. Logging and Tracking Schema

Every (Quarter × Strategy) combination writes one row to `results_log.csv`.

### 8.1 Results Log Schema

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | Unique ID, e.g. `Q3-2024_full_optuna` |
| `quarter` | str | Q3-2024, Q4-2024, Q1-2025, Q2-2025 |
| `strategy` | str | full_optuna / warm_start / fixed_param |
| `train_start` | date | Always Jan 1, 2021 (expanding window) |
| `train_end` | date | Cutoff date for this quarter |
| `n_train_rows` | int | Row count after feature engineering + dropna |
| `n_predict_rows` | int | Row count in prediction quarter |
| `optuna_trials` | int | 200 / 50 / 0 |
| `runtime_minutes` | float | Wall-clock time for full retrain+predict cycle |
| `best_n_estimators` | int | Final tree count used |
| `test_r2` | float | R² on quarter actuals |
| `test_mae` | float | MAE (MW) |
| `test_rmse` | float | RMSE (MW) |
| `test_mape` | float | MAPE % |
| `test_peak_mape` | float | Peak MAPE % (above 90th percentile of quarter) |
| `peak_count` | int | Number of peak hours in quarter |
| `vs_baseline_mape_delta` | float | MAPE minus baseline Phase 1 MAPE (negative = better) |
| `vs_baseline_peak_mape_delta` | float | Same for peak MAPE |
| `best_params_json` | str | JSON-serialized hyperparameters used |
| `notes` | str | Any warnings or anomalies |

### 8.2 Model Artifact Naming Convention

```
artifacts/
├── baseline/
│   ├── baseline_model.json
│   ├── baseline_params.json
│   ├── baseline_study.pkl
│   └── baseline_metrics.json
│
├── Q3-2024/
│   ├── full_optuna_model.json
│   ├── full_optuna_params.json
│   ├── warm_start_model.json
│   ├── warm_start_params.json
│   └── fixed_param_model.json
│
├── Q4-2024/
│   └── ... (same structure)
│
├── Q1-2025/
│   └── ...
│
└── Q2-2025/
    └── ...
```

---

## 9. Phase 3 — Results Analysis and Visualization

After all 12 runs (4 quarters × 3 strategies) complete, generate the following.

### 9.1 Summary Table

A 12-row table sorted by `test_peak_mape` ascending:

```
quarter    strategy      mape%   peak_mape%   rmse    r2      runtime_min
─────────────────────────────────────────────────────────────────────────
Q3-2024    full_optuna    X.XX      X.XX       XXXX   0.XX      XXX
Q3-2024    warm_start     X.XX      X.XX       XXXX   0.XX      XXX
Q3-2024    fixed_param    X.XX      X.XX       XXXX   0.XX      XXX
Q4-2024    full_optuna    ...
...
```

### 9.2 Visualization Plan

**Plot 1 — MAPE Over Time (Line Chart)**
```
X-axis: Quarter (Q3-2024 → Q2-2025)
Y-axis: MAPE %
Lines:  One per strategy (3 lines) + baseline reference (dashed horizontal)
Purpose: Shows whether strategies diverge over time or after regime changes
```

**Plot 2 — Peak MAPE Over Time (Line Chart)**
```
Same structure as Plot 1 but for Peak MAPE
Purpose: Operational priority metric — peak accuracy is what matters most
```

**Plot 3 — Runtime vs Accuracy Scatter**
```
X-axis: Runtime (minutes, log scale)
Y-axis: Average MAPE % across 4 quarters
Points: One per strategy, sized by Peak MAPE
Purpose: Core tradeoff visualization — picks the Pareto-optimal strategy
```

**Plot 4 — Peak MAPE Heatmap**
```
Rows:    Quarters (Q3-2024 to Q2-2025)
Columns: Strategies (Full Optuna, Warm Start, Fixed Param)
Values:  Peak MAPE % (color-coded: green < 4%, yellow 4–6%, red > 6%)
Purpose: Quick visual scan of where each strategy fails or excels
```

**Plot 5 — Actual vs Predicted Overlay (per quarter, per strategy)**
```
12 subplots: one per (quarter, strategy) combination
X-axis: Datetime within quarter
Y-axis: demand_mw
Lines:  Actual (black), Predicted (colored by strategy)
Purpose: Qualitative check for systematic bias, lag, amplitude compression
```

**Plot 6 — Residual Distribution Comparison**
```
KDE plots of (actual - predicted) for all 3 strategies, per quarter
Purpose: Shows whether any strategy has systematic over- or under-prediction bias
```

### 9.3 Aggregate Score Table

Compute average across all 4 quarters per strategy:

| Strategy | Avg MAPE% | Avg Peak MAPE% | Avg RMSE | Avg R² | Avg Runtime (min) |
|----------|-----------|---------------|----------|--------|------------------|
| Full Optuna | | | | | |
| Warm Start | | | | | |
| Fixed Param | | | | | |

---

## 10. Decision Framework

### 10.1 Winner Selection Logic

```python
# Primary criterion: lowest average Peak MAPE across 4 quarters
# Tiebreaker: lowest runtime if Peak MAPE within 0.3% of each other

ranked = results.groupby("strategy").agg(
    avg_peak_mape = ("test_peak_mape", "mean"),
    avg_mape      = ("test_mape", "mean"),
    avg_runtime   = ("runtime_minutes", "mean")
).sort_values("avg_peak_mape")

# Check if warm_start is within 0.3% of full_optuna
if (ranked.loc["full_optuna"]["avg_peak_mape"] -
    ranked.loc["warm_start"]["avg_peak_mape"]) < 0.3:
    winner = "warm_start"  # better compute tradeoff
else:
    winner = ranked.index[0]  # best raw accuracy
```

### 10.2 Recommended Production Strategy

Based on the decision logic, output a final recommendation:

```
IF   warm_start avg Peak MAPE within 0.3% of full_optuna:
     → USE warm_start quarterly + full_optuna annually

ELIF fixed_param avg Peak MAPE within 0.5% of warm_start:
     → USE fixed_param quarterly + warm_start semi-annually + full_optuna annually

ELSE:
     → USE full_optuna quarterly (accuracy gap is too large to accept tradeoff)
```

### 10.3 Mandatory Retrain Triggers (regardless of schedule)

Even after the benchmark determines the best scheduled strategy, always retrain immediately if:

```
Condition                                   Action
──────────────────────────────────────────────────────────────────
7-day rolling MAPE on live data > 6%        Warm-Start retrain
7-day rolling Peak MAPE > 8%               Full Optuna retrain
R² on last 30 days < 0.75                  Full Optuna retrain
New industrial zone / major load addition   Full Optuna retrain
Tariff policy change affecting demand       Full Optuna retrain
Weather station coverage change             Full Optuna retrain
```

---

## 11. File and Folder Structure

```
gujarat_retraining_benchmark/
│
├── data/
│   ├── raw/
│   │   └── gujarat_demand_weather_2021_2025.csv
│   └── processed/
│       └── gujarat_features_engineered.parquet
│
├── artifacts/
│   ├── baseline/
│   ├── Q3-2024/
│   ├── Q4-2024/
│   ├── Q1-2025/
│   └── Q2-2025/
│
├── results/
│   ├── results_log.csv              (all 12 runs, full schema)
│   ├── aggregate_summary.csv        (per-strategy averages)
│   └── recommendation.txt           (auto-generated final decision)
│
├── plots/
│   ├── mape_over_time.png
│   ├── peak_mape_over_time.png
│   ├── runtime_vs_accuracy.png
│   ├── peak_mape_heatmap.png
│   ├── actual_vs_predicted/
│   │   ├── Q3-2024_full_optuna.png
│   │   └── ... (12 plots)
│   └── residuals/
│       └── ... (12 plots)
│
├── src/
│   ├── config.py                    (paths, constants, PEAK_WEIGHT, etc.)
│   ├── data_loader.py               (ingestion, datetime parsing, dedup)
│   ├── feature_engineering.py       (add_extra_features, leakage-safe)
│   ├── splitter.py                  (chronological splits, walk-forward windows)
│   ├── peak_weights.py              (sample weight computation)
│   ├── baseline.py                  (Phase 1 full Optuna run)
│   ├── strategies/
│   │   ├── full_optuna.py
│   │   ├── warm_start.py
│   │   └── fixed_param.py
│   ├── evaluator.py                 (metrics: R², MAE, RMSE, MAPE, Peak MAPE)
│   ├── logger.py                    (results_log.csv writer)
│   └── visualizer.py                (all 6 plot types)
│
├── notebooks/
│   ├── 01_baseline_training.ipynb
│   ├── 02_walkforward_benchmark.ipynb
│   └── 03_results_analysis.ipynb
│
├── run_benchmark.py                 (main entry point — runs full pipeline)
└── README.md
```

---

## 12. Module-wise Implementation Plan

### Module 1: `config.py`

```python
# Constants
PEAK_WEIGHT         = 60.0
PEAK_PERCENTILE     = 0.90
LOG_TRANSFORM       = True           # use log1p / expm1
OPTUNA_FULL_TRIALS  = 200
OPTUNA_WARM_TRIALS  = 50
N_CV_FOLDS          = 5
EARLY_STOPPING_ROUNDS = 50
BLEND_RMSE_WEIGHT   = 0.65
BLEND_PEAK_WEIGHT   = 0.35

# Date boundaries
DATA_START          = "2021-01-01"
BASELINE_TRAIN_END  = "2024-06-30"
BENCHMARK_QUARTERS  = [
    ("Q3-2024", "2024-07-01", "2024-09-30"),
    ("Q4-2024", "2024-10-01", "2024-12-31"),
    ("Q1-2025", "2025-01-01", "2025-03-31"),
    ("Q2-2025", "2025-04-01", "2025-06-30"),
]
```

### Module 2: `feature_engineering.py`

- Implements `add_extra_features(df)` — identical to existing pipeline
- All rolling/EWM computed on `target.shift(1)`, never on contemporaneous target
- Returns feature-engineered DataFrame with `feature_cols` defined

### Module 3: `splitter.py`

- `chronological_split(df, train_frac=0.70, val_frac=0.15)` — for Phase 1
- `walk_forward_windows(df, quarters)` — yields `(train_df, predict_df)` per quarter
- `inner_split(train_df, val_frac=0.15)` — for per-quarter retrain early stopping

### Module 4: `strategies/full_optuna.py`

- `run_full_optuna(train_df, feature_cols, n_trials=200)` → returns `(model, params)`
- Creates fresh Optuna study, no prior seeding
- Uses 5-fold rolling CV, blended objective, peak weighting, log1p transform
- Saves study object to `artifacts/{quarter}/full_optuna_study.pkl`

### Module 5: `strategies/warm_start.py`

- `run_warm_start(train_df, feature_cols, prior_params, n_trials=50)` → returns `(model, params)`
- Injects `prior_params` as first trial via `study.enqueue_trial()`
- Same CV and objective as full Optuna

### Module 6: `strategies/fixed_param.py`

- `run_fixed_param(train_df, feature_cols, saved_params)` → returns `model`
- Two-stage refit: Stage 1 (early stop calibration) → Stage 2 (full retrain)
- No Optuna calls

### Module 7: `evaluator.py`

- `compute_metrics(y_true, y_pred, peak_threshold=None)` → returns dict
- Computes R², MAE, RMSE, MAPE, Peak MAPE
- Peak threshold defaults to 90th percentile of `y_true` if not passed

### Module 8: `logger.py`

- `log_run(run_dict)` → appends one row to `results/results_log.csv`
- Creates file with headers if first run
- Handles JSON serialization of `best_params`

### Module 9: `visualizer.py`

- `plot_mape_over_time(results_df)`
- `plot_peak_mape_over_time(results_df)`
- `plot_runtime_vs_accuracy(results_df)`
- `plot_peak_mape_heatmap(results_df)`
- `plot_actual_vs_predicted(quarter, strategy, y_true, y_pred, datetimes)`
- `plot_residuals(quarter, strategy, y_true, y_pred)`

### Module 10: `run_benchmark.py` (Main Entry Point)

```
1. Load and engineer features
2. Run Phase 1 (baseline)
3. For each quarter:
     a. Slice train_data and predict_data
     b. Run Strategy A (Full Optuna)
     c. Run Strategy B (Warm Start)
     d. Run Strategy C (Fixed Param)
     e. Evaluate all 3
     f. Log results
     g. Update prior_best_params for next warm start
4. Run Phase 3 analysis and generate all plots
5. Print final recommendation
```

---

## 13. Risk and Edge Cases

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Missing hours in prediction quarter | Medium | Forward-fill weather; flag in notes column |
| Optuna trial crashes mid-run | Low | Wrap in try/except; use Optuna's built-in pruning |
| Peak threshold lands at same value as global threshold | Low | Add minimum peak count check (≥ 50 hours) |
| Gujarat holiday calendar gaps | Medium | Maintain a static Gujarat holiday list for 2021–2025 |
| XGBoost version inconsistency across reruns | Low | Pin version in requirements.txt |
| Strategy B warm-start with bad prior params | Low | Fall back to full Optuna if warm-start MAPE > 1.5× fixed-param |
| Regime shift in a quarter (e.g., heatwave) | Medium | Log residual spike; document in notes; flag for manual review |
| Very low demand quarter (e.g., lockdown echoes) | Low | Monitor MAPE; unusual low-demand quarters can inflate percentage errors |

---

## 14. Expected Outcomes and Hypothesis

### Hypothesis

Based on energy forecasting literature and the properties of Gujarat demand data:

```
Strategy          Expected Avg MAPE    Expected Peak MAPE    Avg Runtime
──────────────────────────────────────────────────────────────────────────
Full Optuna         Best (lowest)        Best (lowest)          ~4 hrs
Warm Start          Near-identical       Near-identical         ~1 hr
Fixed Param         +0.5–1.5% worse      +0.5–2.0% worse        ~10 min
```

**Expected winner:** Warm-Start, because:
- TPE sampler converges quickly when anchored near a good solution
- Hyperparameter optima for Gujarat demand are likely stable quarter-to-quarter
- 50 trials is sufficient to compensate for regime drift without full exploration

**Season where Fixed-Param may fail:** Q3-2024 (Jul–Sep), the monsoon quarter, because:
- Temperature-demand relationship shifts non-linearly during monsoon
- Cooling load partially offsets by reduced solar radiation and lower dry-bulb temps
- Fixed params calibrated on summer data may under-weight humidity interactions

### Benchmark Value

Even if the hypothesis is correct, the benchmark's value is in **quantifying** the tradeoff:
- Exactly how much does warm-start lose vs full Optuna? (0.1%? 0.5%? 2%?)
- Does fixed-param degrade gracefully or catastrophically after 2+ quarters?
- Are there specific demand regimes (peak summer, monsoon onset) where full search is essential?

These numbers are unique to Gujarat data and cannot be assumed from generic ML literature.

---

## 15. Execution Checklist

### Pre-Run Setup

- [ ] Confirm dataset covers Jan 2021 – Jun 2025 with no multi-month gaps
- [ ] Validate datetime parsing: no NaT values after coercion
- [ ] Confirm Gujarat public holiday list is populated for all years
- [ ] Install dependencies: `xgboost`, `optuna`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`
- [ ] Set random seeds: `RANDOM_SEED = 42` in config.py
- [ ] Create all artifact and results directories

### Phase 1 (Baseline)

- [ ] Feature engineering runs without errors on full dataset
- [ ] Zone A test metrics saved to `baseline_metrics.json`
- [ ] Optuna study object saved to `baseline_study.pkl`
- [ ] Best params saved to `baseline_params.json`
- [ ] Confirm Zone A test R² > 0.80 and Peak MAPE < 5% before proceeding

### Phase 2 (Walk-Forward)

- [ ] Q3-2024: All 3 strategies complete, results logged
- [ ] Q4-2024: All 3 strategies complete, results logged
- [ ] Q1-2025: All 3 strategies complete, results logged
- [ ] Q2-2025: All 3 strategies complete, results logged
- [ ] `results_log.csv` has exactly 12 rows (+ header)
- [ ] All 12 model artifacts saved

### Phase 3 (Analysis)

- [ ] All 6 visualization types generated and saved to `/plots`
- [ ] `aggregate_summary.csv` computed from results_log
- [ ] `recommendation.txt` auto-generated by decision logic
- [ ] Final recommendation reviewed and documented

---

## Appendix: Key Formulas

### Blended CV Objective
```
score = 0.65 × (RMSE / mean_demand × 100) + 0.35 × Peak_MAPE%
```

### Peak MAPE
```
peak_idx = y_true >= np.percentile(y_true, 90)
peak_mape = np.mean(|y_true[peak_idx] - y_pred[peak_idx]| / y_true[peak_idx]) × 100
```

### Sample Weights
```
weight_i = PEAK_WEIGHT if demand_i >= 90th_percentile else 1.0
```

### Log Transform (applied to target before fitting)
```
y_train_log = log1p(y_train)       # at fit time
y_pred      = expm1(model.predict) # at inference time
```

### Momentum Features
```
lag_momentum_1_24   = lag_1h  - lag_24h
lag_momentum_24_168 = lag_24h - lag_168h
```

---

*Document version: 1.0 | Last updated: April 2026*  
*For questions on implementation, refer to `src/` module docstrings.*