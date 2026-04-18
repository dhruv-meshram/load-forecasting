# Detailed Plan: Single-Score Model Comparison and Parameter Variation Study

## 1. Goal
Build a rigorous experiment framework where different forecasting models are compared using one unified target-aligned score, then analyzed as their parameters vary.

The final outcome should include:
- A single leaderboard score per run
- Parameter sensitivity curves for each model family
- Cross-family comparison plots using the same score scale
- Clear decision rules for model selection

## 2. Why a Single Score Is Useful
Different models optimize different internal objectives and can look good on one metric while failing on another. A single composite score:
- Makes model ranking easier for decision-making
- Allows one graph to compare many runs
- Helps communicate results to product and dashboard stakeholders

Important constraint: the single score should still preserve your business priorities, especially peak demand accuracy and API-readiness.

## 3. Define the Single Comparison Score

### 3.1 Primary business intent
For your use case, the model must:
- Predict demand accurately overall
- Predict peak demand reliably
- Be deployable with acceptable latency and stability

### 3.2 Composite Score Design
Use a weighted normalized score in the range 0 to 100, where higher is better.

Recommended structure:

Composite Score = 100 x (w1 x Accuracy Block + w2 x Peak Block + w3 x Robustness Block + w4 x Efficiency Block)

With recommended initial weights:
- w1 Accuracy Block = 0.40
- w2 Peak Block = 0.30
- w3 Robustness Block = 0.20  
- w4 Efficiency Block = 0.10

You can tune these weights later if business priorities change.

### 3.3 Block definitions

Accuracy Block:
- Uses normalized MAE, RMSE, MAPE, R2 on test set
- Suggested internal weights:
  - MAE: 0.25
  - RMSE: 0.25
  - MAPE: 0.25
  - R2: 0.25

Peak Block:
- Uses peak MAE and peak MAPE on top 10 percent demand hours
- Suggested internal weights:
  - Peak MAPE: 0.70
  - Peak MAE: 0.30

Robustness Block:
- Uses rolling backtest stability
- Metrics:
  - Mean fold score
  - Worst fold score
  - Std deviation across folds

Efficiency Block:
- Uses deployment-facing metrics
- Metrics:
  - p95 inference latency
  - Throughput (req/s)
  - Training time for retraining window

### 3.4 Normalization method
Convert each metric to 0 to 1 before combining.

For error metrics where lower is better:
- norm = (max_val - value) / (max_val - min_val)

For quality metrics where higher is better:
- norm = (value - min_val) / (max_val - min_val)

Clip to [0, 1] and handle division by zero if all values are equal.

Use the same normalization basis across all runs in the same experiment round.

## 4. Models to Include in the Study
At minimum:
- XGBoost
- LSTM
- RNN baseline
- Naive persistence baseline

Optional additions:
- LightGBM
- CatBoost
- Linear regression baseline

All models must use:
- Same cleaned dataset
- Same target variable
- Same chronological split policy
- Same leakage rules

## 5. Data Split and Evaluation Protocol

### 5.1 Fixed split protocol
Use one fixed chronological train/validation/test protocol for all models.

### 5.2 Rolling backtest protocol
Also run rolling-origin folds for robustness.

### 5.3 Reproducibility controls
For each run store:
- Random seed
- Data split boundaries
- Feature list version
- Code version hash or notebook version tag

## 6. Parameter Variation Strategy
Use controlled sweeps per family rather than arbitrary random trials only.

### 6.1 XGBoost variation axes
Primary parameters:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- reg_alpha
- reg_lambda
- gamma

Planned variation method:
- Stage A coarse sweep with broad ranges
- Stage B focused sweep around top 20 percent runs
- Stage C local sensitivity around best run

Recommended coarse ranges:
- n_estimators: 300 to 5000
- max_depth: 2 to 12
- learning_rate: 0.005 to 0.2 (log scale)
- subsample: 0.6 to 1.0
- colsample_bytree: 0.5 to 1.0
- min_child_weight: 1 to 20
- reg_alpha: 1e-8 to 10
- reg_lambda: 1e-4 to 100
- gamma: 0 to 10

### 6.2 LSTM variation axes
Primary parameters:
- sequence length
- hidden size
- number of layers
- dropout
- learning rate
- batch size
- optimizer type
- gradient clipping
- epochs and early stopping patience

Recommended coarse ranges:
- sequence length: 24, 48, 72, 168
- hidden size: 32, 64, 128, 256
- layers: 1 to 4
- dropout: 0.0 to 0.5
- learning rate: 1e-4 to 1e-2
- batch size: 32, 64, 128, 256

### 6.3 RNN variation axes
Same as LSTM with emphasis on:
- hidden size
- sequence length
- recurrent nonlinearity options if available
- dropout and gradient clipping (important for stability)

### 6.4 Fairness controls during variation
To keep comparison fair:
- Use equal tuning budget by run count or wall-clock budget
- Use same early stopping policy style
- Use same scoring function for model selection
- Use same features or equivalent information content

## 7. Experiment Matrix Design
Create a structured run matrix:

Phase 1 Baseline:
- 1 baseline config per model
- Validate data and scoring pipeline

Phase 2 Coarse Sweep:
- 50 to 150 runs per model family depending on compute budget

Phase 3 Focused Sweep:
- Top region zoom-in with 20 to 60 runs per family

Phase 4 Stability Check:
- Re-run top 5 configs with multiple seeds

Phase 5 Operational Check:
- Latency and throughput test on top 3 configs per family

## 8. Graphs to Build

### 8.1 Core comparison graph
Graph A: Composite Score by Run
- X-axis: run id or timestamp
- Y-axis: composite score
- Color: model family
- Marker: best per family

### 8.2 Parameter sensitivity graphs
Graph B series: Score vs parameter for each model family
- One plot per parameter
- X-axis: parameter value
- Y-axis: composite score
- Optional smoothing line

Graph C: 2D heatmaps for top interacting parameters
- Example XGBoost: max_depth vs learning_rate
- Example LSTM: sequence_length vs hidden_size
- Color: composite score

### 8.3 Pareto graph
Graph D: Accuracy vs Latency Pareto frontier
- X-axis: p95 latency
- Y-axis: composite accuracy-only block or test MAPE inverse
- Color: model family

### 8.4 Peak quality graph
Graph E: Peak MAPE vs overall MAPE
- Helps identify models that are good on average but weak at peaks

### 8.5 Robustness graph
Graph F: Mean fold score with error bars (std across folds)

## 9. Logging and Tracking Requirements
For every run log the following fields:
- model_family
- model_config_id
- full parameter dict
- train/val/test metrics
- peak metrics
- fold metrics
- latency metrics
- throughput metrics
- training time
- inference batch size
- seed
- composite score
- block scores

Store in one tabular file:
- model_variation_results.csv

Optionally also store in experiment tracker:
- MLflow or equivalent

## 10. Statistical Validation
After finding top candidates:
- Run paired tests on per-timestamp errors (model A vs model B)
- Use bootstrap confidence intervals for final composite score and key metrics
- Check if winner remains winner across seed variations

This avoids choosing a model by random fluctuation.

## 11. Decision Rules
Use explicit selection gates:

Gate 1 Quality:
- Composite score above predefined threshold

Gate 2 Peak reliability:
- Peak MAPE below business threshold

Gate 3 Latency:
- p95 latency below dashboard SLO

Gate 4 Stability:
- No severe fold instability

Gate 5 Retraining practicality:
- Retraining completes in allowed operational window

If multiple models pass all gates, choose the one with:
- Higher composite score
- Lower p95 latency
- Lower retraining cost

## 12. Implementation Steps (Execution Plan)

Step 1: Finalize composite scoring formula and weights with stakeholders.

Step 2: Build a shared evaluator function that:
- Takes predictions, actuals, fold outputs, and latency stats
- Returns all block scores and final composite score

Step 3: Implement model-specific parameter sweep scripts or notebook cells.

Step 4: Run baseline configs for all model families and validate scoring.

Step 5: Run coarse sweeps and collect all run metadata.

Step 6: Run focused sweeps near best parameter regions.

Step 7: Generate comparison and sensitivity plots.

Step 8: Run stability checks with multiple seeds for top configs.

Step 9: Run operational latency/throughput checks on top candidates.

Step 10: Produce final leaderboard and recommendation.

## 13. Deliverables
The study should produce:
- One consolidated results table
- One composite score leaderboard
- Parameter sensitivity charts per model family
- Pareto frontier chart (accuracy vs latency)
- Peak performance comparison chart
- Final recommendation note for dashboard deployment

## 14. Risks and Mitigation
Risk: Composite score hides important metric-level failures.
Mitigation: Always show both composite score and metric breakdown.

Risk: One family gets more tuning effort.
Mitigation: Enforce equal run budget or equal wall-clock budget.

Risk: Overfitting to validation split.
Mitigation: Use rolling backtests and final untouched test evaluation.

Risk: Latency measured in unrealistic environment.
Mitigation: Benchmark in production-like environment.

## 15. Suggested Next Action
Create a new benchmarking notebook section or script that:
- Implements the composite scoring function
- Runs controlled parameter sweeps for XGBoost and LSTM first
- Saves outputs to a single results table
- Auto-generates the key graphs listed above
