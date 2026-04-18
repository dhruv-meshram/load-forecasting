# Load Forecasting Poster Plan Artifact

## 1) Template Design Analysis (from Copy of emulator poster.pdf)

### 1.1 Observed Template Structure
- Poster format: 2 pages in the PDF, with page 1 containing the full design and page 2 blank.
- Layout style: full-width vertical flow with strong section bars and two-column result region.
- Visual hierarchy:
  - Large title and subtitle at top center.
  - Introductory narrative block.
  - Methodology block with diagram-heavy content.
  - Mid-page timeline strip.
  - Split results row (left and right result narratives).
  - Full-width conclusion and references at bottom.
- Typography style: bold uppercase section headers, short paragraph blocks, and heavy use of bullet lists.
- Evidence style: visual-first, with compact metric tables and multiple figures grouped per section.

### 1.2 What This Means for Your Poster
- Keep exactly the same storytelling rhythm:
  - Problem context -> methodology -> evidence -> decision -> deployment implications.
- Keep results in two complementary blocks:
  - Left block: model quality comparison (XGBoost vs LSTM).
  - Right block: production readiness and latency/throughput reliability.
- Use one explicit decision panel that answers:
  - "Which model is better?"
  - "Better for which objective: peak load accuracy, overall accuracy, or runtime cost?"

---

## 2) Poster Theme and Central Claim

## 2.1 Main Theme
"Gujarat hourly load forecasting: deciding between XGBoost and LSTM using a peak-sensitive accuracy criterion plus production benchmark evidence."

## 2.2 Main Decision Message to Display
- If the objective is peak-demand reliability and production speed, XGBoost is preferred.
- If the objective is absolute average error only, LSTM is competitive or better on overall MAPE/RMSE in multiple windows.
- Practical deployment decision should use a two-axis rule:
  - Accuracy axis: Peak MAPE (primary), then MAPE/RMSE.
  - Ops axis: p95 latency, throughput, and reliability under concurrency.

---

## 3) Poster Section Plan (Content to Show)

## 3.1 Header Block
Include:
- Title: "Load Forecasting for Gujarat: XGBoost vs LSTM under Accuracy and Production Constraints"
- Team details and dates.
- One-line objective statement:
  - "Compare forecasting quality and serving behavior to choose a model for real-world load operations."

## 3.2 Introduction Block
Include 4 concise bullets:
- Why load forecasting matters (grid stability, peak planning, cost optimization).
- Why comparing only one metric is misleading.
- Why peak-hour performance is operationally critical.
- Why production behavior (latency/throughput/reliability) must be co-optimized with accuracy.

## 3.3 Data + Methodology Block
Include:
- Dataset scope:
  - Range: 2020-01-01 to 2025-06-30 (hourly).
  - Final synchronized benchmark rows: 38374 (non-reserved horizon for quality/benchmark tests).
- Feature pipeline summary:
  - Weather + cyclical time + lag/rolling + interaction features.
- Model setup:
  - XGBoost ONNX (50 features).
  - LSTM ONNX (168 lookback, 30 features).
  - Persistence baseline for calibration.
- Split and fairness protocol:
  - Chronological 70/15/15 for static comparison.
  - Reserved future period used for walk-forward.
  - Same timestamp alignment and no shuffling.
- Decision rule box:
  - Primary metric: Peak MAPE (top 10% demand hours).
  - Secondary metrics: MAPE, RMSE, R2.
  - Production tie-breakers: p95, throughput, error/timeout.

## 3.4 Project Timeline Block
Use a horizontal timeline with these phases:

| Phase | Period | What to show in poster |
|---|---|---|
| Data Collection | 2020-01 to 2025-06 | Hourly demand + weather aggregation for Gujarat |
| Data Cleaning & Feature Engineering | Initial modeling phase | Datetime normalization, lag/rolling, weather interactions |
| Baseline Model Training | Pre-benchmark | XGBoost and LSTM model training + ONNX export |
| Static Model Comparison | Non-reserved horizon | Chronological 70/15/15 quality comparison |
| Production Benchmarking | Non-reserved horizon | Latency, throughput, concurrency, cold start, robustness |
| Walk-Forward Evaluation | 2024-Q3 to 2025-Q2 | Quarter-by-quarter strategy comparison |
| Final Decision Synthesis | Current | Combine peak quality + runtime evidence into deployment recommendation |

Optional date labels for presentation:
- "Data period: 2020-2025"
- "Evaluation runs: 2026 Q1-Q2"

## 3.5 Results Block A (Model Quality: XGBoost vs LSTM)
This is the core scientific comparison block.

Mandatory result table:

| Metric (test, model-comparison) | XGBoost_ONNX | LSTM_ONNX |
|---|---:|---:|
| MAE | 311.349 | 299.151 |
| RMSE | 428.848 | 422.775 |
| MAPE (%) | 1.796 | 1.682 |
| Peak MAPE top10 (%) | 1.666 | 1.994 |
| R2 | 0.9718 | 0.9726 |

Interpretation text to print beside table:
- "LSTM slightly improves overall MAPE/RMSE, but XGBoost has lower Peak MAPE, which is prioritized for high-demand operational windows."

Also include:
- Significance line from report (paired tests / DM test indicate differences are statistically measurable).
- One sentence that explains why peak-oriented ranking is used for grid operations.

## 3.6 Results Block B (Production Behavior)
This block explains operational feasibility.

Mandatory benchmark metrics to show:

| Production Metric | XGBoost_ONNX | LSTM_ONNX |
|---|---:|---:|
| p95 latency (normal profile, ms) | 0.068 | 1.826 |
| Throughput at concurrency 16 (rps) | 223854 | 9443 |
| Cold start total (ms) | 52.94 | 52.49 |
| Error rate under tested traffic (%) | 0 | 0 |
| Timeout rate under tested traffic (%) | 0 | 0 |

Interpretation text:
- "Both models satisfy SLO thresholds in local benchmark runs, but XGBoost has a large latency/throughput advantage."

## 3.7 Walk-Forward Strategy Block
Purpose: show how performance evolves over future quarters and how retraining policy affects peak behavior.

Mandatory table (from walk-forward):

| Strategy | Avg MAPE (%) | Avg Peak MAPE (%) | Avg Runtime (min) |
|---|---:|---:|---:|
| fixed_param | 2.380 | 3.191 | 0.0015 |
| full_optuna | 2.627 | 2.610 | 0.0350 |
| warm_start | 2.627 | 2.610 | 0.0148 |

Mandatory interpretation:
- "Fixed-param is fastest and gives lower average MAPE in this run, but peak errors degrade over later quarters."
- "Warm-start matches full-optuna peak quality with lower retraining runtime, making it the preferred retraining policy under the current decision rule."

## 3.8 Final Decision Block (Most Important Panel)
Show this as a boxed decision matrix:

| Decision Objective | Winner | Why |
|---|---|---|
| Peak-demand accuracy (primary) | XGBoost | Lower Peak MAPE than LSTM in static model comparison |
| Average-error minimization only | LSTM | Lower test MAPE/RMSE in static model comparison |
| Low-latency high-throughput serving | XGBoost | Much lower p95 and much higher throughput |
| Future retraining policy | Warm-start strategy | Near full-optuna quality with lower retraining runtime |

One final sentence (poster headline style):
- "For operational deployment, use XGBoost for inference and warm-start retraining cadence for future adaptation."

## 3.9 Conclusion + Future Scope
Include:
- Conclusion bullet 1: peak-sensitive comparison changes model choice versus plain MAPE-only comparison.
- Conclusion bullet 2: production benchmarks are necessary, not optional, for model selection.
- Conclusion bullet 3: walk-forward testing reveals drift sensitivity and retraining trade-offs.
- Future scope bullet 1: run uncapped full trial budgets for quarterly retraining.
- Future scope bullet 2: add statistical confidence for walk-forward quarter deltas.
- Future scope bullet 3: include cost-per-inference and deployment environment benchmarks (GPU/cloud autoscaling).

---

## 4) Exact Figures to Include in Poster

## 4.1 Must-Include Figures (core story)
From model-comparison:
- Actual vs predicted (test) multi-model overlay.
- Peak MAPE comparison bar chart.
- Residual distribution comparison.
- Runtime vs accuracy scatter.

From benchmark:
- Throughput vs concurrency.
- Latency p95 by traffic profile.
- Batch latency scaling.
- SLO status summary visual (or compact table).

From walk-forward:
- Peak MAPE over quarter (line chart).
- MAPE over quarter (line chart).
- Peak MAPE heatmap by strategy and quarter.
- Runtime vs accuracy tradeoff for retraining strategies.

## 4.2 Optional Figures (space permitting)
- Quarter-wise actual vs predicted overlays (small multiples).
- Residual KDE per walk-forward quarter.
- Drift trigger summary (weekly MAPE windows).

---

## 5) Exact Test Results to Include in Poster

## 5.1 Accuracy/Quality Tests (must include)
- Chronological split quality table (test rows only for readability).
- Peak-sensitive ranking result.
- Rolling backtest trend (5-fold MAPE lines).
- Statistical significance summary row (XGBoost vs LSTM).

## 5.2 Production/Performance Tests (must include)
- Traffic profiles A/B/C/D p95 + throughput + reliability.
- Concurrency sweep.
- Batch scaling.
- Cold start.
- Error/timeout robustness summary.
- SLO pass/fail table.

## 5.3 Walk-Forward Tests (must include)
- 12-run table condensed to quarter x strategy metrics.
- Aggregate strategy table.
- Strategy winner logic and rationale.

## 5.4 Other Results to Include (supporting evidence)
- Reference model summary table (xgb_onnx and lstm_onnx in future horizon alignment).
- Baseline metrics reference line used in walk-forward plots.
- Data coverage and feature count summary.

---

## 6) Poster Assembly Blueprint (Section-by-Section Fill)

Use this exact flow while assembling page 1:

1. Header strip
- Project title, names, institution logos, one-sentence objective.

2. Introduction strip
- Problem importance + why peak-sensitive model selection is needed.

3. Methodology strip
- Data pipeline diagram + model pipeline + decision rule box.

4. Timeline strip
- Data collection to deployment decision phases.

5. Results left panel
- Static quality comparison and statistical tests.

6. Results right panel
- Production benchmark and SLO evidence.

7. Walk-forward panel
- Quarter strategy comparison and retraining recommendation.

8. Final decision box
- "Which model is better?" by objective matrix.

9. Conclusion + future scope
- Short takeaways + next experimental plan.

10. References footer
- Benchmark/model-comparison/walk-forward report links and key citations.

---

## 7) Final Poster Message (ready-to-use wording)

"No single metric is enough for load forecasting model selection. In this Gujarat study, LSTM slightly improves average error, but XGBoost delivers better peak-demand accuracy and dramatically better serving performance. Combining quality, reliability, and walk-forward retraining evidence supports deploying XGBoost with warm-start retraining policy."
