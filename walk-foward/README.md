# Walk-Forward Retraining Pipeline

This folder contains the walk-forward retraining benchmark for Gujarat hourly electricity demand forecasting.

## Entry Point

Run the smoke benchmark:

```powershell
& "d:/project course/.venv/Scripts/python.exe" "d:/project course/load forecasting/walk-foward/run_walk_forward.py" --profile quick
```

Supported profiles:

- `quick` for smoke validation and report generation.
- `balanced` for a larger verification pass.
- `full` for the heaviest benchmark configuration.

## Outputs

- `output/results/results_log.csv`
- `output/results/aggregate_summary.csv`
- `output/results/reference_models_summary.csv`
- `output/walk_forward_report.md`
- `output/plots/`
- `output/artifacts/`

The fixed-param strategy reuses the stored ONNX model in `../models/xgboost_model.onnx`.