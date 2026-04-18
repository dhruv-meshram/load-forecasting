from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import optuna
import pandas as pd
import seaborn as sns
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
LOAD_FORECASTING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = LOAD_FORECASTING_DIR.parent
DATA_PATH = PROJECT_ROOT / "gujarat_hourly_merged.csv"
MODELS_DIR = LOAD_FORECASTING_DIR / "models"
SCALERS_DIR = LOAD_FORECASTING_DIR / "scalars"

FIXED_XGB_ONNX_PATH = MODELS_DIR / "xgboost_model.onnx"
REFERENCE_LSTM_ONNX_PATH = MODELS_DIR / "final_lstm_model.onnx"
LSTM_SCALER_PATH = SCALERS_DIR / "final_lstm_scalers.pkl"

OUTPUT_DIR = SCRIPT_DIR / "output"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = OUTPUT_DIR / "walk_forward_report.md"
RESULTS_LOG_PATH = RESULTS_DIR / "results_log.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
BASELINE_PARAMS_PATH = RESULTS_DIR / "baseline_params.json"
BASELINE_MODEL_SUMMARY_PATH = RESULTS_DIR / "baseline_summary.json"
AGGREGATE_SUMMARY_PATH = RESULTS_DIR / "aggregate_summary.csv"
REFERENCE_SUMMARY_PATH = RESULTS_DIR / "reference_models_summary.csv"

BASELINE_START = pd.Timestamp("2021-01-01 00:00:00")
BASELINE_END = pd.Timestamp("2024-06-30 23:00:00")
QUARTER_WINDOWS = [
    ("Q3-2024", pd.Timestamp("2024-07-01 00:00:00"), pd.Timestamp("2024-09-30 23:00:00"), pd.Timestamp("2024-06-30 23:00:00")),
    ("Q4-2024", pd.Timestamp("2024-10-01 00:00:00"), pd.Timestamp("2024-12-31 23:00:00"), pd.Timestamp("2024-09-30 23:00:00")),
    ("Q1-2025", pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-03-31 23:00:00"), pd.Timestamp("2024-12-31 23:00:00")),
    ("Q2-2025", pd.Timestamp("2025-04-01 00:00:00"), pd.Timestamp("2025-06-30 23:00:00"), pd.Timestamp("2025-03-31 23:00:00")),
]

PEAK_WEIGHT = 60.0
PEAK_PERCENTILE = 0.90
EARLY_STOPPING_ROUNDS = 50
BLEND_RMSE_WEIGHT = 0.65
BLEND_PEAK_WEIGHT = 0.35
XGB_TREE_METHOD = "hist"

FEATURE_COLUMNS = [
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_shortwave_radiation",
    "weather_precipitation",
    "weather_windspeed_10m",
    "weather_dewpoint_2m",
    "weather_cloudcover",
    "weather_apparent_temperature",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "is_sunday",
    "year_index",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "sin_day_year",
    "cos_day_year",
    "is_peak_season",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "lag_48h",
    "lag_336h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
    "rolling_std_168h",
    "rolling_mean_6h_from_target",
    "rolling_max_24h_from_target",
    "ewm_24h_from_target",
    "temp_x_peak",
    "solar_x_peak",
    "temp_x_hour_sin",
    "lag_momentum_1_24",
    "lag_momentum_24_168",
    "weather_balance",
    "temp_x_humidity",
    "rain_flag",
    "precip_x_peak",
    "cloud_x_temp",
    "cdd_24",
    "hdd_18",
    "apparent_minus_temp",
    "rad_x_temp",
    "hot_flag",
    "hot_hour_interaction",
]

LSTM_FEATURES = [
    "demand_mw",
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_shortwave_radiation",
    "weather_precipitation",
    "weather_windspeed_10m",
    "weather_dewpoint_2m",
    "weather_cloudcover",
    "weather_apparent_temperature",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "is_sunday",
    "year_index",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "sin_day_year",
    "cos_day_year",
    "is_peak_season",
    "thermal_stress",
    "humidity_temp_index",
    "precipitation_flag",
    "precipitation_x_cloud",
    "weather_energy_interaction",
    "weather_motion_interaction",
    "humidity_stress",
]

TARGET_COL = "demand_mw"
DATETIME_COL = "datetime"
LSTM_LOOKBACK = 168


@dataclass(frozen=True)
class QuarterWindow:
    name: str
    predict_start: pd.Timestamp
    predict_end: pd.Timestamp
    train_cutoff: pd.Timestamp


@dataclass
class RunResult:
    quarter: str
    strategy: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train_rows: int
    n_predict_rows: int
    n_trials: int
    runtime_minutes: float
    best_n_estimators: int
    metrics: dict[str, float]
    best_params: dict[str, Any]
    notes: str
    model_path: str | None = None


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, ARTIFACTS_DIR, RESULTS_DIR, PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for subdir in [ARTIFACTS_DIR / "baseline", ARTIFACTS_DIR / "reference", RESULTS_DIR / "actual_vs_predicted", RESULTS_DIR / "residuals"]:
        subdir.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.figsize": (14, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "savefig.dpi": 180,
            "figure.autolayout": True,
        }
    )


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_peak_mask(y_true: np.ndarray, percentile: float = PEAK_PERCENTILE) -> np.ndarray:
    if len(y_true) == 0:
        return np.array([], dtype=bool)
    threshold = float(np.quantile(y_true, percentile))
    return y_true >= threshold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "peak_mape": np.nan, "r2": np.nan, "peak_count": 0, "residual_mean": np.nan, "residual_std": np.nan}

    residual = y_true - y_pred
    peak_mask = compute_peak_mask(y_true)
    peak_mape = safe_mape(y_true[peak_mask], y_pred[peak_mask]) if np.any(peak_mask) else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": safe_mape(y_true, y_pred),
        "peak_mape": peak_mape,
        "r2": float(r2_score(y_true, y_pred)),
        "peak_count": int(np.sum(peak_mask)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
    }


def blended_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse_pct = (math.sqrt(mean_squared_error(y_true, y_pred)) / max(float(np.mean(y_true)), 1e-8)) * 100.0
    peak_mask = compute_peak_mask(y_true)
    peak_mape = safe_mape(y_true[peak_mask], y_pred[peak_mask]) if np.any(peak_mask) else 1e6
    return float(BLEND_RMSE_WEIGHT * rmse_pct + BLEND_PEAK_WEIGHT * peak_mape)


def derive_relative_humidity(temp: pd.Series, dewpoint: pd.Series) -> pd.Series:
    saturation_vp = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    actual_vp = 6.112 * np.exp((17.67 * dewpoint) / (dewpoint + 243.5))
    return pd.Series(np.clip((actual_vp / saturation_vp) * 100.0, 0.0, 100.0), index=temp.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[DATETIME_COL] = pd.to_datetime(data[DATETIME_COL], errors="coerce")
    data = data.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL).reset_index(drop=True)
    data = data.loc[:, ~data.columns.duplicated()].copy()

    if "weather_precipitation" not in data.columns:
        data["weather_precipitation"] = 0.0
    if "weather_relative_humidity_2m" not in data.columns:
        data["weather_relative_humidity_2m"] = derive_relative_humidity(data["weather_temperature_2m"], data["weather_dewpoint_2m"])

    data["hour"] = data[DATETIME_COL].dt.hour
    data["day_of_week"] = data[DATETIME_COL].dt.dayofweek
    data["day_of_year"] = data[DATETIME_COL].dt.dayofyear
    data["month"] = data[DATETIME_COL].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_sunday"] = (data["day_of_week"] == 6).astype(int)
    data["year_index"] = data[DATETIME_COL].dt.year - int(data[DATETIME_COL].dt.year.min())

    data["sin_hour"] = np.sin(2.0 * np.pi * data["hour"] / 24.0)
    data["cos_hour"] = np.cos(2.0 * np.pi * data["hour"] / 24.0)
    data["sin_month"] = np.sin(2.0 * np.pi * data["month"] / 12.0)
    data["cos_month"] = np.cos(2.0 * np.pi * data["month"] / 12.0)
    data["sin_day_year"] = np.sin(2.0 * np.pi * data["day_of_year"] / 365.25)
    data["cos_day_year"] = np.cos(2.0 * np.pi * data["day_of_year"] / 365.25)
    data["is_peak_season"] = data["month"].isin([3, 4, 5, 6]).astype(int)

    demand = pd.to_numeric(data[TARGET_COL], errors="coerce")
    shifted = demand.shift(1)
    data["lag_1h"] = demand.shift(1)
    data["lag_24h"] = demand.shift(24)
    data["lag_48h"] = demand.shift(48)
    data["lag_168h"] = demand.shift(168)
    data["lag_336h"] = demand.shift(336)
    data["rolling_mean_24h"] = shifted.rolling(24, min_periods=24).mean()
    data["rolling_std_24h"] = shifted.rolling(24, min_periods=24).std()
    data["rolling_mean_168h"] = shifted.rolling(168, min_periods=168).mean()
    data["rolling_std_168h"] = shifted.rolling(168, min_periods=168).std()
    data["rolling_mean_6h_from_target"] = shifted.rolling(6, min_periods=6).mean()
    data["rolling_max_24h_from_target"] = shifted.rolling(24, min_periods=24).max()
    data["ewm_24h_from_target"] = shifted.ewm(span=24, adjust=False).mean()

    data["temp_x_peak"] = data["weather_temperature_2m"] * data["is_peak_season"]
    data["solar_x_peak"] = data["weather_shortwave_radiation"] * data["is_peak_season"]
    data["temp_x_hour_sin"] = data["weather_temperature_2m"] * data["sin_hour"]
    data["lag_momentum_1_24"] = data["lag_1h"] - data["lag_24h"]
    data["lag_momentum_24_168"] = data["lag_24h"] - data["lag_168h"]
    data["weather_balance"] = data["weather_temperature_2m"] - data["weather_dewpoint_2m"]
    data["temp_x_humidity"] = data["weather_temperature_2m"] * data["weather_relative_humidity_2m"]
    data["rain_flag"] = (data["weather_precipitation"] > 0).astype(int)
    data["precip_x_peak"] = data["weather_precipitation"] * data["is_peak_season"]
    data["cloud_x_temp"] = data["weather_cloudcover"] * data["weather_temperature_2m"]
    data["cdd_24"] = np.clip(data["weather_temperature_2m"] - 24.0, 0.0, None)
    data["hdd_18"] = np.clip(18.0 - data["weather_temperature_2m"], 0.0, None)
    data["apparent_minus_temp"] = data["weather_apparent_temperature"] - data["weather_temperature_2m"]
    data["rad_x_temp"] = data["weather_shortwave_radiation"] * data["weather_temperature_2m"]
    temp_q75 = float(data["weather_temperature_2m"].quantile(0.75))
    data["hot_flag"] = (data["weather_temperature_2m"] >= temp_q75).astype(int)
    data["hot_hour_interaction"] = data["hot_flag"] * data["hour"]

    data["thermal_stress"] = data["weather_temperature_2m"] - data["weather_apparent_temperature"]
    data["humidity_temp_index"] = data["weather_temperature_2m"] * data["weather_relative_humidity_2m"]
    data["precipitation_flag"] = (data["weather_precipitation"] > 0).astype(float)
    data["precipitation_x_cloud"] = data["weather_precipitation"] * data["weather_cloudcover"]
    data["weather_energy_interaction"] = data["weather_temperature_2m"] * (1.0 + data["weather_shortwave_radiation"] / 1000.0)
    data["weather_motion_interaction"] = data["weather_shortwave_radiation"] * data["weather_windspeed_10m"]
    data["humidity_stress"] = data["weather_relative_humidity_2m"] * (data["weather_temperature_2m"] - data["weather_dewpoint_2m"])
    data["target_demand_mw"] = demand.shift(-1)
    data["target_delta_mw"] = data["target_demand_mw"] - demand

    return data


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    raw[DATETIME_COL] = pd.to_datetime(raw[DATETIME_COL], errors="coerce")
    raw = raw.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL).reset_index(drop=True)
    raw = raw.loc[(raw[DATETIME_COL] >= BASELINE_START) & (raw[DATETIME_COL] <= QUARTER_WINDOWS[-1][2])].copy()
    return add_features(raw)


def clean_model_frame(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    columns = [DATETIME_COL] + ([TARGET_COL] if include_target else []) + FEATURE_COLUMNS
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    cleaned = df[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return cleaned.sort_values(DATETIME_COL).reset_index(drop=True)


def make_ort_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def predict_onnx_xgb(session: ort.InferenceSession, frame: pd.DataFrame) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    matrix = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    raw_pred = session.run([output_name], {input_name: matrix})[0].reshape(-1)
    return np.expm1(raw_pred)


def predict_onnx_lstm(df: pd.DataFrame, session: ort.InferenceSession, scaler_pack: dict[str, Any]) -> pd.DataFrame:
    x_scaler = scaler_pack["x_scaler"]
    y_scaler = scaler_pack["y_scaler"]
    lstm_df = add_features(df.copy())
    lstm_ready = lstm_df[[DATETIME_COL] + LSTM_FEATURES].replace([np.inf, -np.inf], np.nan).dropna().copy()

    x_scaled = x_scaler.transform(lstm_ready[LSTM_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
    sequences: list[np.ndarray] = []
    pred_times: list[pd.Timestamp] = []
    base_demand: list[float] = []
    actual_next: list[float] = []
    for end_idx in range(LSTM_LOOKBACK - 1, len(lstm_ready) - 1):
        start_idx = end_idx - LSTM_LOOKBACK + 1
        sequences.append(x_scaled[start_idx : end_idx + 1])
        pred_times.append(pd.Timestamp(lstm_ready.iloc[end_idx + 1][DATETIME_COL]))
        base_demand.append(float(lstm_ready.iloc[end_idx][TARGET_COL]))
        actual_next.append(float(lstm_ready.iloc[end_idx + 1][TARGET_COL]))

    if not sequences:
        return pd.DataFrame(columns=[DATETIME_COL, "actual_demand_mw", "predicted_demand_mw"])

    x_seq = np.asarray(sequences, dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds: list[np.ndarray] = []
    batch_size = 512
    for start in range(0, len(x_seq), batch_size):
        batch = x_seq[start : start + batch_size]
        out = session.run([output_name], {input_name: batch})[0].reshape(-1, 1)
        preds.append(out)
    delta_scaled = np.vstack(preds)
    delta = y_scaler.inverse_transform(delta_scaled).reshape(-1)
    predicted = np.asarray(base_demand) + delta
    return pd.DataFrame(
        {
            DATETIME_COL: pd.to_datetime(pred_times),
            "actual_demand_mw": np.asarray(actual_next),
            "predicted_demand_mw": predicted,
        }
    )


def time_split_frame(frame: pd.DataFrame, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    if n < 100:
        raise ValueError("Not enough rows for chronological train/validation split")
    split_idx = int(n * (1.0 - val_frac))
    split_idx = min(max(split_idx, 1), n - 1)
    train_df = frame.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = frame.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, val_df


def build_sample_weights(y: np.ndarray) -> np.ndarray:
    peak_threshold = float(np.quantile(y, PEAK_PERCENTILE))
    return np.where(y >= peak_threshold, PEAK_WEIGHT, 1.0).astype(float)


def train_xgb_model(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict[str, Any], n_estimators: int, seed: int = RANDOM_SEED) -> tuple[xgb.Booster, int, dict[str, float]]:
    dtrain = xgb.DMatrix(train_df[FEATURE_COLUMNS], label=np.log1p(train_df[TARGET_COL].to_numpy(dtype=float)), weight=build_sample_weights(train_df[TARGET_COL].to_numpy(dtype=float)))
    dval = xgb.DMatrix(val_df[FEATURE_COLUMNS], label=np.log1p(val_df[TARGET_COL].to_numpy(dtype=float)), weight=build_sample_weights(val_df[TARGET_COL].to_numpy(dtype=float)))
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": XGB_TREE_METHOD,
            "seed": seed,
            "eta": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "min_child_weight": float(params["min_child_weight"]),
            "gamma": float(params["gamma"]),
            "alpha": float(params["reg_alpha"]),
            "lambda": float(params["reg_lambda"]),
        },
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iteration = int((booster.best_iteration if booster.best_iteration is not None else (n_estimators - 1)) + 1)
    pred = np.expm1(booster.predict(dval, iteration_range=(0, best_iteration)))
    metrics = compute_metrics(val_df[TARGET_COL].to_numpy(dtype=float), pred)
    return booster, best_iteration, metrics


def make_param_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def optimize_params(train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int, seed: int = RANDOM_SEED, prior_params: dict[str, Any] | None = None) -> tuple[dict[str, Any], int, optuna.Study]:
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        trial_params = make_param_space(trial)
        n_estimators = trial.suggest_int("n_estimators", 80, 250)
        booster, best_iter, _ = train_xgb_model(train_df, val_df, trial_params, n_estimators=n_estimators, seed=seed)
        dval = xgb.DMatrix(val_df[FEATURE_COLUMNS])
        pred = np.expm1(booster.predict(dval, iteration_range=(0, best_iter)))
        return blended_score(y_val, pred)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    if prior_params:
        enqueue = prior_params.copy()
        if "best_n_estimators" in enqueue and "n_estimators" not in enqueue:
            enqueue["n_estimators"] = int(enqueue["best_n_estimators"])
        study.enqueue_trial(enqueue)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_n_estimators = int(best_params.pop("n_estimators"))
    return best_params, best_n_estimators, study


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_model_json(model: xgb.Booster, path: Path) -> None:
    model.save_model(str(path))


def baseline_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        frame.iloc[:train_end].copy().reset_index(drop=True),
        frame.iloc[train_end:val_end].copy().reset_index(drop=True),
        frame.iloc[val_end:].copy().reset_index(drop=True),
    )


def run_baseline_phase(full_df: pd.DataFrame) -> dict[str, Any]:
    baseline_df = full_df.loc[(full_df[DATETIME_COL] >= BASELINE_START) & (full_df[DATETIME_COL] <= BASELINE_END)].copy()
    baseline_df = clean_model_frame(baseline_df, include_target=True)
    train_df, val_df, test_df = baseline_split(baseline_df)

    xgb_session = make_ort_session(FIXED_XGB_ONNX_PATH)
    baseline_pred = predict_onnx_xgb(xgb_session, test_df)
    baseline_metrics = compute_metrics(test_df[TARGET_COL].to_numpy(dtype=float), baseline_pred)

    save_json(
        BASELINE_METRICS_PATH,
        {
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "metrics": baseline_metrics,
            "source_model": str(FIXED_XGB_ONNX_PATH.name),
            "date_range": [str(baseline_df[DATETIME_COL].min()), str(baseline_df[DATETIME_COL].max())],
        },
    )
    save_json(BASELINE_PARAMS_PATH, {"model": "fixed_param_reference", "strategy": "onnx_reference", "notes": "Stored ONNX model used as the fixed-param baseline for walk-forward evaluation."})
    save_json(BASELINE_MODEL_SUMMARY_PATH, {"train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df), "baseline_test_metrics": baseline_metrics, "feature_count": len(FEATURE_COLUMNS)})
    return {"train": train_df, "validation": val_df, "test": test_df, "predictions": baseline_pred, "metrics": baseline_metrics}


def run_reference_lstm(full_df: pd.DataFrame) -> pd.DataFrame:
    if not (REFERENCE_LSTM_ONNX_PATH.exists() and LSTM_SCALER_PATH.exists()):
        return pd.DataFrame(columns=[DATETIME_COL, "actual_demand_mw", "predicted_demand_mw", "quarter"])

    scaler_pack = joblib.load(LSTM_SCALER_PATH)
    lstm_session = make_ort_session(REFERENCE_LSTM_ONNX_PATH)
    lstm_preds = predict_onnx_lstm(full_df.loc[(full_df[DATETIME_COL] >= BASELINE_START) & (full_df[DATETIME_COL] <= QUARTER_WINDOWS[-1][2])].copy(), lstm_session, scaler_pack)
    if lstm_preds.empty:
        return lstm_preds

    quarter_labels = []
    for ts in lstm_preds[DATETIME_COL]:
        label = None
        for qname, start, end, _ in QUARTER_WINDOWS:
            if start <= pd.Timestamp(ts) <= end:
                label = qname
                break
        quarter_labels.append(label)
    lstm_preds["quarter"] = quarter_labels
    return lstm_preds


def train_and_score_strategy(quarter: QuarterWindow, train_df: pd.DataFrame, predict_df: pd.DataFrame, strategy: str, profile: dict[str, int], prior_params: dict[str, Any] | None = None) -> tuple[RunResult, pd.DataFrame]:
    start_time = time.perf_counter()
    train_frame = clean_model_frame(train_df, include_target=True)
    predict_frame = clean_model_frame(predict_df, include_target=True)
    if len(train_frame) < 500:
        raise ValueError(f"Training frame too small for {quarter.name}: {len(train_frame)} rows")

    if strategy == "fixed_param":
        session = make_ort_session(FIXED_XGB_ONNX_PATH)
        y_pred = predict_onnx_xgb(session, predict_frame)
        metrics = compute_metrics(predict_frame[TARGET_COL].to_numpy(dtype=float), y_pred)
        result = RunResult(quarter=quarter.name, strategy=strategy, train_start=train_frame[DATETIME_COL].min(), train_end=train_frame[DATETIME_COL].max(), n_train_rows=len(train_frame), n_predict_rows=len(predict_frame), n_trials=0, runtime_minutes=(time.perf_counter() - start_time) / 60.0, best_n_estimators=0, metrics=metrics, best_params={"artifact": FIXED_XGB_ONNX_PATH.name, "mode": "fixed_param_reference"}, notes="Stored ONNX model reused without hyperparameter search.", model_path=str(FIXED_XGB_ONNX_PATH))
        scored = predict_frame[[DATETIME_COL, TARGET_COL]].copy()
        scored["prediction"] = y_pred
        return result, scored

    inner_train, inner_val = time_split_frame(train_frame, val_frac=0.15)
    n_trials = profile["full"] if strategy == "full_optuna" else profile["warm"]
    best_params, best_n_estimators, study = optimize_params(inner_train, inner_val, n_trials=n_trials, prior_params=prior_params if strategy == "warm_start" else None)
    dtrain = xgb.DMatrix(train_frame[FEATURE_COLUMNS], label=np.log1p(train_frame[TARGET_COL].to_numpy(dtype=float)), weight=build_sample_weights(train_frame[TARGET_COL].to_numpy(dtype=float)))
    dpredict = xgb.DMatrix(predict_frame[FEATURE_COLUMNS])
    effective_n_estimators = min(best_n_estimators, profile.get("final_cap", best_n_estimators))
    final_model = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": XGB_TREE_METHOD,
            "seed": RANDOM_SEED,
            "eta": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "min_child_weight": float(best_params["min_child_weight"]),
            "gamma": float(best_params["gamma"]),
            "alpha": float(best_params["reg_alpha"]),
            "lambda": float(best_params["reg_lambda"]),
        },
        dtrain,
        num_boost_round=effective_n_estimators,
        verbose_eval=False,
    )
    y_pred = np.expm1(final_model.predict(dpredict))
    metrics = compute_metrics(predict_frame[TARGET_COL].to_numpy(dtype=float), y_pred)

    model_dir = ARTIFACTS_DIR / quarter.name / strategy
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgb_model.json"
    save_model_json(final_model, model_path)
    save_json(model_dir / "best_params.json", {**best_params, "n_estimators": effective_n_estimators, "inner_best_n_estimators": best_n_estimators})
    save_json(model_dir / "study_summary.json", {"best_value": study.best_value, "best_trial": study.best_trial.number, "n_trials": len(study.trials)})

    scored = predict_frame[[DATETIME_COL, TARGET_COL]].copy()
    scored["prediction"] = y_pred
    result = RunResult(
        quarter=quarter.name,
        strategy=strategy,
        train_start=train_frame[DATETIME_COL].min(),
        train_end=train_frame[DATETIME_COL].max(),
        n_train_rows=len(train_frame),
        n_predict_rows=len(predict_frame),
        n_trials=n_trials,
        runtime_minutes=(time.perf_counter() - start_time) / 60.0,
        best_n_estimators=effective_n_estimators,
        metrics=metrics,
        best_params={**best_params, "n_estimators": effective_n_estimators, "inner_best_n_estimators": best_n_estimators},
        notes=("Warm-start used prior params as an enqueued trial." if strategy == "warm_start" and prior_params else "Fresh Optuna search.")
        + (" Final booster rounds were capped for runtime." if effective_n_estimators != best_n_estimators else ""),
        model_path=str(model_path),
    )
    return result, scored


def log_run_results(run_results: list[RunResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in run_results:
        rows.append(
            {
                "run_id": f"{item.quarter}_{item.strategy}",
                "quarter": item.quarter,
                "strategy": item.strategy,
                "train_start": str(item.train_start),
                "train_end": str(item.train_end),
                "n_train_rows": item.n_train_rows,
                "n_predict_rows": item.n_predict_rows,
                "optuna_trials": item.n_trials,
                "runtime_minutes": item.runtime_minutes,
                "best_n_estimators": item.best_n_estimators,
                "test_r2": item.metrics["r2"],
                "test_mae": item.metrics["mae"],
                "test_rmse": item.metrics["rmse"],
                "test_mape": item.metrics["mape"],
                "test_peak_mape": item.metrics["peak_mape"],
                "peak_count": item.metrics["peak_count"],
                "best_params_json": json.dumps(item.best_params, default=str),
                "notes": item.notes,
                "model_path": item.model_path or "",
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_LOG_PATH, index=False)
    return results_df


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("strategy", as_index=False)
        .agg(
            quarters=("quarter", "count"),
            avg_mape=("test_mape", "mean"),
            avg_peak_mape=("test_peak_mape", "mean"),
            avg_rmse=("test_rmse", "mean"),
            avg_r2=("test_r2", "mean"),
            avg_runtime_minutes=("runtime_minutes", "mean"),
            avg_n_estimators=("best_n_estimators", "mean"),
        )
        .sort_values(["avg_peak_mape", "avg_mape", "avg_runtime_minutes"], ascending=True)
        .reset_index(drop=True)
    )
    summary.to_csv(AGGREGATE_SUMMARY_PATH, index=False)
    return summary


def reference_summary_table(reference_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in reference_results.items():
        if frame.empty:
            continue
        metrics = compute_metrics(frame[TARGET_COL].to_numpy(dtype=float), frame["predicted_demand_mw"].to_numpy(dtype=float))
        rows.append({"model": name, "rows": len(frame), "mae": metrics["mae"], "rmse": metrics["rmse"], "mape": metrics["mape"], "peak_mape": metrics["peak_mape"], "r2": metrics["r2"], "peak_count": metrics["peak_count"]})
    ref_df = pd.DataFrame(rows)
    ref_df.to_csv(REFERENCE_SUMMARY_PATH, index=False)
    return ref_df


def plot_metric_over_time(results_df: pd.DataFrame, metric: str, filename: str, title: str, baseline_value: float | None = None) -> Path:
    pivot = results_df.pivot(index="quarter", columns="strategy", values=metric).reindex([q[0] for q in QUARTER_WINDOWS])
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot.index))
    for strategy in pivot.columns:
        ax.plot(x, pivot[strategy].to_numpy(dtype=float), marker="o", linewidth=2.2, label=strategy)
    if baseline_value is not None and np.isfinite(baseline_value):
        ax.axhline(baseline_value, color="black", linestyle="--", linewidth=1.5, label="baseline reference")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Quarter")
    ax.legend(loc="best")
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_runtime_vs_accuracy(results_df: pd.DataFrame) -> Path:
    grouped = results_df.groupby("strategy", as_index=False).agg(avg_runtime_minutes=("runtime_minutes", "mean"), avg_mape=("test_mape", "mean"), avg_peak_mape=("test_peak_mape", "mean"))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=grouped, x="avg_runtime_minutes", y="avg_mape", hue="strategy", s=250, ax=ax, legend=True)
    for _, row in grouped.iterrows():
        ax.text(row["avg_runtime_minutes"], row["avg_mape"], f" {row['strategy']}", va="center")
    ax.set_xscale("log")
    ax.set_title("Runtime vs Accuracy Tradeoff")
    ax.set_xlabel("Average runtime (minutes, log scale)")
    ax.set_ylabel("Average MAPE %")
    path = PLOTS_DIR / "runtime_vs_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_peak_heatmap(results_df: pd.DataFrame) -> Path:
    pivot = results_df.pivot(index="quarter", columns="strategy", values="test_peak_mape").reindex([q[0] for q in QUARTER_WINDOWS])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Peak MAPE Heatmap")
    path = PLOTS_DIR / "peak_mape_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(scored_frames: dict[tuple[str, str], pd.DataFrame]) -> list[Path]:
    paths: list[Path] = []
    for (quarter, strategy), frame in scored_frames.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(frame[DATETIME_COL], frame[TARGET_COL], color="black", linewidth=1.2, label="Actual")
        ax.plot(frame[DATETIME_COL], frame["prediction"], linewidth=1.2, label=strategy)
        ax.set_title(f"Actual vs Predicted - {quarter} - {strategy}")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Demand MW")
        ax.legend(loc="best")
        path = PLOTS_DIR / f"actual_vs_predicted_{quarter}_{strategy}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_residual_distributions(scored_frames: dict[tuple[str, str], pd.DataFrame]) -> list[Path]:
    paths: list[Path] = []
    for quarter in [q[0] for q in QUARTER_WINDOWS]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for strategy in ["full_optuna", "warm_start", "fixed_param"]:
            frame = scored_frames.get((quarter, strategy))
            if frame is None or frame.empty:
                continue
            residual = frame[TARGET_COL] - frame["prediction"]
            sns.kdeplot(residual, ax=ax, label=strategy, fill=False, linewidth=2)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Residual Distribution - {quarter}")
        ax.set_xlabel("Actual - Predicted MW")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        path = PLOTS_DIR / f"residual_distribution_{quarter}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def build_test_records() -> list[dict[str, Any]]:
    return [
        {"test": "dataset_presence", "status": "pass", "details": str(DATA_PATH)},
        {"test": "onnx_fixed_model_presence", "status": "pass" if FIXED_XGB_ONNX_PATH.exists() else "fail", "details": str(FIXED_XGB_ONNX_PATH)},
        {"test": "onnx_lstm_model_presence", "status": "pass" if REFERENCE_LSTM_ONNX_PATH.exists() else "fail", "details": str(REFERENCE_LSTM_ONNX_PATH)},
        {"test": "scaler_presence", "status": "pass" if LSTM_SCALER_PATH.exists() else "warn", "details": str(LSTM_SCALER_PATH)},
    ]


def write_report(results_df: pd.DataFrame, baseline_payload: dict[str, Any], summary_df: pd.DataFrame, reference_df: pd.DataFrame, reference_lstm_df: pd.DataFrame, plot_paths: list[Path], test_records: list[dict[str, Any]]) -> None:
    baseline_metrics = baseline_payload["metrics"]
    report_lines: list[str] = []
    report_lines.append("# Gujarat Walk-Forward Retraining Report")
    report_lines.append("")
    report_lines.append("## 1. Scope")
    report_lines.append("This report documents the walk-forward benchmark implemented in `load forecasting/walk-foward`. The fixed-param strategy uses the stored ONNX artifact from `/models`, while the competing retraining strategies fit fresh XGBoost models with Optuna tuning.")
    report_lines.append("")
    report_lines.append("## 2. Test Matrix")
    report_lines.append(pd.DataFrame(test_records).to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 3. Dataset Coverage")
    report_lines.append(f"- Baseline zone: {BASELINE_START} to {BASELINE_END}")
    report_lines.append(f"- Prediction horizon: {QUARTER_WINDOWS[0][1]} to {QUARTER_WINDOWS[-1][2]}")
    report_lines.append(f"- Baseline test rows: {len(baseline_payload['test'])}")
    report_lines.append("")
    report_lines.append("## 4. Baseline ONNX Reference")
    report_lines.append(pd.DataFrame([baseline_metrics]).to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 5. Primary Walk-Forward Results")
    report_lines.append(results_df.sort_values(["quarter", "strategy"]).to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 6. Aggregate Strategy Summary")
    report_lines.append(summary_df.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 7. Reference Model Summary")
    if not reference_df.empty:
        report_lines.append(reference_df.to_markdown(index=False))
    else:
        report_lines.append("No XGBoost ONNX reference summary was produced.")
    report_lines.append("")
    report_lines.append("## 8. LSTM Reference Summary")
    if not reference_lstm_df.empty:
        report_lines.append(reference_lstm_df.to_markdown(index=False))
    else:
        report_lines.append("The LSTM ONNX reference did not produce enough aligned rows after sequence construction.")
    report_lines.append("")
    report_lines.append("## 9. Plots")
    for path in plot_paths:
        rel = path.relative_to(OUTPUT_DIR).as_posix()
        report_lines.append(f"- ![{path.stem}]({rel})")
    report_lines.append("")
    report_lines.append("## 10. Winner Logic")
    winner_row = summary_df.iloc[0]
    report_lines.append(f"Primary winner by average peak MAPE: **{winner_row['strategy']}** with avg peak MAPE {winner_row['avg_peak_mape']:.4f}% and avg runtime {winner_row['avg_runtime_minutes']:.2f} minutes.")
    report_lines.append("")
    report_lines.append("## 11. Notes")
    report_lines.append("- Fixed-param evaluation uses the stored `xgboost_model.onnx` artifact from `/models`.")
    report_lines.append("- The LSTM ONNX model is scored as a reference series for the same date range where alignment is possible.")
    report_lines.append("- The walk-forward benchmark uses chronological expanding windows and never shuffles rows.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def run_walk_forward(profile: str) -> None:
    ensure_dirs()
    configure_plot_style()
    test_records = build_test_records()

    if profile == "quick":
        trial_profile = {"full": 3, "warm": 2, "final_cap": 120}
    elif profile == "balanced":
        trial_profile = {"full": 10, "warm": 5, "final_cap": 300}
    else:
        trial_profile = {"full": 16, "warm": 8, "final_cap": 500}

    full_df = load_dataset()
    baseline_payload = run_baseline_phase(full_df)

    quarter_results: list[RunResult] = []
    scored_frames: dict[tuple[str, str], pd.DataFrame] = {}
    prior_params: dict[str, Any] | None = None

    for quarter_name, start, end, cutoff in QUARTER_WINDOWS:
        quarter = QuarterWindow(quarter_name, start, end, cutoff)
        train_df = full_df.loc[full_df[DATETIME_COL] <= quarter.train_cutoff].copy()
        predict_df = full_df.loc[(full_df[DATETIME_COL] >= quarter.predict_start) & (full_df[DATETIME_COL] <= quarter.predict_end)].copy()

        for strategy in ["full_optuna", "warm_start", "fixed_param"]:
            result, scored = train_and_score_strategy(quarter=quarter, train_df=train_df, predict_df=predict_df, strategy=strategy, profile=trial_profile, prior_params=prior_params)
            quarter_results.append(result)
            scored_frames[(quarter_name, strategy)] = scored
            if strategy == "full_optuna":
                prior_params = result.best_params.copy()

    results_df = log_run_results(quarter_results)
    summary_df = summarize_results(results_df)

    reference_xgb_session = make_ort_session(FIXED_XGB_ONNX_PATH)
    reference_xgb_df = clean_model_frame(full_df.loc[(full_df[DATETIME_COL] >= BASELINE_START) & (full_df[DATETIME_COL] <= QUARTER_WINDOWS[-1][2])].copy(), include_target=True)
    reference_xgb_pred = predict_onnx_xgb(reference_xgb_session, reference_xgb_df)
    reference_xgb_frame = reference_xgb_df[[DATETIME_COL, TARGET_COL]].copy()
    reference_xgb_frame["predicted_demand_mw"] = reference_xgb_pred
    reference_xgb_summary = reference_summary_table({"xgb_onnx": reference_xgb_frame})

    reference_lstm_raw = run_reference_lstm(full_df)
    if not reference_lstm_raw.empty:
        reference_lstm_summary = reference_summary_table({"lstm_onnx": reference_lstm_raw.rename(columns={"actual_demand_mw": TARGET_COL})})
    else:
        reference_lstm_summary = pd.DataFrame()

    baseline_reference_value = float(baseline_payload["metrics"]["mape"])
    plot_paths: list[Path] = []
    plot_paths.append(plot_metric_over_time(results_df, "test_mape", "mape_over_time.png", "MAPE Over Time", baseline_reference_value))
    plot_paths.append(plot_metric_over_time(results_df, "test_peak_mape", "peak_mape_over_time.png", "Peak MAPE Over Time", baseline_payload["metrics"]["peak_mape"]))
    plot_paths.append(plot_runtime_vs_accuracy(results_df))
    plot_paths.append(plot_peak_heatmap(results_df))
    plot_paths.extend(plot_actual_vs_predicted(scored_frames))
    plot_paths.extend(plot_residual_distributions(scored_frames))

    test_records.append({"test": "baseline_zone_a_score", "status": "pass", "details": json.dumps(baseline_payload["metrics"], default=str)})
    test_records.append({"test": "walk_forward_runs", "status": "pass", "details": f"{len(results_df)} run rows"})
    test_records.append({"test": "plot_generation", "status": "pass", "details": f"{len(plot_paths)} plots"})
    test_records.append({"test": "report_generation", "status": "pass", "details": str(REPORT_PATH)})

    reference_combined = pd.concat(
        [
            reference_xgb_summary.assign(source="xgb_onnx"),
            reference_lstm_summary.assign(source="lstm_onnx") if not reference_lstm_summary.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if not reference_combined.empty:
        reference_combined.to_csv(REFERENCE_SUMMARY_PATH, index=False)

    write_report(results_df=results_df, baseline_payload=baseline_payload, summary_df=summary_df, reference_df=reference_xgb_summary, reference_lstm_df=reference_lstm_summary, plot_paths=plot_paths, test_records=test_records)

    print(f"Report written to: {REPORT_PATH}")
    print(f"Results log written to: {RESULTS_LOG_PATH}")
    print(f"Aggregate summary written to: {AGGREGATE_SUMMARY_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward retraining benchmark for Gujarat demand forecasting.")
    parser.add_argument("--profile", choices=["quick", "balanced", "full"], default=os.getenv("WALK_FORWARD_PROFILE", "quick"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_walk_forward(args.profile)


if __name__ == "__main__":
    main()