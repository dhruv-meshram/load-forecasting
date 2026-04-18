from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "gujarat_hourly_merged.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SCALERS_DIR = PROJECT_ROOT / "scalars"
OUTPUT_DIR = PROJECT_ROOT / "model-comparison"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.onnx"
LSTM_MODEL_PATH = MODELS_DIR / "final_lstm_model.onnx"
LSTM_SCALER_PATH = SCALERS_DIR / "final_lstm_scalers.pkl"

TARGET_COL = "demand_mw"
DATETIME_COL = "datetime"

# Reserved for future walk-forward work. We exclude this range from this implementation.
RESERVED_START = pd.Timestamp("2024-06-01 00:00:00")
RESERVED_END = pd.Timestamp("2025-06-30 23:59:59")

# Evaluation horizon for this comparison run.
EVAL_START = pd.Timestamp("2020-01-01 00:00:00")
EVAL_END = RESERVED_START - pd.Timedelta(hours=1)

LSTM_LOOKBACK = 168
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


XGB_FEATURES = [
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


@dataclass
class ModelRuntime:
    model: str
    avg_runtime_sec: float
    n_samples: int
    runtime_per_1k_samples_sec: float
    model_file_mb: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def make_ort_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def validate_inputs() -> None:
    required = [DATA_PATH, XGB_MODEL_PATH, LSTM_MODEL_PATH, LSTM_SCALER_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    rng = max(float(np.max(y_true) - np.min(y_true)), eps)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse / rng)


def classify_season(month: int) -> str:
    if month in (3, 4, 5, 6):
        return "summer"
    if month in (7, 8, 9):
        return "monsoon"
    if month == 10:
        return "post_monsoon"
    return "winter"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_q10: float, peak_q5: float) -> dict:
    residual = y_true - y_pred
    abs_err = np.abs(residual)

    peak10_mask = y_true >= peak_q10
    peak5_mask = y_true >= peak_q5
    low20_mask = y_true <= np.quantile(y_true, 0.2)

    peak_mape = safe_mape(y_true[peak10_mask], y_pred[peak10_mask]) if np.any(peak10_mask) else np.nan
    peak_mae = float(np.mean(abs_err[peak10_mask])) if np.any(peak10_mask) else np.nan
    top5_mape = safe_mape(y_true[peak5_mask], y_pred[peak5_mask]) if np.any(peak5_mask) else np.nan

    peak_under_rate = float(np.mean(y_pred[peak10_mask] < y_true[peak10_mask]) * 100.0) if np.any(peak10_mask) else np.nan
    low_over_rate = float(np.mean(y_pred[low20_mask] > y_true[low20_mask]) * 100.0) if np.any(low20_mask) else np.nan

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": safe_mape(y_true, y_pred),
        "smape_pct": safe_smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "nrmse": nrmse(y_true, y_pred),
        "peak_mape_pct_top10": peak_mape,
        "peak_mae_top10": peak_mae,
        "top5_mape_pct": top5_mape,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
        "bias_pct": float((np.mean(y_pred - y_true) / np.maximum(np.mean(y_true), 1e-8)) * 100.0),
        "peak_underprediction_rate_pct": peak_under_rate,
        "low_overprediction_rate_pct": low_over_rate,
        "pearson_corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman_rank_corr": float(stats.spearmanr(y_true, y_pred, nan_policy="omit").statistic),
        "peak_count_top10": int(np.sum(peak10_mask)),
        "peak_count_top5": int(np.sum(peak5_mask)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        scores[i] = metric_fn(y_true[sample_idx], y_pred[sample_idx])
    lower = float(np.quantile(scores, alpha / 2))
    upper = float(np.quantile(scores, 1 - alpha / 2))
    return lower, upper


def dm_test(loss_1: np.ndarray, loss_2: np.ndarray, h: int = 1) -> dict:
    d = np.asarray(loss_1) - np.asarray(loss_2)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan}

    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)

    # Newey-West style autocovariance correction.
    max_lag = min(h, n - 1)
    gamma_sum = 0.0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[:-lag], d[lag:], ddof=1)[0, 1]
        gamma_sum += 2.0 * (1.0 - lag / (max_lag + 1)) * cov

    var_d = (gamma0 + gamma_sum) / n
    if var_d <= 0 or not np.isfinite(var_d):
        return {"dm_stat": np.nan, "p_value": np.nan}

    dm_stat = d_bar / math.sqrt(var_d)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return {"dm_stat": float(dm_stat), "p_value": p_value}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values(DATETIME_COL).reset_index(drop=True)

    if "weather_relative_humidity_2m" not in data.columns:
        temp = data["weather_temperature_2m"]
        dewpoint = data["weather_dewpoint_2m"]
        saturation_vp = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
        actual_vp = 6.112 * np.exp((17.67 * dewpoint) / (dewpoint + 243.5))
        data["weather_relative_humidity_2m"] = np.clip((actual_vp / saturation_vp) * 100.0, 0.0, 100.0)

    if "weather_precipitation" not in data.columns:
        data["weather_precipitation"] = 0.0

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

    shifted = data[TARGET_COL].shift(1)
    data["lag_48h"] = data[TARGET_COL].shift(48)
    data["lag_336h"] = data[TARGET_COL].shift(336)
    data["rolling_mean_6h_from_target"] = shifted.rolling(6, min_periods=6).mean()
    data["rolling_max_24h_from_target"] = shifted.rolling(24, min_periods=24).max()
    data["ewm_24h_from_target"] = shifted.ewm(span=24, adjust=False).mean()

    q75_temp = data["weather_temperature_2m"].quantile(0.75)
    data["hot_flag"] = (data["weather_temperature_2m"] >= q75_temp).astype(int)
    data["hot_hour_interaction"] = data["hot_flag"] * data["hour"]

    # LSTM engineered inputs from its training notebook.
    data["thermal_stress"] = data["weather_temperature_2m"] - data["weather_apparent_temperature"]
    data["humidity_temp_index"] = data["weather_temperature_2m"] * data["weather_relative_humidity_2m"]
    data["precipitation_flag"] = (data["weather_precipitation"] > 0).astype(float)
    data["precipitation_x_cloud"] = data["weather_precipitation"] * data["weather_cloudcover"]
    data["weather_energy_interaction"] = data["weather_temperature_2m"] * (1.0 + data["weather_shortwave_radiation"] / 1000.0)
    data["weather_motion_interaction"] = data["weather_shortwave_radiation"] * data["weather_windspeed_10m"]
    data["humidity_stress"] = data["weather_relative_humidity_2m"] * (
        data["weather_temperature_2m"] - data["weather_dewpoint_2m"]
    )

    # LSTM predicts next-step delta; we reconstruct demand by adding current demand.
    data["target_demand_mw"] = data[TARGET_COL].shift(-1)
    data["target_delta_mw"] = data["target_demand_mw"] - data[TARGET_COL]

    return data


def run_xgb_inference(df: pd.DataFrame) -> pd.DataFrame:
    needed = XGB_FEATURES + [DATETIME_COL, TARGET_COL]
    xgb_df = df[needed].dropna().copy().reset_index(drop=True)

    sess = make_ort_session(XGB_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    x_mat = xgb_df[XGB_FEATURES].to_numpy(dtype=np.float32)
    pred_log = sess.run([output_name], {input_name: x_mat})[0].reshape(-1)
    # The training pipeline used log1p on target, so ONNX outputs must be mapped back.
    pred = np.expm1(pred_log)

    out = xgb_df[[DATETIME_COL, TARGET_COL, "lag_1h"]].copy()
    out["xgb_pred"] = pred
    return out


def run_lstm_inference(df: pd.DataFrame) -> pd.DataFrame:
    scaler_pack = joblib.load(LSTM_SCALER_PATH)
    x_scaler = scaler_pack["x_scaler"]
    y_scaler = scaler_pack["y_scaler"]

    needed = LSTM_FEATURES + [DATETIME_COL, "target_delta_mw", "target_demand_mw"]
    lstm_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy().reset_index(drop=True)

    if len(lstm_df) < LSTM_LOOKBACK + 5:
        raise ValueError("Not enough rows for LSTM lookback after cleaning.")

    x_scaled = x_scaler.transform(lstm_df[LSTM_FEATURES]).astype(np.float32)

    sequences = []
    pred_times = []
    base_demand = []
    actual_next = []

    for end_idx in range(LSTM_LOOKBACK - 1, len(lstm_df) - 1):
        start_idx = end_idx - LSTM_LOOKBACK + 1
        sequences.append(x_scaled[start_idx : end_idx + 1])
        pred_times.append(lstm_df.iloc[end_idx + 1][DATETIME_COL])
        base_demand.append(float(lstm_df.iloc[end_idx][TARGET_COL]))
        actual_next.append(float(lstm_df.iloc[end_idx]["target_demand_mw"]))

    x_seq = np.asarray(sequences, dtype=np.float32)

    sess = make_ort_session(LSTM_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    preds = []
    batch_size = 512
    for start in range(0, len(x_seq), batch_size):
        batch = x_seq[start : start + batch_size]
        out = sess.run([output_name], {input_name: batch})[0].reshape(-1, 1)
        preds.append(out)

    y_delta_scaled = np.vstack(preds)
    y_delta = y_scaler.inverse_transform(y_delta_scaled).reshape(-1)

    y_pred = np.asarray(base_demand) + y_delta

    out = pd.DataFrame(
        {
            DATETIME_COL: pd.to_datetime(pred_times),
            "actual_demand_mw": np.asarray(actual_next),
            "lstm_pred": y_pred,
        }
    )
    return out


def run_persistence_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[DATETIME_COL, TARGET_COL, "lag_1h"]].copy()
    out["baseline_pred"] = out["lag_1h"]
    return out.dropna(subset=["baseline_pred"]).reset_index(drop=True)


def chronological_split(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = eval_df.sort_values(DATETIME_COL).reset_index(drop=True).copy()
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "validation"
    df["split"] = split
    return df


def compute_metric_tables(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    window_records = []
    bias_records = []

    for split_name, split_df in eval_df.groupby("split"):
        y_true = split_df[TARGET_COL].to_numpy()
        q10 = float(np.quantile(y_true, 0.9))
        q5 = float(np.quantile(y_true, 0.95))

        for model_col, model_name in [
            ("xgb_pred", "XGBoost_ONNX"),
            ("lstm_pred", "LSTM_ONNX"),
            ("baseline_pred", "Persistence_lag1h"),
        ]:
            y_pred = split_df[model_col].to_numpy()
            metrics = compute_metrics(y_true, y_pred, q10, q5)
            rec = {
                "split": split_name,
                "model": model_name,
                "n_samples": len(split_df),
                **metrics,
            }
            records.append(rec)

            temp = split_df[[DATETIME_COL, TARGET_COL, model_col]].copy()
            temp["year"] = temp[DATETIME_COL].dt.year
            temp["month"] = temp[DATETIME_COL].dt.month
            temp["season"] = temp["month"].apply(classify_season)
            temp["is_weekend_slice"] = temp[DATETIME_COL].dt.dayofweek >= 5
            temp["day_night"] = np.where(temp[DATETIME_COL].dt.hour.between(7, 20), "daytime", "nighttime")

            for year, g in temp.groupby("year"):
                window_records.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "slice_type": "year",
                        "slice_value": str(year),
                        "mae": float(mean_absolute_error(g[TARGET_COL], g[model_col])),
                        "rmse": float(math.sqrt(mean_squared_error(g[TARGET_COL], g[model_col]))),
                        "mape_pct": safe_mape(g[TARGET_COL].to_numpy(), g[model_col].to_numpy()),
                    }
                )

            for season, g in temp.groupby("season"):
                window_records.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "slice_type": "season",
                        "slice_value": season,
                        "mae": float(mean_absolute_error(g[TARGET_COL], g[model_col])),
                        "rmse": float(math.sqrt(mean_squared_error(g[TARGET_COL], g[model_col]))),
                        "mape_pct": safe_mape(g[TARGET_COL].to_numpy(), g[model_col].to_numpy()),
                    }
                )

            for wk_label, g in temp.groupby("is_weekend_slice"):
                window_records.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "slice_type": "weekday_weekend",
                        "slice_value": "weekend" if wk_label else "weekday",
                        "mae": float(mean_absolute_error(g[TARGET_COL], g[model_col])),
                        "rmse": float(math.sqrt(mean_squared_error(g[TARGET_COL], g[model_col]))),
                        "mape_pct": safe_mape(g[TARGET_COL].to_numpy(), g[model_col].to_numpy()),
                    }
                )

            for dn_label, g in temp.groupby("day_night"):
                window_records.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "slice_type": "day_night",
                        "slice_value": dn_label,
                        "mae": float(mean_absolute_error(g[TARGET_COL], g[model_col])),
                        "rmse": float(math.sqrt(mean_squared_error(g[TARGET_COL], g[model_col]))),
                        "mape_pct": safe_mape(g[TARGET_COL].to_numpy(), g[model_col].to_numpy()),
                    }
                )

            residual = split_df[TARGET_COL] - split_df[model_col]
            bias_records.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "residual_mean": float(np.mean(residual)),
                    "residual_std": float(np.std(residual)),
                    "mean_signed_error": float(np.mean(split_df[model_col] - split_df[TARGET_COL])),
                    "bias_pct": float(np.mean((split_df[model_col] - split_df[TARGET_COL]) / np.maximum(split_df[TARGET_COL], 1e-8)) * 100.0),
                }
            )

    main_df = pd.DataFrame(records)
    window_df = pd.DataFrame(window_records)
    bias_df = pd.DataFrame(bias_records)
    return main_df, window_df, bias_df


def rolling_backtest(eval_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    df = eval_df.sort_values(DATETIME_COL).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold_idx, (_, test_idx) in enumerate(tscv.split(df), start=1):
        fold = df.iloc[test_idx]
        y_true = fold[TARGET_COL].to_numpy()
        q10 = float(np.quantile(y_true, 0.9))
        q5 = float(np.quantile(y_true, 0.95))

        for col, name in [
            ("xgb_pred", "XGBoost_ONNX"),
            ("lstm_pred", "LSTM_ONNX"),
            ("baseline_pred", "Persistence_lag1h"),
        ]:
            met = compute_metrics(y_true, fold[col].to_numpy(), q10, q5)
            rows.append(
                {
                    "fold": fold_idx,
                    "model": name,
                    "start": fold[DATETIME_COL].min(),
                    "end": fold[DATETIME_COL].max(),
                    "mae": met["mae"],
                    "rmse": met["rmse"],
                    "mape_pct": met["mape_pct"],
                    "peak_mape_pct_top10": met["peak_mape_pct_top10"],
                    "r2": met["r2"],
                }
            )

    return pd.DataFrame(rows)


def significance_tests(test_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("xgb_pred", "lstm_pred", "XGBoost_ONNX", "LSTM_ONNX"),
        ("xgb_pred", "baseline_pred", "XGBoost_ONNX", "Persistence_lag1h"),
        ("lstm_pred", "baseline_pred", "LSTM_ONNX", "Persistence_lag1h"),
    ]

    y_true = test_df[TARGET_COL].to_numpy()
    rows = []

    for a_col, b_col, a_name, b_name in pairs:
        err_a = np.abs(y_true - test_df[a_col].to_numpy())
        err_b = np.abs(y_true - test_df[b_col].to_numpy())

        t_stat, t_p = stats.ttest_rel(err_a, err_b, nan_policy="omit")
        w_stat, w_p = stats.wilcoxon(err_a, err_b, zero_method="wilcox", alternative="two-sided", mode="approx")

        dm = dm_test(err_a, err_b, h=24)

        rows.append(
            {
                "model_a": a_name,
                "model_b": b_name,
                "mean_abs_error_a": float(np.mean(err_a)),
                "mean_abs_error_b": float(np.mean(err_b)),
                "paired_t_stat": float(t_stat),
                "paired_t_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "dm_stat_abs_error": dm["dm_stat"],
                "dm_p_abs_error": dm["p_value"],
            }
        )

    return pd.DataFrame(rows)


def bootstrap_table(test_df: pd.DataFrame) -> pd.DataFrame:
    y_true = test_df[TARGET_COL].to_numpy()
    rows = []

    for col, name in [
        ("xgb_pred", "XGBoost_ONNX"),
        ("lstm_pred", "LSTM_ONNX"),
        ("baseline_pred", "Persistence_lag1h"),
    ]:
        y_pred = test_df[col].to_numpy()

        mae_l, mae_u = bootstrap_ci(y_true, y_pred, lambda a, b: float(mean_absolute_error(a, b)))
        rmse_l, rmse_u = bootstrap_ci(y_true, y_pred, lambda a, b: float(math.sqrt(mean_squared_error(a, b))))
        mape_l, mape_u = bootstrap_ci(y_true, y_pred, safe_mape)

        rows.append(
            {
                "model": name,
                "mae_ci_low": mae_l,
                "mae_ci_high": mae_u,
                "rmse_ci_low": rmse_l,
                "rmse_ci_high": rmse_u,
                "mape_ci_low": mape_l,
                "mape_ci_high": mape_u,
            }
        )

    return pd.DataFrame(rows)


def measure_inference_runtime(eval_df: pd.DataFrame, x_seq_test: np.ndarray) -> pd.DataFrame:
    test_df = eval_df[eval_df["split"] == "test"].copy()

    xgb_sess = make_ort_session(XGB_MODEL_PATH)
    xgb_input_name = xgb_sess.get_inputs()[0].name
    xgb_output_name = xgb_sess.get_outputs()[0].name

    lstm_sess = make_ort_session(LSTM_MODEL_PATH)
    lstm_input_name = lstm_sess.get_inputs()[0].name
    lstm_output_name = lstm_sess.get_outputs()[0].name

    xgb_mat = test_df[XGB_FEATURES].to_numpy(dtype=np.float32)

    xgb_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = xgb_sess.run([xgb_output_name], {xgb_input_name: xgb_mat})[0]
        xgb_times.append(time.perf_counter() - t0)

    # Limit runtime benchmarking sample size to keep comparison stable and memory-safe.
    lstm_bench = x_seq_test[: min(len(x_seq_test), 4000)]

    lstm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        # Benchmark with chunking to mirror production-friendly inference behavior.
        for start in range(0, len(lstm_bench), 512):
            batch = lstm_bench[start : start + 512]
            _ = lstm_sess.run([lstm_output_name], {lstm_input_name: batch})[0]
        lstm_times.append(time.perf_counter() - t0)

    run_rows = [
        ModelRuntime(
            model="XGBoost_ONNX",
            avg_runtime_sec=float(np.mean(xgb_times)),
            n_samples=len(xgb_mat),
            runtime_per_1k_samples_sec=float(np.mean(xgb_times) * 1000.0 / max(len(xgb_mat), 1)),
            model_file_mb=float(XGB_MODEL_PATH.stat().st_size / (1024**2)),
        ),
        ModelRuntime(
            model="LSTM_ONNX",
            avg_runtime_sec=float(np.mean(lstm_times)),
            n_samples=len(lstm_bench),
            runtime_per_1k_samples_sec=float(np.mean(lstm_times) * 1000.0 / max(len(lstm_bench), 1)),
            model_file_mb=float((LSTM_MODEL_PATH.stat().st_size + (LSTM_MODEL_PATH.with_suffix(".onnx.data")).stat().st_size) / (1024**2)),
        ),
    ]

    return pd.DataFrame([r.__dict__ for r in run_rows])


def make_plots(eval_df: pd.DataFrame, metrics_df: pd.DataFrame, rolling_df: pd.DataFrame, runtime_df: pd.DataFrame) -> list[Path]:
    sns.set_theme(style="whitegrid")
    saved = []

    test_df = eval_df[eval_df["split"] == "test"].copy().sort_values(DATETIME_COL)
    plot_cols = [
        ("xgb_pred", "XGBoost", "#1b9e77"),
        ("lstm_pred", "LSTM", "#d95f02"),
        ("baseline_pred", "Persistence", "#7570b3"),
    ]

    # 1) Actual vs predicted
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(test_df[DATETIME_COL], test_df[TARGET_COL], label="Actual", color="black", linewidth=1.4)
    for col, name, color in plot_cols:
        ax.plot(test_df[DATETIME_COL], test_df[col], label=name, linewidth=1.1, alpha=0.85, color=color)
    ax.set_title("Test Window: Actual vs Predicted (All Models)")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper left")
    p = PLOTS_DIR / "01_actual_vs_predicted_test.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 2) Residual distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, name, color in plot_cols:
        residual = test_df[TARGET_COL] - test_df[col]
        sns.kdeplot(residual, ax=ax, label=name, color=color, fill=False, linewidth=1.6)
    ax.set_title("Residual Distribution (Test)")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.legend()
    p = PLOTS_DIR / "02_residual_distribution_test.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 3) Residuals over time
    fig, ax = plt.subplots(figsize=(16, 5))
    for col, name, color in plot_cols:
        residual = test_df[TARGET_COL] - test_df[col]
        ax.plot(test_df[DATETIME_COL], residual, label=name, alpha=0.85, linewidth=1.0, color=color)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title("Residuals Over Time (Test)")
    ax.legend()
    p = PLOTS_DIR / "03_residuals_over_time_test.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 4) Error by hour of day
    temp = test_df.copy()
    temp["hour"] = temp[DATETIME_COL].dt.hour
    hour_rows = []
    for col, name, _ in plot_cols:
        g = temp.groupby("hour").apply(lambda d: mean_absolute_error(d[TARGET_COL], d[col]))
        for h, mae in g.items():
            hour_rows.append({"hour": h, "model": name, "mae": mae})
    hour_df = pd.DataFrame(hour_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=hour_df, x="hour", y="mae", hue="model", marker="o", ax=ax)
    ax.set_title("MAE by Hour of Day (Test)")
    p = PLOTS_DIR / "04_mae_by_hour_test.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 5) Error by month
    temp["month"] = temp[DATETIME_COL].dt.to_period("M").astype(str)
    month_rows = []
    for col, name, _ in plot_cols:
        for month, g in temp.groupby("month"):
            month_rows.append(
                {
                    "month": month,
                    "model": name,
                    "mape_pct": safe_mape(g[TARGET_COL].to_numpy(), g[col].to_numpy()),
                }
            )
    month_df = pd.DataFrame(month_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=month_df, x="month", y="mape_pct", hue="model", marker="o", ax=ax)
    ax.set_title("MAPE by Month (Test)")
    ax.tick_params(axis="x", rotation=45)
    p = PLOTS_DIR / "05_mape_by_month_test.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 6) Predicted vs actual scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (col, name, color) in zip(axes, plot_cols):
        ax.scatter(test_df[TARGET_COL], test_df[col], s=8, alpha=0.35, color=color)
        min_v = float(min(test_df[TARGET_COL].min(), test_df[col].min()))
        max_v = float(max(test_df[TARGET_COL].max(), test_df[col].max()))
        ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    p = PLOTS_DIR / "06_predicted_vs_actual_scatter_test.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 7) Fold-wise backtest MAPE
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=rolling_df, x="fold", y="mape_pct", hue="model", marker="o", ax=ax)
    ax.set_title("Fold-wise Rolling Backtest MAPE")
    p = PLOTS_DIR / "07_foldwise_backtest_mape.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 8) Peak MAPE comparison on test
    peak_test = metrics_df[(metrics_df["split"] == "test")].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=peak_test, x="model", y="peak_mape_pct_top10", palette="viridis", ax=ax)
    ax.set_title("Peak MAPE (Top 10%) on Test")
    ax.tick_params(axis="x", rotation=15)
    p = PLOTS_DIR / "08_peak_mape_test_bar.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # 9) Runtime vs accuracy
    avg_test = metrics_df[metrics_df["split"] == "test"][["model", "mape_pct", "peak_mape_pct_top10"]].copy()
    vis = avg_test.merge(runtime_df, on="model", how="left")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        vis["avg_runtime_sec"],
        vis["mape_pct"],
        s=np.maximum(vis["peak_mape_pct_top10"].to_numpy() * 40, 40),
        c=np.arange(len(vis)),
        cmap="Set2",
        alpha=0.9,
    )
    for _, row in vis.iterrows():
        ax.annotate(row["model"], (row["avg_runtime_sec"], row["mape_pct"]), xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Average inference runtime (sec, log scale)")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Runtime vs Accuracy Tradeoff")
    p = PLOTS_DIR / "09_runtime_vs_accuracy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    return saved


def to_md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._"
    clipped = df.head(max_rows)
    return clipped.to_markdown(index=False)


def build_report(
    report_path: Path,
    data_summary: dict,
    metrics_df: pd.DataFrame,
    window_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    plots: list[Path],
) -> None:
    test_table = metrics_df[metrics_df["split"] == "test"].sort_values("peak_mape_pct_top10")
    val_table = metrics_df[metrics_df["split"] == "validation"].sort_values("mape_pct")
    train_table = metrics_df[metrics_df["split"] == "train"].sort_values("mape_pct")

    ranked = (
        test_table[["model", "peak_mape_pct_top10", "mape_pct", "rmse", "r2"]]
        .sort_values(["peak_mape_pct_top10", "mape_pct", "rmse"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    winner = ranked.iloc[0]["model"] if len(ranked) > 0 else "N/A"

    seasonal_test = window_df[(window_df["split"] == "test") & (window_df["slice_type"] == "season")].copy()
    yearly_test = window_df[(window_df["split"] == "test") & (window_df["slice_type"] == "year")].copy()
    dn_test = window_df[(window_df["split"] == "test") & (window_df["slice_type"] == "day_night")].copy()

    model_cost = runtime_df.copy()
    model_cost["training_time_sec"] = np.nan
    model_cost["tuning_time_sec"] = np.nan

    plot_section = "\n".join([f"- ![{p.name}](plots/{p.name})" for p in plots])

    content = f"""# Gujarat Model Comparison Report (From-Scratch Implementation)

## 1. Objective
This report implements the Model Comparison Plan from scratch using only the saved deployed artifacts:
- XGBoost ONNX model: {XGB_MODEL_PATH.name}
- LSTM ONNX model: {LSTM_MODEL_PATH.name}
- LSTM scaler pack: {LSTM_SCALER_PATH.name}

Walk-forward reserved dates were explicitly excluded from this comparison run, as requested.

## 2. Data Scope and Protocol
- Source dataset: {DATA_PATH.name}
- Global cleaned range in file: {data_summary['full_start']} to {data_summary['full_end']}
- Comparison evaluation range used: {data_summary['eval_start']} to {data_summary['eval_end']}
- Reserved range excluded: {RESERVED_START} to {RESERVED_END}
- Rows before cleaning: {data_summary['rows_before']}
- Rows after feature engineering + dropna constraints: {data_summary['rows_after']}
- Final synchronized rows used for model-to-model comparison: {data_summary['rows_eval']}
- Chronological split (train/validation/test): {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{1 - TRAIN_RATIO - VAL_RATIO:.0%}

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
{to_md_table(test_table, max_rows=20)}

### 5.2 Validation Set
{to_md_table(val_table, max_rows=20)}

### 5.3 Train Set
{to_md_table(train_table, max_rows=20)}

## 6. Winner Ranking (Test Priority Rule)
Ranking keys: lowest Peak MAPE (top-10%), then MAPE, then RMSE.

{to_md_table(ranked, max_rows=20)}

Selected winner by defined rule: **{winner}**

## 7. Robustness and Stability
### 7.1 Rolling Backtest (TimeSeriesSplit)
{to_md_table(rolling_df.sort_values(['model', 'fold']), max_rows=50)}

### 7.2 Time-window Slices (Test)
#### By Year
{to_md_table(yearly_test.sort_values(['slice_value', 'model']), max_rows=50)}

#### By Season
{to_md_table(seasonal_test.sort_values(['slice_value', 'model']), max_rows=50)}

#### Day vs Night
{to_md_table(dn_test.sort_values(['slice_value', 'model']), max_rows=20)}

## 8. Residual and Bias Diagnostics
{to_md_table(bias_df.sort_values(['split', 'model']), max_rows=20)}

## 9. Statistical Significance
Paired tests were computed on identical timestamps of the test set.

{to_md_table(sig_df, max_rows=20)}

Interpretation guidance:
- p < 0.05 suggests the error difference is statistically significant.
- Diebold-Mariano values use absolute-error loss differential.

## 10. Bootstrap Confidence Intervals
95% bootstrap confidence intervals on test metrics.

{to_md_table(boot_df, max_rows=20)}

## 11. Operational Cost Layer
Training/tuning times are not directly inferable from static pre-trained artifacts, but inference cost and model size are measured.

{to_md_table(model_cost, max_rows=20)}

## 12. Graphs
{plot_section}

## 13. Decision Summary
- Best peak-demand performer (primary criterion): **{winner}**
- Accuracy-vs-cost should be judged using both Section 5 and Section 11.
- If two models are close on peak MAPE, lower runtime and simpler operations should be preferred.

## 14. Notes and Constraints
- This is intentionally a from-scratch implementation and does not reuse prior comparison outputs.
- Reserved period June 2024 to June 2025 is left untouched for later walk-forward experiments.
- Report tables were generated directly from this run and saved under model-comparison/tables.

---
Generated by: model-comparison/run_model_comparison.py
"""

    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    validate_inputs()

    raw = pd.read_csv(DATA_PATH)
    raw[DATETIME_COL] = pd.to_datetime(raw[DATETIME_COL], errors="coerce")
    raw = raw.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL).drop_duplicates(subset=[DATETIME_COL], keep="last")

    full_start = raw[DATETIME_COL].min()
    full_end = raw[DATETIME_COL].max()

    # Exclude the reserved walk-forward horizon.
    in_eval = (raw[DATETIME_COL] >= EVAL_START) & (raw[DATETIME_COL] <= EVAL_END)
    df = raw.loc[in_eval].copy().reset_index(drop=True)

    engineered = add_engineered_features(df)

    xgb_df = run_xgb_inference(engineered)
    lstm_df = run_lstm_inference(engineered)
    base_df = run_persistence_baseline(engineered)

    merged = (
        engineered[[DATETIME_COL, TARGET_COL] + XGB_FEATURES]
        .merge(xgb_df[[DATETIME_COL, "xgb_pred"]], on=DATETIME_COL, how="left")
        .merge(base_df[[DATETIME_COL, "baseline_pred"]], on=DATETIME_COL, how="left")
        .merge(lstm_df[[DATETIME_COL, "lstm_pred"]], on=DATETIME_COL, how="left")
    )

    eval_df = merged.dropna(subset=[TARGET_COL, "xgb_pred", "lstm_pred", "baseline_pred"]).copy()
    eval_df = chronological_split(eval_df)

    metrics_df, window_df, bias_df = compute_metric_tables(eval_df)
    rolling_df = rolling_backtest(eval_df, n_splits=5)

    test_df = eval_df[eval_df["split"] == "test"].copy()
    sig_df = significance_tests(test_df)
    boot_df = bootstrap_table(test_df)

    # Build test sequences for runtime benchmarking.
    scaler_pack = joblib.load(LSTM_SCALER_PATH)
    x_scaled_all = scaler_pack["x_scaler"].transform(
        engineered[LSTM_FEATURES].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32)
    ).astype(np.float32)
    x_seq_rt = []
    for end_idx in range(LSTM_LOOKBACK - 1, len(x_scaled_all) - 1):
        x_seq_rt.append(x_scaled_all[end_idx - LSTM_LOOKBACK + 1 : end_idx + 1])
    x_seq_rt = np.asarray(x_seq_rt, dtype=np.float32)

    runtime_df = measure_inference_runtime(eval_df, x_seq_rt)

    metrics_df.to_csv(TABLES_DIR / "main_metrics.csv", index=False)
    window_df.to_csv(TABLES_DIR / "window_metrics.csv", index=False)
    bias_df.to_csv(TABLES_DIR / "bias_metrics.csv", index=False)
    rolling_df.to_csv(TABLES_DIR / "rolling_backtest_metrics.csv", index=False)
    sig_df.to_csv(TABLES_DIR / "significance_tests.csv", index=False)
    boot_df.to_csv(TABLES_DIR / "bootstrap_ci.csv", index=False)
    runtime_df.to_csv(TABLES_DIR / "runtime_cost.csv", index=False)

    plots = make_plots(eval_df, metrics_df, rolling_df, runtime_df)

    data_summary = {
        "rows_before": int(len(raw)),
        "rows_after": int(len(engineered.dropna(subset=XGB_FEATURES + LSTM_FEATURES + [TARGET_COL]))),
        "rows_eval": int(len(eval_df)),
        "full_start": str(full_start),
        "full_end": str(full_end),
        "eval_start": str(EVAL_START),
        "eval_end": str(EVAL_END),
    }

    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    report_path = OUTPUT_DIR / "model_comparison_report.md"
    build_report(
        report_path=report_path,
        data_summary=data_summary,
        metrics_df=metrics_df,
        window_df=window_df,
        bias_df=bias_df,
        rolling_df=rolling_df,
        sig_df=sig_df,
        boot_df=boot_df,
        runtime_df=runtime_df,
        plots=plots,
    )

    print("Completed model comparison run.")
    print(f"Report: {report_path}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
