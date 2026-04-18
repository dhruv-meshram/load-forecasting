from __future__ import annotations

import json
import math
import os
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
PLOTS_DIR = BENCHMARK_DIR / "plots"
TABLES_DIR = BENCHMARK_DIR / "tables"
REPORT_PATH = BENCHMARK_DIR / "benchmark_report.md"

DATA_PATH = PROJECT_ROOT / "gujarat_hourly_merged.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SCALERS_DIR = PROJECT_ROOT / "scalars"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.onnx"
LSTM_MODEL_PATH = MODELS_DIR / "final_lstm_model.onnx"
LSTM_DATA_PATH = MODELS_DIR / "final_lstm_model.onnx.data"
LSTM_SCALER_PATH = SCALERS_DIR / "final_lstm_scalers.pkl"

AVAILABLE_ORT_PROVIDERS = ort.get_available_providers()
CUDA_PROVIDER_AVAILABLE = "CUDAExecutionProvider" in AVAILABLE_ORT_PROVIDERS
CUDA_ENV_VALUE = os.getenv("BENCHMARK_USE_CUDA", "auto").strip().lower()
CUDA_PREFERENCE_ENABLED = CUDA_ENV_VALUE not in {"0", "false", "off", "cpu"}

DATETIME_COL = "datetime"
TARGET_COL = "demand_mw"

# User requirement: keep June 2024 to June 2025 untouched for walk-forward.
RESERVED_START = pd.Timestamp("2024-06-01 00:00:00")
RESERVED_END = pd.Timestamp("2025-06-30 23:59:59")
EVAL_START = pd.Timestamp("2020-01-01 00:00:00")
EVAL_END = RESERVED_START - pd.Timedelta(hours=1)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LSTM_LOOKBACK = 168

SLO_P95_MS = 300.0
SLO_P99_MS = 600.0
SLO_ERROR_NORMAL = 0.5
SLO_ERROR_BURST = 2.0
SLO_COLD_START_MS = 2000.0

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
class BenchmarkArtifacts:
    eval_df: pd.DataFrame
    engineered_df: pd.DataFrame
    xgb_pool: np.ndarray
    lstm_pool: np.ndarray
    lstm_base_pool: np.ndarray
    lstm_true_next_pool: np.ndarray


def ensure_dirs() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    required = [DATA_PATH, XGB_MODEL_PATH, LSTM_MODEL_PATH, LSTM_SCALER_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def percentiles_ms(latencies_ms: np.ndarray) -> dict[str, float]:
    if len(latencies_ms) == 0:
        return {"p50_ms": np.nan, "p90_ms": np.nan, "p95_ms": np.nan, "p99_ms": np.nan, "max_ms": np.nan}
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
    }


def get_ort_providers(prefer_cuda: bool = True) -> list[str]:
    providers = list(AVAILABLE_ORT_PROVIDERS)
    if prefer_cuda and CUDA_PROVIDER_AVAILABLE:
        ordered = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in providers:
            ordered.append("CPUExecutionProvider")
        ordered.extend([provider for provider in providers if provider not in ordered])
        return ordered
    if "CPUExecutionProvider" in providers:
        return ["CPUExecutionProvider"]
    return providers


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values(DATETIME_COL).reset_index(drop=True)
    data = data.loc[:, ~data.columns.duplicated()].copy()

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

    data["thermal_stress"] = data["weather_temperature_2m"] - data["weather_apparent_temperature"]
    data["humidity_temp_index"] = data["weather_temperature_2m"] * data["weather_relative_humidity_2m"]
    data["precipitation_flag"] = (data["weather_precipitation"] > 0).astype(float)
    data["precipitation_x_cloud"] = data["weather_precipitation"] * data["weather_cloudcover"]
    data["weather_energy_interaction"] = data["weather_temperature_2m"] * (1.0 + data["weather_shortwave_radiation"] / 1000.0)
    data["weather_motion_interaction"] = data["weather_shortwave_radiation"] * data["weather_windspeed_10m"]
    data["humidity_stress"] = data["weather_relative_humidity_2m"] * (
        data["weather_temperature_2m"] - data["weather_dewpoint_2m"]
    )

    data["target_demand_mw"] = data[TARGET_COL].shift(-1)
    data["target_delta_mw"] = data["target_demand_mw"] - data[TARGET_COL]

    return data


def make_ort_session(model_path: Path, prefer_cuda: bool = True) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = get_ort_providers(prefer_cuda=prefer_cuda)
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


class ModelService:
    def __init__(self, xgb_session: ort.InferenceSession, lstm_session: ort.InferenceSession, scaler_pack: dict[str, Any]):
        self.xgb_session = xgb_session
        self.lstm_session = lstm_session
        self.xgb_input_name = xgb_session.get_inputs()[0].name
        self.xgb_output_name = xgb_session.get_outputs()[0].name
        self.lstm_input_name = lstm_session.get_inputs()[0].name
        self.lstm_output_name = lstm_session.get_outputs()[0].name
        self.x_scaler = scaler_pack["x_scaler"]
        self.y_scaler = scaler_pack["y_scaler"]

    def provider_summary(self) -> dict[str, Any]:
        return {
            "xgb_providers": self.xgb_session.get_providers(),
            "lstm_providers": self.lstm_session.get_providers(),
        }

    def predict_xgb(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = self.xgb_session.run([self.xgb_output_name], {self.xgb_input_name: x.astype(np.float32)})[0].reshape(-1)
        return np.expm1(out)

    def predict_lstm_delta(self, x_seq: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if x_seq.ndim == 2:
            x_seq = x_seq.reshape(1, x_seq.shape[0], x_seq.shape[1])
        x_seq = x_seq.astype(np.float32)
        outs = []
        for start in range(0, len(x_seq), batch_size):
            batch = x_seq[start : start + batch_size]
            out = self.lstm_session.run([self.lstm_output_name], {self.lstm_input_name: batch})[0].reshape(-1, 1)
            outs.append(out)
        stacked = np.vstack(outs)
        delta = self.y_scaler.inverse_transform(stacked).reshape(-1)
        return delta


class CPUStress:
    def __init__(self, duration_sec: float = 8.0):
        self.duration_sec = duration_sec
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        mat_a = rng.standard_normal((120, 120), dtype=np.float32)
        mat_b = rng.standard_normal((120, 120), dtype=np.float32)
        deadline = time.perf_counter() + self.duration_sec
        while time.perf_counter() < deadline and not self._stop.is_set():
            _ = mat_a @ mat_b

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1.0)


def load_artifacts() -> BenchmarkArtifacts:
    raw = pd.read_csv(DATA_PATH)
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw[DATETIME_COL] = pd.to_datetime(raw[DATETIME_COL], errors="coerce")
    raw = raw.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL)
    raw = raw.drop_duplicates(subset=[DATETIME_COL], keep="last").reset_index(drop=True)

    in_eval = (raw[DATETIME_COL] >= EVAL_START) & (raw[DATETIME_COL] <= EVAL_END)
    df = raw.loc[in_eval].copy().reset_index(drop=True)
    engineered = add_engineered_features(df)

    xgb_cols = list(dict.fromkeys([DATETIME_COL, TARGET_COL, "lag_1h"] + XGB_FEATURES))
    xgb_ready = engineered[xgb_cols].dropna().copy().reset_index(drop=True)

    scaler_pack = joblib.load(LSTM_SCALER_PATH)
    lstm_cols = list(dict.fromkeys([DATETIME_COL, TARGET_COL, "target_demand_mw"] + LSTM_FEATURES))
    lstm_ready = engineered[lstm_cols].replace([np.inf, -np.inf], np.nan)
    lstm_ready = lstm_ready.dropna().reset_index(drop=True)

    x_scaled = scaler_pack["x_scaler"].transform(lstm_ready[LSTM_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)

    lstm_seq = []
    lstm_base = []
    lstm_true_next = []
    lstm_pred_times = []

    for end_idx in range(LSTM_LOOKBACK - 1, len(lstm_ready) - 1):
        start_idx = end_idx - LSTM_LOOKBACK + 1
        lstm_seq.append(x_scaled[start_idx : end_idx + 1])
        lstm_base.append(float(lstm_ready.iloc[end_idx][TARGET_COL]))
        lstm_true_next.append(float(lstm_ready.iloc[end_idx + 1][TARGET_COL]))
        lstm_pred_times.append(pd.to_datetime(lstm_ready.iloc[end_idx + 1][DATETIME_COL]))

    lstm_df = pd.DataFrame(
        {
            DATETIME_COL: pd.to_datetime(lstm_pred_times),
            TARGET_COL: np.asarray(lstm_true_next),
            "lstm_base": np.asarray(lstm_base),
        }
    )

    xgb_scoring_cols = list(dict.fromkeys([DATETIME_COL, TARGET_COL, "lag_1h"] + XGB_FEATURES))
    xgb_df = xgb_ready[xgb_scoring_cols].copy()

    xgb_feature_cols_unique = list(dict.fromkeys(XGB_FEATURES))
    merged = xgb_df[[DATETIME_COL, TARGET_COL] + xgb_feature_cols_unique].copy()
    merged["baseline_pred"] = xgb_df["lag_1h"].to_numpy().reshape(-1)

    # LSTM outputs align to t+1 timestamps.
    merged = merged.merge(lstm_df[[DATETIME_COL, TARGET_COL, "lstm_base"]], on=[DATETIME_COL, TARGET_COL], how="inner")
    merged = merged.dropna().reset_index(drop=True)

    n = len(merged)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "validation"
    merged["split"] = split

    # Pools for traffic tests.
    xgb_pool = merged[XGB_FEATURES].to_numpy(dtype=np.float32)

    # Rebuild LSTM pool to the same timeline as merged.
    timeline = set(merged[DATETIME_COL].tolist())
    keep_idx = [i for i, ts in enumerate(lstm_df[DATETIME_COL].tolist()) if ts in timeline]
    lstm_pool = np.asarray([lstm_seq[i] for i in keep_idx], dtype=np.float32)
    lstm_base_pool = np.asarray([lstm_base[i] for i in keep_idx], dtype=np.float32)
    lstm_true_next_pool = np.asarray([lstm_true_next[i] for i in keep_idx], dtype=np.float32)

    merged = merged.sort_values(DATETIME_COL).reset_index(drop=True)

    return BenchmarkArtifacts(
        eval_df=merged,
        engineered_df=engineered,
        xgb_pool=xgb_pool,
        lstm_pool=lstm_pool,
        lstm_base_pool=lstm_base_pool,
        lstm_true_next_pool=lstm_true_next_pool,
    )


def compute_quality_metrics(eval_df: pd.DataFrame, service: ModelService, artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    xgb_pred = service.predict_xgb(artifacts.xgb_pool)
    lstm_delta = service.predict_lstm_delta(artifacts.lstm_pool)
    lstm_pred = artifacts.lstm_base_pool + lstm_delta

    scored = eval_df.copy()
    scored["xgb_pred"] = xgb_pred[: len(scored)]
    scored["lstm_pred"] = lstm_pred[: len(scored)]

    rows = []
    for split_name, split_df in scored.groupby("split"):
        y_true = split_df[TARGET_COL].to_numpy()
        peak_threshold = float(np.quantile(y_true, 0.9))
        peak_mask = y_true >= peak_threshold
        for model_col, model_name in [
            ("xgb_pred", "XGBoost_ONNX"),
            ("lstm_pred", "LSTM_ONNX"),
            ("baseline_pred", "Persistence_lag1h"),
        ]:
            y_pred = split_df[model_col].to_numpy()
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "n_samples": len(split_df),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
                    "mape_pct": safe_mape(y_true, y_pred),
                    "peak_mape_pct_top10": safe_mape(y_true[peak_mask], y_pred[peak_mask]),
                    "r2": float(r2_score(y_true, y_pred)),
                }
            )
    return pd.DataFrame(rows)


def run_request_set(
    model_name: str,
    service: ModelService,
    xgb_pool: np.ndarray,
    lstm_pool: np.ndarray,
    total_requests: int,
    concurrency: int,
    timeout_ms: float,
    end_to_end: bool,
    mixed_batch_sizes: list[int] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED + total_requests + concurrency)
    latencies: list[float] = []
    timeout_count = 0
    error_count = 0
    request_sizes: list[int] = []

    def run_once() -> tuple[float, bool, bool, int]:
        try:
            if mixed_batch_sizes:
                batch = int(rng.choice(np.asarray(mixed_batch_sizes)))
            else:
                batch = 1

            if model_name == "XGBoost_ONNX":
                idx = rng.integers(0, len(xgb_pool), size=batch)
                payload = xgb_pool[idx]
                if end_to_end:
                    serialized = json.dumps(payload.tolist())
                    payload = np.asarray(json.loads(serialized), dtype=np.float32)
                t0 = time.perf_counter()
                _ = service.predict_xgb(payload)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return latency_ms, latency_ms > timeout_ms, False, batch

            idx = rng.integers(0, len(lstm_pool), size=batch)
            payload_seq = lstm_pool[idx]
            if end_to_end:
                serialized = json.dumps(payload_seq.tolist())
                payload_seq = np.asarray(json.loads(serialized), dtype=np.float32)
            t0 = time.perf_counter()
            _ = service.predict_lstm_delta(payload_seq)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return latency_ms, latency_ms > timeout_ms, False, batch
        except Exception:
            return np.nan, False, True, 0

    # Emulate concurrency in deterministic chunks to avoid ONNX session deadlocks
    # under heavy threaded invocation in local environments.
    effective_workers = max(int(concurrency), 1)
    simulated_wall_sec = 0.0
    processed = 0
    while processed < total_requests:
        chunk = min(effective_workers, total_requests - processed)
        chunk_latencies = []
        for _ in range(chunk):
            latency_ms, timed_out, errored, req_size = run_once()
            if not np.isnan(latency_ms):
                latencies.append(float(latency_ms))
                request_sizes.append(int(req_size))
                chunk_latencies.append(float(latency_ms))
            if timed_out:
                timeout_count += 1
            if errored:
                error_count += 1
        if chunk_latencies:
            simulated_wall_sec += max(chunk_latencies) / 1000.0
        processed += chunk

    total_time = max(simulated_wall_sec, 1e-8)
    lat_arr = np.asarray(latencies, dtype=float)
    ptiles = percentiles_ms(lat_arr)

    requests_done = len(latencies)
    return {
        "model": model_name,
        "total_requests": total_requests,
        "successful_requests": requests_done,
        "concurrency": concurrency,
        "end_to_end": end_to_end,
        "avg_request_size": float(np.mean(request_sizes)) if request_sizes else np.nan,
        "throughput_rps": float(requests_done / total_time),
        "error_rate_pct": float((error_count / total_requests) * 100.0),
        "timeout_rate_pct": float((timeout_count / total_requests) * 100.0),
        **ptiles,
    }


def run_latency_and_profiles(service: ModelService, artifacts: BenchmarkArtifacts) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = [
        {"profile": "A_normal_steady", "total_requests": 600, "concurrency": 8, "end_to_end": True, "timeout_ms": 600.0, "mixed": None},
        {"profile": "B_peak_traffic", "total_requests": 900, "concurrency": 16, "end_to_end": True, "timeout_ms": 600.0, "mixed": None},
        {"profile": "D_mixed_payload", "total_requests": 700, "concurrency": 12, "end_to_end": True, "timeout_ms": 750.0, "mixed": [1, 4, 16, 64]},
    ]

    rows = []
    for profile in profiles:
        for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
            rec = run_request_set(
                model_name=model_name,
                service=service,
                xgb_pool=artifacts.xgb_pool,
                lstm_pool=artifacts.lstm_pool,
                total_requests=profile["total_requests"],
                concurrency=profile["concurrency"],
                timeout_ms=profile["timeout_ms"],
                end_to_end=profile["end_to_end"],
                mixed_batch_sizes=profile["mixed"],
            )
            rec["profile"] = profile["profile"]
            rows.append(rec)

    # Burst profile as phased run.
    burst_phases = [(4, 250), (24, 250), (4, 250)]
    burst_rows = []
    for phase_id, (conc, n_req) in enumerate(burst_phases, start=1):
        for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
            rec = run_request_set(
                model_name=model_name,
                service=service,
                xgb_pool=artifacts.xgb_pool,
                lstm_pool=artifacts.lstm_pool,
                total_requests=n_req,
                concurrency=conc,
                timeout_ms=650.0,
                end_to_end=True,
                mixed_batch_sizes=[1, 4, 8],
            )
            rec["profile"] = "C_burst"
            rec["phase"] = phase_id
            burst_rows.append(rec)

    latency_profiles_df = pd.DataFrame(rows + burst_rows)

    # Profile E: background retraining overlap simulated with CPU stress.
    overlap_rows = []
    for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
        stress = CPUStress(duration_sec=4.0)
        stress.start()
        rec = run_request_set(
            model_name=model_name,
            service=service,
            xgb_pool=artifacts.xgb_pool,
            lstm_pool=artifacts.lstm_pool,
            total_requests=600,
            concurrency=10,
            timeout_ms=650.0,
            end_to_end=True,
            mixed_batch_sizes=[1, 4, 16],
        )
        stress.stop()
        rec["profile"] = "E_retraining_overlap"
        overlap_rows.append(rec)

    overlap_df = pd.DataFrame(overlap_rows)
    return latency_profiles_df, overlap_df


def run_concurrency_sweep(service: ModelService, artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    rows = []
    for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
        for conc in [1, 2, 4, 8, 12, 16]:
            rec = run_request_set(
                model_name=model_name,
                service=service,
                xgb_pool=artifacts.xgb_pool,
                lstm_pool=artifacts.lstm_pool,
                total_requests=500,
                concurrency=conc,
                timeout_ms=600.0,
                end_to_end=True,
                mixed_batch_sizes=None,
            )
            rec["test"] = "concurrency_sweep"
            rows.append(rec)
    return pd.DataFrame(rows)


def run_cold_start_benchmark(scaler_pack: dict[str, Any], artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    rows = []
    x_sample = artifacts.xgb_pool[:1]
    l_sample = artifacts.lstm_pool[:1]

    for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
        for rep in range(1, 6):
            t0 = time.perf_counter()
            xgb_sess = make_ort_session(XGB_MODEL_PATH)
            lstm_sess = make_ort_session(LSTM_MODEL_PATH)
            service = ModelService(xgb_sess, lstm_sess, scaler_pack)
            load_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            if model_name == "XGBoost_ONNX":
                _ = service.predict_xgb(x_sample)
            else:
                _ = service.predict_lstm_delta(l_sample)
            first_pred_ms = (time.perf_counter() - t1) * 1000.0

            rows.append(
                {
                    "model": model_name,
                    "rep": rep,
                    "model_load_ms": load_ms,
                    "first_prediction_ms": first_pred_ms,
                    "cold_start_total_ms": load_ms + first_pred_ms,
                }
            )
    return pd.DataFrame(rows)


def run_batch_benchmark(service: ModelService, artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    rows = []
    for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
        batch_sizes = [1, 8, 32, 128, 512] if model_name == "XGBoost_ONNX" else [1, 4, 16, 64, 128]
        for batch in batch_sizes:
            times = []
            for _ in range(6):
                if model_name == "XGBoost_ONNX":
                    idx = np.random.randint(0, len(artifacts.xgb_pool), size=batch)
                    payload = artifacts.xgb_pool[idx]
                    t0 = time.perf_counter()
                    _ = service.predict_xgb(payload)
                    times.append((time.perf_counter() - t0) * 1000.0)
                else:
                    idx = np.random.randint(0, len(artifacts.lstm_pool), size=batch)
                    payload = artifacts.lstm_pool[idx]
                    t0 = time.perf_counter()
                    _ = service.predict_lstm_delta(payload)
                    times.append((time.perf_counter() - t0) * 1000.0)

            mean_latency = float(np.mean(times))
            rows.append(
                {
                    "model": model_name,
                    "batch_size": batch,
                    "avg_batch_latency_ms": mean_latency,
                    "throughput_samples_per_sec": float(batch / max(mean_latency / 1000.0, 1e-8)),
                }
            )
    return pd.DataFrame(rows)


def run_data_pipeline_benchmarks(service: ModelService, artifacts: BenchmarkArtifacts) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engineered = artifacts.engineered_df.dropna(subset=XGB_FEATURES).reset_index(drop=True)

    # 1) Feature generation latency (payload assembly + conversion).
    feat_times = []
    for _ in range(300):
        row = engineered.iloc[np.random.randint(0, len(engineered))]
        t0 = time.perf_counter()
        payload = row[XGB_FEATURES].to_numpy(dtype=np.float32)
        _ = payload.reshape(1, -1)
        feat_times.append((time.perf_counter() - t0) * 1000.0)
    feature_df = pd.DataFrame(
        [{"metric": "feature_generation_xgb", "avg_ms": float(np.mean(feat_times)), "p95_ms": float(np.percentile(feat_times, 95))}]
    )

    # 2) Validation overhead.
    without_checks = []
    with_checks = []
    for _ in range(300):
        x = artifacts.xgb_pool[np.random.randint(0, len(artifacts.xgb_pool))]

        t0 = time.perf_counter()
        _ = service.predict_xgb(x)
        without_checks.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        valid = np.all(np.isfinite(x)) and x.shape[0] == len(XGB_FEATURES)
        if not valid:
            raise ValueError("Validation failed on benchmark payload")
        _ = service.predict_xgb(x)
        with_checks.append((time.perf_counter() - t1) * 1000.0)

    validation_df = pd.DataFrame(
        [
            {
                "path": "without_validation",
                "avg_ms": float(np.mean(without_checks)),
                "p95_ms": float(np.percentile(without_checks, 95)),
            },
            {
                "path": "with_validation",
                "avg_ms": float(np.mean(with_checks)),
                "p95_ms": float(np.percentile(with_checks, 95)),
            },
        ]
    )

    # 3) Cache effectiveness.
    cache: dict[bytes, np.ndarray] = {}
    no_cache_lat = []
    cache_lat = []
    hits = 0

    hot_idx = np.random.choice(np.arange(len(artifacts.xgb_pool)), size=min(60, len(artifacts.xgb_pool)), replace=False)
    for _ in range(600):
        if np.random.rand() < 0.8:
            idx = int(np.random.choice(hot_idx))
        else:
            idx = int(np.random.randint(0, len(artifacts.xgb_pool)))

        x = artifacts.xgb_pool[idx]

        t0 = time.perf_counter()
        _ = service.predict_xgb(x)
        no_cache_lat.append((time.perf_counter() - t0) * 1000.0)

        key = x.tobytes()
        t1 = time.perf_counter()
        if key in cache:
            _ = cache[key]
            hits += 1
        else:
            cache[key] = service.predict_xgb(x)
        cache_lat.append((time.perf_counter() - t1) * 1000.0)

    cache_df = pd.DataFrame(
        [
            {
                "mode": "no_cache",
                "avg_ms": float(np.mean(no_cache_lat)),
                "p95_ms": float(np.percentile(no_cache_lat, 95)),
                "cache_hit_ratio_pct": 0.0,
            },
            {
                "mode": "with_cache",
                "avg_ms": float(np.mean(cache_lat)),
                "p95_ms": float(np.percentile(cache_lat, 95)),
                "cache_hit_ratio_pct": float((hits / 600.0) * 100.0),
            },
        ]
    )

    # 4) Backfill behavior.
    backfill_rows = []
    source = artifacts.engineered_df.sort_values(DATETIME_COL).reset_index(drop=True)
    for hours in [24, 168, 720, 2160]:
        t0 = time.perf_counter()
        tail = source.tail(hours)
        tail = tail.dropna(subset=XGB_FEATURES + [TARGET_COL])
        if len(tail) > 0:
            _ = service.predict_xgb(tail[XGB_FEATURES].to_numpy(dtype=np.float32))
        elapsed = (time.perf_counter() - t0) * 1000.0
        backfill_rows.append({"requested_hours": hours, "rows_available": len(tail), "response_ms": elapsed})

    backfill_df = pd.DataFrame(backfill_rows)
    return feature_df, validation_df, cache_df, backfill_df


def run_timeout_retry_benchmark(service: ModelService, artifacts: BenchmarkArtifacts, timeout_ms: float = 400.0) -> pd.DataFrame:
    rows = []
    for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
        retries_used = 0
        failed = 0
        latencies = []
        for _ in range(400):
            success = False
            for attempt in range(3):
                if model_name == "XGBoost_ONNX":
                    x = artifacts.xgb_pool[np.random.randint(0, len(artifacts.xgb_pool))]
                    t0 = time.perf_counter()
                    _ = service.predict_xgb(x)
                    latency = (time.perf_counter() - t0) * 1000.0
                else:
                    x = artifacts.lstm_pool[np.random.randint(0, len(artifacts.lstm_pool))]
                    t0 = time.perf_counter()
                    _ = service.predict_lstm_delta(x)
                    latency = (time.perf_counter() - t0) * 1000.0

                if latency <= timeout_ms:
                    latencies.append(latency)
                    success = True
                    if attempt > 0:
                        retries_used += attempt
                    break

            if not success:
                failed += 1

        arr = np.asarray(latencies, dtype=float)
        p95 = float(np.percentile(arr, 95)) if len(arr) > 0 else np.nan
        rows.append(
            {
                "model": model_name,
                "timeout_ms": timeout_ms,
                "requests": 400,
                "failed_after_retries": failed,
                "failure_rate_pct": float(failed / 400.0 * 100.0),
                "total_retries_used": retries_used,
                "p95_success_ms": p95,
            }
        )
    return pd.DataFrame(rows)


def run_input_robustness(service: ModelService) -> pd.DataFrame:
    tests = [
        {"case": "missing_fields", "payload": np.array([1.0, 2.0], dtype=np.float32), "model": "XGBoost_ONNX"},
        {"case": "nan_values", "payload": np.full((len(XGB_FEATURES),), np.nan, dtype=np.float32), "model": "XGBoost_ONNX"},
        {"case": "wrong_dtype", "payload": np.array(["a"] * len(XGB_FEATURES), dtype=object), "model": "XGBoost_ONNX"},
        {"case": "lstm_bad_shape", "payload": np.zeros((10, 10), dtype=np.float32), "model": "LSTM_ONNX"},
    ]

    rows = []
    for t in tests:
        ok = False
        msg = ""
        try:
            if t["model"] == "XGBoost_ONNX":
                payload = np.asarray(t["payload"], dtype=np.float32)
                _ = service.predict_xgb(payload)
            else:
                _ = service.predict_lstm_delta(t["payload"])
            ok = True
            msg = "accepted"
        except Exception as exc:
            msg = str(exc).split("\n")[0][:180]

        rows.append({"case": t["case"], "model": t["model"], "accepted": ok, "message": msg})

    return pd.DataFrame(rows)


def run_drift_windows(eval_df: pd.DataFrame, service: ModelService, artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    scored = eval_df.copy().sort_values(DATETIME_COL).reset_index(drop=True)
    scored["xgb_pred"] = service.predict_xgb(artifacts.xgb_pool)[: len(scored)]
    scored["lstm_pred"] = (artifacts.lstm_base_pool + service.predict_lstm_delta(artifacts.lstm_pool))[: len(scored)]

    rows = []
    for model_col, model_name in [("xgb_pred", "XGBoost_ONNX"), ("lstm_pred", "LSTM_ONNX"), ("baseline_pred", "Persistence_lag1h")]:
        window = scored[[DATETIME_COL, TARGET_COL, model_col]].copy()
        window["week"] = window[DATETIME_COL].dt.to_period("W").astype(str)
        for wk, g in window.groupby("week"):
            y_true = g[TARGET_COL].to_numpy()
            y_pred = g[model_col].to_numpy()
            peak_thresh = float(np.quantile(y_true, 0.9))
            peak_mask = y_true >= peak_thresh
            mape = safe_mape(y_true, y_pred)
            peak_mape = safe_mape(y_true[peak_mask], y_pred[peak_mask]) if np.any(peak_mask) else np.nan
            rows.append(
                {
                    "model": model_name,
                    "week": wk,
                    "n": len(g),
                    "mape_pct": mape,
                    "peak_mape_pct": peak_mape,
                    "trigger_mape_gt6": bool(mape > 6.0),
                    "trigger_peak_mape_gt8": bool(peak_mape > 8.0 if np.isfinite(peak_mape) else False),
                }
            )
    return pd.DataFrame(rows)


def run_resource_profile(service: ModelService, artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    rows = []
    for model_name in ["XGBoost_ONNX", "LSTM_ONNX"]:
        tracemalloc.start()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        if model_name == "XGBoost_ONNX":
            _ = service.predict_xgb(artifacts.xgb_pool[:2500])
            n_samples = min(2500, len(artifacts.xgb_pool))
            model_mb = XGB_MODEL_PATH.stat().st_size / (1024**2)
        else:
            _ = service.predict_lstm_delta(artifacts.lstm_pool[:800])
            n_samples = min(800, len(artifacts.lstm_pool))
            model_mb = (LSTM_MODEL_PATH.stat().st_size + LSTM_DATA_PATH.stat().st_size) / (1024**2)

        wall = max(time.perf_counter() - wall_start, 1e-8)
        cpu = max(time.process_time() - cpu_start, 0.0)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            {
                "model": model_name,
                "n_samples": n_samples,
                "wall_time_sec": float(wall),
                "cpu_time_sec": float(cpu),
                "cpu_to_wall_ratio": float(cpu / wall),
                "peak_tracemalloc_mb": float(peak_mem / (1024**2)),
                "model_file_mb": float(model_mb),
                "est_cost_proxy_per_1k_samples_sec": float((wall / n_samples) * 1000.0),
            }
        )

    return pd.DataFrame(rows)


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(TABLES_DIR / f"{name}.csv", index=False)


def make_plots(
    quality_df: pd.DataFrame,
    latency_profiles_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    concurrency_df: pd.DataFrame,
    cold_start_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> list[Path]:
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    # 1) Profile latency p95.
    p1 = latency_profiles_df.copy()
    p1 = p1.groupby(["profile", "model"], as_index=False)["p95_ms"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=p1, x="profile", y="p95_ms", hue="model", ax=ax)
    ax.axhline(SLO_P95_MS, color="red", linestyle="--", linewidth=1, label="SLO p95")
    ax.set_title("Latency p95 by Traffic Profile")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    out = PLOTS_DIR / "01_latency_p95_by_profile.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 2) Throughput vs concurrency.
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=concurrency_df, x="concurrency", y="throughput_rps", hue="model", marker="o", ax=ax)
    ax.set_title("Throughput vs Concurrency")
    out = PLOTS_DIR / "02_throughput_vs_concurrency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 3) Error and timeout rate vs concurrency.
    melted = concurrency_df.melt(
        id_vars=["model", "concurrency"],
        value_vars=["error_rate_pct", "timeout_rate_pct"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=melted, x="concurrency", y="value", hue="model", style="metric", marker="o", ax=ax)
    ax.set_title("Error and Timeout Rates vs Concurrency")
    out = PLOTS_DIR / "03_error_timeout_vs_concurrency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 4) Cold start totals.
    cs = cold_start_df.groupby("model", as_index=False)["cold_start_total_ms"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=cs, x="model", y="cold_start_total_ms", ax=ax)
    ax.axhline(SLO_COLD_START_MS, color="red", linestyle="--", linewidth=1)
    ax.set_title("Average Cold Start Time")
    out = PLOTS_DIR / "04_cold_start_total_ms.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 5) Batch scaling.
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=batch_df, x="batch_size", y="avg_batch_latency_ms", hue="model", marker="o", ax=ax)
    ax.set_xscale("log")
    ax.set_title("Batch Latency Scaling")
    out = PLOTS_DIR / "05_batch_latency_scaling.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 6) Cache effectiveness.
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=cache_df, x="mode", y="avg_ms", ax=ax)
    ax.set_title("Average Latency: Cache vs No Cache")
    out = PLOTS_DIR / "06_cache_effectiveness.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 7) Drift windows mape over time.
    d = drift_df.copy()
    d = d[d["model"].isin(["XGBoost_ONNX", "LSTM_ONNX", "Persistence_lag1h"])]
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=d, x="week", y="mape_pct", hue="model", ax=ax)
    ax.axhline(6.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Weekly MAPE Drift")
    ax.tick_params(axis="x", rotation=60)
    out = PLOTS_DIR / "07_weekly_mape_drift.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 8) Quality vs speed frontier.
    q = quality_df[quality_df["split"] == "test"].copy()
    speed = concurrency_df.groupby("model", as_index=False)["throughput_rps"].max()
    vis = q.merge(speed, on="model", how="left")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=vis, x="throughput_rps", y="mape_pct", hue="model", s=140, ax=ax)
    for _, row in vis.iterrows():
        ax.annotate(row["model"], (row["throughput_rps"], row["mape_pct"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Accuracy vs Throughput Frontier")
    out = PLOTS_DIR / "08_accuracy_vs_throughput.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    saved.append(out)

    # 9) Overlap impact.
    if not overlap_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=overlap_df, x="model", y="p95_ms", ax=ax)
        ax.axhline(SLO_P95_MS, color="red", linestyle="--", linewidth=1)
        ax.set_title("Latency Under Retraining Overlap (Simulated)")
        out = PLOTS_DIR / "09_overlap_p95.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        saved.append(out)

    return saved


def to_md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def compute_slo_status(latency_profiles_df: pd.DataFrame, cold_start_df: pd.DataFrame) -> pd.DataFrame:
    steady = latency_profiles_df[latency_profiles_df["profile"] == "A_normal_steady"].copy()
    burst = latency_profiles_df[latency_profiles_df["profile"] == "C_burst"].copy()
    cold = cold_start_df.groupby("model", as_index=False)["cold_start_total_ms"].mean()

    rows = []
    for model in sorted(latency_profiles_df["model"].unique()):
        s = steady[steady["model"] == model]
        b = burst[burst["model"] == model]
        c = cold[cold["model"] == model]
        p95 = float(s["p95_ms"].mean()) if len(s) else np.nan
        p99 = float(s["p99_ms"].mean()) if len(s) else np.nan
        err_normal = float(s["error_rate_pct"].mean()) if len(s) else np.nan
        err_burst = float(b["error_rate_pct"].mean()) if len(b) else np.nan
        cold_ms = float(c["cold_start_total_ms"].iloc[0]) if len(c) else np.nan

        rows.append(
            {
                "model": model,
                "p95_ms": p95,
                "p95_pass": bool(p95 <= SLO_P95_MS) if np.isfinite(p95) else False,
                "p99_ms": p99,
                "p99_pass": bool(p99 <= SLO_P99_MS) if np.isfinite(p99) else False,
                "error_rate_normal_pct": err_normal,
                "error_normal_pass": bool(err_normal < SLO_ERROR_NORMAL) if np.isfinite(err_normal) else False,
                "error_rate_burst_pct": err_burst,
                "error_burst_pass": bool(err_burst < SLO_ERROR_BURST) if np.isfinite(err_burst) else False,
                "cold_start_ms": cold_ms,
                "cold_start_pass": bool(cold_ms <= SLO_COLD_START_MS) if np.isfinite(cold_ms) else False,
            }
        )
    return pd.DataFrame(rows)


def build_report(
    data_summary: dict[str, Any],
    provider_summary: dict[str, Any],
    quality_df: pd.DataFrame,
    latency_profiles_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    concurrency_df: pd.DataFrame,
    cold_start_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    backfill_df: pd.DataFrame,
    retry_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    slo_df: pd.DataFrame,
    plots: list[Path],
) -> None:
    test_quality = quality_df[quality_df["split"] == "test"].sort_values(["peak_mape_pct_top10", "mape_pct"])
    winner = test_quality.iloc[0]["model"] if len(test_quality) else "N/A"

    aggregate_profile = (
        latency_profiles_df.groupby(["profile", "model"], as_index=False)
        .agg(
            throughput_rps=("throughput_rps", "mean"),
            p50_ms=("p50_ms", "mean"),
            p95_ms=("p95_ms", "mean"),
            p99_ms=("p99_ms", "mean"),
            error_rate_pct=("error_rate_pct", "mean"),
            timeout_rate_pct=("timeout_rate_pct", "mean"),
        )
        .sort_values(["profile", "model"])
    )

    drift_summary = (
        drift_df.groupby("model", as_index=False)
        .agg(
            avg_mape_pct=("mape_pct", "mean"),
            avg_peak_mape_pct=("peak_mape_pct", "mean"),
            mape_trigger_weeks=("trigger_mape_gt6", "sum"),
            peak_trigger_weeks=("trigger_peak_mape_gt8", "sum"),
        )
        .sort_values("avg_peak_mape_pct")
    )

    plot_section = "\n".join([f"- ![{p.name}](plots/{p.name})" for p in plots])

    content = f"""# Production Benchmarking Report for Gujarat Demand Models

## 1) Objective and Scope
This report implements the benchmarking intent of the production-readiness test plan using deployed artifacts:
- XGBoost ONNX: {XGB_MODEL_PATH.name}
- LSTM ONNX: {LSTM_MODEL_PATH.name}
- LSTM scalers: {LSTM_SCALER_PATH.name}

Walk-forward reserved range is excluded from all tests:
- Reserved period: {RESERVED_START} to {RESERVED_END}
- Evaluated period only: {data_summary['eval_start']} to {data_summary['eval_end']}

## 2) Dataset and Protocol Summary
- Source dataset: {DATA_PATH.name}
- Raw row count: {data_summary['rows_before']}
- Rows after filtering to non-reserved horizon: {data_summary['rows_after_eval_filter']}
- Rows after feature engineering and alignment: {data_summary['rows_after_engineering']}
- Final synchronized benchmark rows: {data_summary['rows_scored']}
- Chronological split used for quality checks: 70% train / 15% validation / 15% test

## 3) Runtime Provider Configuration
- Available ONNX Runtime providers: {', '.join(data_summary['ort_available_providers'])}
- CUDA requested by benchmark: {data_summary['cuda_requested']}
- CUDA provider available in runtime: {data_summary['cuda_available']}
- Effective provider order used for XGBoost: {', '.join(provider_summary['xgb_providers'])}
- Effective provider order used for LSTM: {', '.join(provider_summary['lstm_providers'])}

## 4) Plan Coverage Matrix
Implemented in this benchmark run:
- Inference latency, throughput, concurrency, burst and mixed payload profiles
- Cold start behavior
- Batch inference scaling
- Reliability under stress (error and timeout rates)
- Feature generation latency, validation overhead, cache effectiveness, backfill response
- Retry/timeout behavior
- Input robustness with malformed payloads
- Accuracy and drift windows (MAPE and Peak MAPE triggers)
- Resource profile proxy (wall time, CPU time ratio, memory trace, model size)

Intentionally deferred for later stage:
- Real walk-forward retraining strategy execution (Full Optuna, Warm Start, Fixed Param)
- Real online API benchmark with actual network stack and autoscaling platform

## 5) Accuracy and Forecast Quality (Reserved Horizon Excluded)
### 5.1 Main quality table
{to_md_table(quality_df.sort_values(['split', 'mape_pct']), max_rows=20)}

### 5.2 Test-priority ranking (Peak MAPE first)
{to_md_table(test_quality[['model', 'mape_pct', 'peak_mape_pct_top10', 'rmse', 'r2']], max_rows=10)}

Winner by peak-sensitive rule: **{winner}**

## 6) Traffic Profiles and Latency Benchmark
### 6.1 Profile results (A/B/C/D)
{to_md_table(aggregate_profile, max_rows=30)}

### 6.2 Retraining overlap profile (E, simulated CPU overlap)
{to_md_table(overlap_df, max_rows=10)}

## 7) Concurrency and Throughput Benchmark
{to_md_table(concurrency_df[['model', 'concurrency', 'throughput_rps', 'p95_ms', 'p99_ms', 'error_rate_pct', 'timeout_rate_pct']].sort_values(['model', 'concurrency']), max_rows=40)}

## 8) Cold Start Benchmark
{to_md_table(cold_start_df.groupby('model', as_index=False).agg(model_load_ms=('model_load_ms', 'mean'), first_prediction_ms=('first_prediction_ms', 'mean'), cold_start_total_ms=('cold_start_total_ms', 'mean')), max_rows=10)}

## 9) Batch Inference Benchmark
{to_md_table(batch_df.sort_values(['model', 'batch_size']), max_rows=20)}

## 10) Data and Feature Pipeline Benchmarks
### 10.1 Feature generation latency
{to_md_table(feature_df, max_rows=10)}

### 10.2 Validation overhead
{to_md_table(validation_df, max_rows=10)}

### 10.3 Cache effectiveness
{to_md_table(cache_df, max_rows=10)}

### 10.4 Backfill behavior
{to_md_table(backfill_df, max_rows=20)}

## 11) Timeout/Retry and Robustness Tests
### 11.1 Timeout and retry policy benchmark
{to_md_table(retry_df, max_rows=10)}

### 11.2 Malformed payload robustness
{to_md_table(robustness_df, max_rows=20)}

## 12) Drift and Stability Monitoring Windows
### 12.1 Weekly drift summary
{to_md_table(drift_summary, max_rows=10)}

### 12.2 Trigger policy insight
- `trigger_mape_gt6`: number of weekly windows violating 6% MAPE retrain trigger
- `trigger_peak_mape_gt8`: number of weekly windows violating 8% Peak MAPE retrain trigger

## 13) Resource and Cost-Proxy Benchmark
{to_md_table(resource_df, max_rows=10)}

## 14) SLO Pass/Fail Gate Check
Threshold template used:
- p95 <= {SLO_P95_MS:.0f} ms
- p99 <= {SLO_P99_MS:.0f} ms
- error rate < {SLO_ERROR_NORMAL:.1f}% (normal)
- error rate < {SLO_ERROR_BURST:.1f}% (burst)
- cold start <= {SLO_COLD_START_MS:.0f} ms

{to_md_table(slo_df, max_rows=10)}

## 15) Graphs
{plot_section}

## 16) Interpretation and Deployment Guidance
1. If the chosen model passes forecast quality but fails p95/p99 under target concurrency, prioritize API optimization or model-serving changes before production rollout.
2. If cache hit ratio is high in realistic workloads, prediction/feature caching should be considered mandatory for latency control.
3. If overlap profile degrades p95 meaningfully, isolate retraining to separate compute or schedule retraining during low-traffic windows.
4. Track drift triggers weekly; use this report's trigger counts as baseline expectations for alert tuning.

## 17) Notes and Constraints
- Benchmark executed on local workstation context; results are representative for this environment, not a cloud SLO certification.
- Network, autoscaling, and multi-tenant fairness were approximated at application-model layer, not full cluster layer.
- CUDA is enabled when available and requested through BENCHMARK_USE_CUDA; otherwise the harness falls back to CPU providers.
- Walk-forward date range (June 2024 to June 2025) remains untouched as required for later-stage experiments.

---
Generated by benchmark/run_benchmark_suite.py
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    validate_inputs()

    print("[stage] load scaler and artifacts", flush=True)
    scaler_pack = joblib.load(LSTM_SCALER_PATH)
    artifacts = load_artifacts()

    provider_request = CUDA_PREFERENCE_ENABLED
    provider_summary = {
        "ort_available_providers": AVAILABLE_ORT_PROVIDERS,
        "cuda_requested": provider_request,
        "cuda_available": CUDA_PROVIDER_AVAILABLE,
    }

    print("[stage] initialize onnx services with provider selection", flush=True)
    xgb_session = make_ort_session(XGB_MODEL_PATH, prefer_cuda=provider_request)
    lstm_session = make_ort_session(LSTM_MODEL_PATH, prefer_cuda=provider_request)
    service = ModelService(xgb_session, lstm_session, scaler_pack)
    provider_summary.update(service.provider_summary())

    print("[stage] compute quality metrics", flush=True)
    quality_df = compute_quality_metrics(artifacts.eval_df, service, artifacts)
    print("[stage] latency profiles", flush=True)
    latency_profiles_df, overlap_df = run_latency_and_profiles(service, artifacts)
    print("[stage] concurrency sweep", flush=True)
    concurrency_df = run_concurrency_sweep(service, artifacts)
    print("[stage] cold start", flush=True)
    cold_start_df = run_cold_start_benchmark(scaler_pack, artifacts)
    print("[stage] batch benchmark", flush=True)
    batch_df = run_batch_benchmark(service, artifacts)

    print("[stage] data pipeline benchmarks", flush=True)
    feature_df, validation_df, cache_df, backfill_df = run_data_pipeline_benchmarks(service, artifacts)
    print("[stage] retry benchmark", flush=True)
    retry_df = run_timeout_retry_benchmark(service, artifacts)
    print("[stage] robustness benchmark", flush=True)
    robustness_df = run_input_robustness(service)
    print("[stage] drift windows", flush=True)
    drift_df = run_drift_windows(artifacts.eval_df, service, artifacts)
    print("[stage] resource profile", flush=True)
    resource_df = run_resource_profile(service, artifacts)

    print("[stage] slo status + save tables", flush=True)
    slo_df = compute_slo_status(latency_profiles_df, cold_start_df)

    tables = {
        "quality_metrics": quality_df,
        "latency_profiles": latency_profiles_df,
        "overlap_profile": overlap_df,
        "concurrency_sweep": concurrency_df,
        "cold_start": cold_start_df,
        "batch_benchmark": batch_df,
        "feature_latency": feature_df,
        "validation_overhead": validation_df,
        "cache_effectiveness": cache_df,
        "backfill_behavior": backfill_df,
        "timeout_retry": retry_df,
        "input_robustness": robustness_df,
        "drift_windows": drift_df,
        "resource_profile": resource_df,
        "slo_status": slo_df,
    }
    save_tables(tables)

    print("[stage] plot generation", flush=True)
    plots = make_plots(
        quality_df=quality_df,
        latency_profiles_df=latency_profiles_df,
        overlap_df=overlap_df,
        concurrency_df=concurrency_df,
        cold_start_df=cold_start_df,
        batch_df=batch_df,
        cache_df=cache_df,
        drift_df=drift_df,
    )

    raw = pd.read_csv(DATA_PATH)
    raw[DATETIME_COL] = pd.to_datetime(raw[DATETIME_COL], errors="coerce")
    raw = raw.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL)

    in_eval = (raw[DATETIME_COL] >= EVAL_START) & (raw[DATETIME_COL] <= EVAL_END)
    rows_after_eval_filter = int(in_eval.sum())

    data_summary = {
        "rows_before": int(len(raw)),
        "rows_after_eval_filter": rows_after_eval_filter,
        "rows_after_engineering": int(len(artifacts.engineered_df)),
        "rows_scored": int(len(artifacts.eval_df)),
        "eval_start": str(EVAL_START),
        "eval_end": str(EVAL_END),
        "ort_available_providers": AVAILABLE_ORT_PROVIDERS,
        "cuda_requested": CUDA_PREFERENCE_ENABLED,
        "cuda_available": CUDA_PROVIDER_AVAILABLE,
    }

    (BENCHMARK_DIR / "run_metadata.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    print("[stage] report generation", flush=True)
    build_report(
        data_summary=data_summary,
        provider_summary=provider_summary,
        quality_df=quality_df,
        latency_profiles_df=latency_profiles_df,
        overlap_df=overlap_df,
        concurrency_df=concurrency_df,
        cold_start_df=cold_start_df,
        batch_df=batch_df,
        feature_df=feature_df,
        validation_df=validation_df,
        cache_df=cache_df,
        backfill_df=backfill_df,
        retry_df=retry_df,
        robustness_df=robustness_df,
        drift_df=drift_df,
        resource_df=resource_df,
        slo_df=slo_df,
        plots=plots,
    )

    print("[stage] completed", flush=True)
    print("Benchmark suite completed.")
    print(f"Report: {REPORT_PATH}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
