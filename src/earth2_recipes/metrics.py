"""Small deterministic forecast metrics for starter benchmarks."""

from __future__ import annotations

import numpy as np


def rmse(truth: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecast - truth) ** 2)))


def mae(truth: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - truth)))


def bias(truth: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(forecast - truth))


def anomaly_correlation(truth: np.ndarray, forecast: np.ndarray) -> float:
    truth_anom = truth - np.mean(truth)
    forecast_anom = forecast - np.mean(forecast)
    denominator = np.linalg.norm(truth_anom.ravel()) * np.linalg.norm(forecast_anom.ravel())
    if denominator == 0.0:
        return 0.0
    return float(np.dot(truth_anom.ravel(), forecast_anom.ravel()) / denominator)


def summarize_forecast(truth: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(truth, forecast),
        "mae": mae(truth, forecast),
        "bias": bias(truth, forecast),
        "anomaly_correlation": anomaly_correlation(truth, forecast),
    }
