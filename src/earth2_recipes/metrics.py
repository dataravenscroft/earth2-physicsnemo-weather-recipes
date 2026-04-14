"""Small deterministic forecast metrics for starter benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path

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


# Units for each metric given the field units (e.g. "K" for temperature).
# anomaly_correlation is always dimensionless.
_DIMENSIONLESS = {"anomaly_correlation"}


def write_metrics_csv(
    output_path: Path,
    metrics: dict[str, float],
    variable: str,
    baseline: str,
    units: str,
) -> None:
    """Write a benchmark metrics table as CSV.

    Columns: metric, value, units, variable, baseline
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value", "units", "variable", "baseline"])
        for name, value in metrics.items():
            row_units = "dimensionless" if name in _DIMENSIONLESS else units
            writer.writerow([name, f"{value:.4f}", row_units, variable, baseline])

