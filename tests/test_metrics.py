"""Tests for forecast metrics in earth2_recipes.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from earth2_recipes.metrics import anomaly_correlation, bias, mae, rmse, summarize_forecast


RNG = np.random.default_rng(0)
SHAPE = (24, 48)


def _pair(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    truth = RNG.standard_normal(SHAPE)
    forecast = truth + RNG.standard_normal(SHAPE) * scale
    return truth, forecast


# --- rmse ---

def test_rmse_nonnegative():
    truth, forecast = _pair()
    assert rmse(truth, forecast) >= 0.0


def test_rmse_perfect():
    arr = RNG.standard_normal(SHAPE)
    assert rmse(arr, arr) == pytest.approx(0.0, abs=1e-12)


def test_rmse_known_value():
    # constant offset of 2 → RMSE == 2
    truth = np.zeros(SHAPE)
    forecast = np.full(SHAPE, 2.0)
    assert rmse(truth, forecast) == pytest.approx(2.0)


# --- mae ---

def test_mae_nonnegative():
    truth, forecast = _pair()
    assert mae(truth, forecast) >= 0.0


def test_mae_perfect():
    arr = RNG.standard_normal(SHAPE)
    assert mae(arr, arr) == pytest.approx(0.0, abs=1e-12)


def test_mae_le_rmse_by_cauchy_schwarz():
    # MAE ≤ RMSE always (Cauchy-Schwarz)
    truth, forecast = _pair()
    assert mae(truth, forecast) <= rmse(truth, forecast) + 1e-12


# --- bias ---

def test_bias_signed_positive():
    truth = np.zeros(SHAPE)
    forecast = np.ones(SHAPE)
    assert bias(truth, forecast) == pytest.approx(1.0)


def test_bias_signed_negative():
    truth = np.ones(SHAPE)
    forecast = np.zeros(SHAPE)
    assert bias(truth, forecast) == pytest.approx(-1.0)


def test_bias_perfect():
    arr = RNG.standard_normal(SHAPE)
    assert bias(arr, arr) == pytest.approx(0.0, abs=1e-12)


# --- anomaly_correlation ---

def test_acc_range():
    truth, forecast = _pair()
    acc = anomaly_correlation(truth, forecast)
    assert -1.0 <= acc <= 1.0


def test_acc_perfect():
    arr = RNG.standard_normal(SHAPE)
    assert anomaly_correlation(arr, arr) == pytest.approx(1.0, abs=1e-10)


def test_acc_anticorrelated():
    arr = RNG.standard_normal(SHAPE)
    assert anomaly_correlation(arr, -arr) == pytest.approx(-1.0, abs=1e-10)


def test_acc_constant_forecast_returns_zero():
    truth = RNG.standard_normal(SHAPE)
    forecast = np.full(SHAPE, 5.0)  # zero anomaly → denominator == 0
    assert anomaly_correlation(truth, forecast) == pytest.approx(0.0)


# --- summarize_forecast ---

def test_summarize_keys():
    truth, forecast = _pair()
    result = summarize_forecast(truth, forecast)
    assert set(result.keys()) == {"rmse", "mae", "bias", "anomaly_correlation"}


def test_summarize_all_floats():
    truth, forecast = _pair()
    result = summarize_forecast(truth, forecast)
    assert all(isinstance(v, float) for v in result.values())


# --- write_metrics_csv ---

def test_write_metrics_csv_columns(tmp_path):
    import csv as csv_mod
    from earth2_recipes.metrics import write_metrics_csv

    truth, forecast = _pair()
    metrics = summarize_forecast(truth, forecast)
    out = tmp_path / "metrics.csv"
    write_metrics_csv(out, metrics, variable="2m_temperature", baseline="persistence", units="K")

    rows = list(csv_mod.DictReader(out.open()))
    assert {r["metric"] for r in rows} == {"rmse", "mae", "bias", "anomaly_correlation"}
    assert all(r["variable"] == "2m_temperature" for r in rows)
    assert all(r["baseline"] == "persistence" for r in rows)


def test_write_metrics_csv_acc_is_dimensionless(tmp_path):
    import csv as csv_mod
    from earth2_recipes.metrics import write_metrics_csv

    truth, forecast = _pair()
    metrics = summarize_forecast(truth, forecast)
    out = tmp_path / "metrics.csv"
    write_metrics_csv(out, metrics, variable="z500", baseline="persistence", units="m2/s2")

    rows = {r["metric"]: r for r in csv_mod.DictReader(out.open())}
    assert rows["anomaly_correlation"]["units"] == "dimensionless"
    assert rows["rmse"]["units"] == "m2/s2"


def test_write_metrics_csv_creates_parent(tmp_path):
    from earth2_recipes.metrics import write_metrics_csv

    truth, forecast = _pair()
    metrics = summarize_forecast(truth, forecast)
    out = tmp_path / "nested" / "dir" / "metrics.csv"
    write_metrics_csv(out, metrics, variable="t850", baseline="persistence", units="K")
    assert out.exists()
