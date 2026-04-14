"""Tests for forecast model interface in earth2_recipes.model."""

from __future__ import annotations

import numpy as np
import pytest

from earth2_recipes.model import ForecastModel, NoisyPersistenceModel, PersistenceModel

RNG = np.random.default_rng(0)
SHAPE = (4, 24, 48)  # (C, H, W)


# --- interface ---

def test_persistence_is_forecastmodel():
    assert isinstance(PersistenceModel(), ForecastModel)


def test_noisy_is_forecastmodel():
    assert isinstance(NoisyPersistenceModel(), ForecastModel)


def test_forecastmodel_is_abstract():
    with pytest.raises(TypeError):
        ForecastModel()  # type: ignore[abstract]


# --- PersistenceModel ---

def test_persistence_output_shape():
    field = RNG.standard_normal(SHAPE)
    out = PersistenceModel().predict(field)
    assert out.shape == SHAPE


def test_persistence_is_copy_not_view():
    field = RNG.standard_normal(SHAPE)
    out = PersistenceModel().predict(field)
    out[0, 0, 0] += 999.0
    assert field[0, 0, 0] != out[0, 0, 0]


def test_persistence_values_equal():
    field = RNG.standard_normal(SHAPE)
    out = PersistenceModel().predict(field)
    np.testing.assert_array_equal(out, field)


# --- NoisyPersistenceModel ---

def test_noisy_output_shape():
    field = RNG.standard_normal(SHAPE)
    out = NoisyPersistenceModel(scale=1.0, seed=42).predict(field)
    assert out.shape == SHAPE


def test_noisy_differs_from_input():
    field = RNG.standard_normal(SHAPE)
    out = NoisyPersistenceModel(scale=1.0, seed=42).predict(field)
    assert not np.allclose(out, field)


def test_noisy_reproducible_with_same_seed():
    field = RNG.standard_normal(SHAPE)
    out_a = NoisyPersistenceModel(scale=1.0, seed=7).predict(field)
    out_b = NoisyPersistenceModel(scale=1.0, seed=7).predict(field)
    np.testing.assert_array_equal(out_a, out_b)


def test_noisy_scale_zero_equals_persistence():
    field = RNG.standard_normal(SHAPE)
    out = NoisyPersistenceModel(scale=0.0, seed=0).predict(field)
    np.testing.assert_allclose(out, field)
