"""Forecast model interface for weather recipe demos.

The ABC defines the seam where a real model plugs in.  Two stub
implementations ship with the repo:

  PersistenceModel      — predict(x) = x  (the baseline to beat)
  NoisyPersistenceModel — predict(x) = x + noise  (for testing error flow)

A PhysicsNeMo or Earth2Studio model slots in by subclassing ForecastModel
and implementing predict().  The demo script needs nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ForecastModel(ABC):
    """Minimal interface for a one-step deterministic forecast model."""

    @abstractmethod
    def predict(self, field: np.ndarray) -> np.ndarray:
        """Return a one-step forecast given an initial condition.

        Args:
            field: Initial condition, shape (C, H, W).

        Returns:
            Predicted field at t+1, shape (C, H, W).
        """


class PersistenceModel(ForecastModel):
    """Persistence baseline: predict(x) = x.

    The simplest possible forecast — assume nothing changes.
    Every trained model should beat this on RMSE within a few epochs.
    """

    def predict(self, field: np.ndarray) -> np.ndarray:
        return field.copy()  # (C, H, W)


class NoisyPersistenceModel(ForecastModel):
    """Persistence plus Gaussian noise — used in tests to verify error propagation.

    Not a meaningful forecast; useful for confirming that non-zero RMSE
    flows correctly through the evaluation pipeline.
    """

    def __init__(self, scale: float = 1.0, seed: int = 0) -> None:
        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def predict(self, field: np.ndarray) -> np.ndarray:
        return field + self._rng.normal(scale=self._scale, size=field.shape)  # (C, H, W)
