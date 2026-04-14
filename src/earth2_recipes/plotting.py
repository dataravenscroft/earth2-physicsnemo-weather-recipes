"""Plotting helpers for forecast comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_forecast_comparison(
    truth: np.ndarray,
    forecast: np.ndarray,
    output_path: Path,
    title: str,
    units: str = "",
) -> None:
    """Save a compact truth/forecast/error comparison figure.

    Args:
        truth: Ground-truth field, shape (H, W).
        forecast: Predicted field, shape (H, W).
        output_path: Where to write the PNG.
        title: Figure suptitle.
        units: Physical units string shown on colorbars (e.g. "K", "m/s").
    """
    error = forecast - truth
    rmse_val = float(np.sqrt(np.mean(error**2)))
    error_label = f"Error  (RMSE {rmse_val:.3f} {units})".strip()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), constrained_layout=True)
    panels = [
        (truth,    "Truth",       "viridis", units),
        (forecast, "Forecast",    "viridis", units),
        (error,    error_label,   "coolwarm", f"Δ{units}" if units else "Δ"),
    ]

    for ax, (field, label, cmap, cb_label) in zip(axes, panels, strict=True):
        image = ax.imshow(field, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(image, ax=ax, shrink=0.8)
        if cb_label:
            cb.set_label(cb_label, fontsize=8)

    fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
