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
) -> None:
    """Save a compact truth/forecast/error comparison figure."""
    error = forecast - truth

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), constrained_layout=True)
    panels = [
        (truth, "Truth", "viridis"),
        (forecast, "Forecast", "viridis"),
        (error, "Error", "coolwarm"),
    ]

    for ax, (field, label, cmap) in zip(axes, panels, strict=True):
        image = ax.imshow(field, cmap=cmap)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.8)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
