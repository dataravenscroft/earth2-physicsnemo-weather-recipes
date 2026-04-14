#!/usr/bin/env python3
"""Run a lightweight deterministic forecast demo with plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earth2_recipes.metrics import summarize_forecast
from earth2_recipes.plotting import plot_forecast_comparison
from earth2_recipes.utils import (
    ensure_directory,
    load_yaml,
    optional_dependency_available,
    resolve_configured_path,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-config",
        default="conf/inference/earth2studio_deterministic_demo.yaml",
        help="Path to the inference YAML config.",
    )
    parser.add_argument(
        "--paths-config",
        default="conf/paths/local.yaml",
        help="Path to the local paths YAML config.",
    )
    return parser.parse_args()


def synthetic_truth_and_forecast(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a small, reproducible field pair for a starter demo."""
    rng = np.random.default_rng(seed)
    lon = np.linspace(0.0, 2.0 * np.pi, 48)
    lat = np.linspace(-1.0, 1.0, 24)
    xx, yy = np.meshgrid(lon, lat)

    initial = 280.0 + 8.0 * np.sin(xx) * np.cos(np.pi * yy)
    truth = initial + 1.5 * np.cos(2.0 * xx) - 0.5 * yy
    forecast = initial + rng.normal(loc=0.0, scale=0.6, size=initial.shape)
    return initial, truth, forecast


def main() -> int:
    args = parse_args()
    inference_cfg = load_yaml(REPO_ROOT / args.inference_config)
    paths_cfg = load_yaml(REPO_ROOT / args.paths_config)

    runtime_cfg = inference_cfg.get("runtime", {})
    forecast_cfg = inference_cfg.get("forecast", {})
    artifact_cfg = inference_cfg.get("artifacts", {})
    set_random_seed(int(runtime_cfg.get("seed", 7)))

    _, truth, forecast = synthetic_truth_and_forecast(seed=int(runtime_cfg.get("seed", 7)))
    summary = summarize_forecast(truth=truth, forecast=forecast)

    output_dir = resolve_configured_path(
        paths_cfg,
        artifact_cfg.get("output_dir_key", "artifact_root"),
        REPO_ROOT,
    )
    ensure_directory(output_dir)
    figure_path = output_dir / artifact_cfg.get("figure_name", "inference_demo.png")

    title = (
        f"{inference_cfg.get('engine', {}).get('name', 'engine')} demo: "
        f"{forecast_cfg.get('output_field', 'field')} at "
        f"{forecast_cfg.get('lead_hours', ['?'])[0]}h lead"
    )
    plot_forecast_comparison(truth=truth, forecast=forecast, output_path=figure_path, title=title)

    print(f"Saved figure to {figure_path}")
    print("Forecast summary:")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    if optional_dependency_available("earth2studio"):
        print("TODO: Replace the synthetic fallback with a real Earth2Studio model call.")
    else:
        print("earth2studio is not installed; ran the synthetic starter demo instead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
