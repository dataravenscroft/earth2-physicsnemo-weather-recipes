#!/usr/bin/env python3
"""Run a lightweight deterministic forecast demo with plotting.

Data source priority:
  1. Manifest (manifests/era5_surface_manifest.jsonl) — uses two consecutive
     train records to build a persistence baseline: forecast(t) = truth(t-1).
  2. Synthetic fallback — generated in-memory if no manifest is available.

Run the full pipeline:
  python scripts/create_synthetic_era5.py
  python scripts/build_manifest.py
  python scripts/run_inference_demo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from earth2_recipes.metrics import summarize_forecast, write_metrics_csv
from earth2_recipes.plotting import plot_forecast_comparison
from earth2_recipes.utils import (
    ensure_directory,
    load_yaml,
    resolve_configured_path,
    set_random_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument(
        "--datapipe-config",
        default="conf/datapipe/era5_surface_manifest.yaml",
        help="Path to the datapipe YAML config (used to locate the manifest).",
    )
    return parser.parse_args()


def load_persistence_pair_from_manifest(
    manifest_path: Path,
    variable_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load truth/persistence pair from two consecutive train records.

    Returns (truth, persistence_forecast, description) or None if unavailable.
    truth             — field at time t+1,  shape (H, W)
    persistence_forecast — field at time t, shape (H, W)  (persistence baseline)
    """
    if not manifest_path.exists():
        return None

    records = []
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    train = [r for r in records if r["split"] == "train"]
    if len(train) < 2:
        return None

    try:
        field_t = np.load(train[0]["path"])   # (C, H, W)
        field_t1 = np.load(train[1]["path"])  # (C, H, W)
    except (FileNotFoundError, ValueError):
        return None

    truth = field_t1[variable_index].astype(np.float64)        # (H, W)
    persistence = field_t[variable_index].astype(np.float64)   # (H, W)
    desc = (
        f"manifest: {Path(train[0]['path']).name} → {Path(train[1]['path']).name}"
    )
    return truth, persistence, desc


def synthetic_truth_and_forecast(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a small, reproducible field pair for fallback demos."""
    rng = np.random.default_rng(seed)
    lon = np.linspace(0.0, 2.0 * np.pi, 48)
    lat = np.linspace(-1.0, 1.0, 24)
    xx, yy = np.meshgrid(lon, lat)

    initial = 280.0 + 8.0 * np.sin(xx) * np.cos(np.pi * yy)
    truth = initial + 1.5 * np.cos(2.0 * xx) - 0.5 * yy
    forecast = initial + rng.normal(loc=0.0, scale=0.6, size=initial.shape)
    return truth, forecast


def main() -> int:
    args = parse_args()
    inference_cfg = load_yaml(REPO_ROOT / args.inference_config)
    paths_cfg = load_yaml(REPO_ROOT / args.paths_config)
    datapipe_cfg = load_yaml(REPO_ROOT / args.datapipe_config)

    runtime_cfg = inference_cfg.get("runtime", {})
    forecast_cfg = inference_cfg.get("forecast", {})
    artifact_cfg = inference_cfg.get("artifacts", {})
    set_random_seed(int(runtime_cfg.get("seed", 7)))

    # Locate manifest written by build_manifest.py
    manifest_dir = resolve_configured_path(paths_cfg, "manifest_root", REPO_ROOT)
    manifest_path = manifest_dir / datapipe_cfg.get("manifest", {}).get(
        "output_filename", "era5_surface_manifest.jsonl"
    )

    result = load_persistence_pair_from_manifest(manifest_path)
    if result is not None:
        truth, forecast, data_desc = result
        print(f"Data source : {data_desc}")
        print("Forecast    : persistence baseline (predict t+1 = t)")
    else:
        truth, forecast = synthetic_truth_and_forecast(seed=int(runtime_cfg.get("seed", 7)))
        data_desc = "synthetic fallback"
        print("Data source : synthetic (run create_synthetic_era5.py + build_manifest.py for real data)")
        print("Forecast    : noisy synthetic field")

    summary = summarize_forecast(truth=truth, forecast=forecast)

    variable = forecast_cfg.get("output_field", "2m_temperature")
    lead = forecast_cfg.get("lead_hours", ["?"])[0]
    # Map the standard ERA5 surface variable names to display units.
    UNITS = {
        "2m_temperature": "K",
        "10m_u_component_of_wind": "m/s",
        "10m_v_component_of_wind": "m/s",
        "mean_sea_level_pressure": "Pa",
    }
    units = UNITS.get(variable, "")

    output_dir = resolve_configured_path(
        paths_cfg,
        artifact_cfg.get("output_dir_key", "artifact_root"),
        REPO_ROOT,
    )
    ensure_directory(output_dir)

    figure_path = output_dir / artifact_cfg.get("figure_name", "inference_demo.png")
    metrics_path = output_dir / artifact_cfg.get("metrics_name", "metrics_summary.csv")

    title = f"Persistence baseline — {variable} ({units}) at {lead}h lead"
    plot_forecast_comparison(
        truth=truth, forecast=forecast, output_path=figure_path, title=title, units=units
    )
    write_metrics_csv(
        output_path=metrics_path,
        metrics=summary,
        variable=variable,
        baseline="persistence",
        units=units,
    )

    print(f"Figure      : {figure_path.relative_to(REPO_ROOT)}")
    print(f"Metrics CSV : {metrics_path.relative_to(REPO_ROOT)}")
    print("Metrics:")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
