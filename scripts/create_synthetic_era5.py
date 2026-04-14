#!/usr/bin/env python3
"""Create small synthetic ERA5-style .npy files for end-to-end demo runs.

Each file stores shape (4, H, W) with physically-plausible fields:
  C=0  2m_temperature        (K)
  C=1  10m_u_component_of_wind  (m/s)
  C=2  10m_v_component_of_wind  (m/s)
  C=3  mean_sea_level_pressure  (Pa)

Files are named era5_surface_YYYYMMDDHH.npy and placed under data/era5/
so that build_manifest.py picks them up with the default glob.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small grid — big enough to show spatial structure, fast to generate.
H, W = 24, 48  # lat x lon

# Timestamps spanning train / validation / test splits (matching default config).
TIMESTAMPS = [
    "2018010100",  # train
    "2018070100",  # train
    "2019010100",  # train
    "2019070100",  # train
    "2020010100",  # train  — anchor for lead-time pairs
    "2020010106",  # train  — +6h  from anchor
    "2020010112",  # train  — +12h from anchor
    "2020010200",  # train  — +24h from anchor
    "2021010100",  # validation
    "2022010100",  # test
]


def _base_grid() -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(0.0, 2.0 * np.pi, W)
    lat = np.linspace(-np.pi / 2, np.pi / 2, H)
    return np.meshgrid(lon, lat)  # (H, W) each


def make_field(timestamp: str, rng: np.random.Generator) -> np.ndarray:
    """Return shape (4, H, W) with small random perturbation per timestamp."""
    xx, yy = _base_grid()

    # Seed a small temporal offset from the timestamp digits so consecutive
    # frames differ slightly — mimics ERA5 6-hourly evolution.
    hour_offset = int(timestamp[-2:]) / 24.0

    t2m = (
        280.0
        + 20.0 * np.cos(yy)                          # latitudinal gradient
        + 5.0 * np.sin(xx + hour_offset)              # zonal wave
        + rng.normal(scale=0.5, size=(H, W))          # noise
    )
    u10 = (
        8.0 * np.cos(yy) * np.sin(xx + hour_offset)
        + rng.normal(scale=0.3, size=(H, W))
    )
    v10 = (
        3.0 * np.sin(2.0 * yy) * np.cos(xx)
        + rng.normal(scale=0.3, size=(H, W))
    )
    mslp = (
        101325.0
        + 1200.0 * np.sin(yy) * np.cos(xx + hour_offset)
        + rng.normal(scale=50.0, size=(H, W))
    )

    return np.stack([t2m, u10, v10, mslp]).astype(np.float32)  # (4, H, W)


def main() -> None:
    out_dir = REPO_ROOT / "data" / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for ts in TIMESTAMPS:
        field = make_field(ts, rng)
        path = out_dir / f"era5_surface_{ts}.npy"
        np.save(path, field)
        print(f"  wrote {path.relative_to(REPO_ROOT)}  shape={field.shape}")

    print(f"\n{len(TIMESTAMPS)} files in {out_dir.relative_to(REPO_ROOT)}/")
    print("Next: python scripts/build_manifest.py")


if __name__ == "__main__":
    main()
