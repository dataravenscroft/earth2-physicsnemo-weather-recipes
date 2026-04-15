# data/

## What is committed here

`data/era5/.gitkeep` — keeps the directory tracked so the expected path
exists after `git clone`.

`data/era5/*.npy` files are **not** committed. Generate them locally:

```bash
python scripts/create_synthetic_era5.py
```

Each file stores shape `(4, H=24, W=48)` with physically-plausible surface
fields (2m temperature, 10m u/v wind, MSLP). Filenames follow the ERA5
naming convention `era5_surface_YYYYMMDDHH.npy` so the manifest builder
and timestamp parser behave identically to real data.

## What is NOT committed here

Real ERA5 data. When replacing the synthetic fixtures with real data:

1. Point `conf/paths/local.yaml` → `era5_root` at your local data directory.
2. Update `conf/datapipe/era5_surface_manifest.yaml` → `file_glob` to match
   your file format (e.g. `**/*.nc` for NetCDF).
3. Run `python scripts/build_manifest.py` to rebuild the manifest.

WeatherBench2 provides a public ERA5 zarr store suitable for this workflow.
