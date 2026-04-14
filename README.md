# earth2-physicsnemo-weather-recipes

A lightweight, runnable weather ML pipeline aligned with the NVIDIA Earth-2
and PhysicsNeMo ecosystem. Covers ERA5-style data ingest, JSONL manifest
generation, a pluggable forecast model interface, and benchmark-quality
evaluation artifacts.

---

## What this demonstrates

- **ERA5-style manifest pipeline** — directory scan with year-based
  train / validation / test splitting, written as config-driven JSONL
- **Pluggable model interface** — `ForecastModel` ABC with `predict(field) -> field`;
  `PersistenceModel` ships as the baseline; a PhysicsNeMo or Earth2Studio
  model drops in by subclassing
- **Persistence baseline benchmark** — RMSE, MAE, bias, and ACC saved to
  `artifacts/metrics_summary.csv` with physical units; the reference a
  trained model must beat
- **Polished figure artifact** — three-panel truth / forecast / error PNG
  with colorbars in physical units and RMSE annotated on the error panel
- **Test suite** — 42 tests covering metric properties, manifest roundtrip,
  timestamp parsing, and model contract enforcement

---

## Pipeline

```
create_synthetic_era5.py
        │
        ▼  data/era5/*.npy  (4 vars × 24 × 48, train/val/test years)
        │
build_manifest.py
        │
        ▼  manifests/era5_surface_manifest.jsonl  (8 records, split by year)
        │
run_inference_demo.py
        │
        ├──▶  artifacts/inference_demo.png      (truth / forecast / error)
        └──▶  artifacts/metrics_summary.csv     (persistence baseline)
```

---

## Quickstart

```bash
pip install -e ".[dev]"
```

Run the full pipeline:

```bash
python scripts/create_synthetic_era5.py   # generate ERA5-style demo data
python scripts/build_manifest.py          # scan → JSONL manifest
python scripts/run_inference_demo.py      # persistence baseline → artifacts
```

Expected output:

```
Data source : manifest: era5_surface_2018010100.npy → era5_surface_2018070100.npy
Model       : PersistenceModel
Figure      : artifacts/inference_demo.png
Metrics CSV : artifacts/metrics_summary.csv
Metrics:
  rmse: 0.6955
  mae: 0.5493
  bias: 0.0024
  anomaly_correlation: 0.9957
```

Run tests:

```bash
pytest tests/ -q
# 42 passed in 0.09s
```

---

## Persistence baseline (2m_temperature)

| metric | value | units | baseline |
|---|---|---|---|
| RMSE | 0.6955 | K | persistence |
| MAE | 0.5493 | K | persistence |
| bias | 0.0024 | K | persistence |
| ACC | 0.9957 | dimensionless | persistence |

On real ERA5 data the persistence RMSE for 2m temperature at 6h lead is
typically 1–2 K; a well-trained model reduces that by 30–50%.

---

## Where a real model plugs in

`src/earth2_recipes/model.py` defines a one-method ABC:

```python
class ForecastModel(ABC):
    @abstractmethod
    def predict(self, field: np.ndarray) -> np.ndarray:
        """(C, H, W) -> (C, H, W)"""
```

Swap the model in `run_inference_demo.py` at the single instantiation point:

```python
model = PersistenceModel()        # current default
# model = MyPhysicsNeMoModel()   # drop-in replacement
```

Everything downstream — manifest loading, metric computation, figure and
CSV writing — is model-agnostic.

---

## Repository layout

```
earth2-physicsnemo-weather-recipes/
├── conf/
│   ├── datapipe/   era5_surface_manifest.yaml
│   ├── experiment/ era5_surface_demo.yaml
│   ├── inference/  earth2studio_deterministic_demo.yaml
│   ├── paths/      local.yaml
│   └── training/   fcn_era5_baseline.yaml
├── data/
│   └── era5/       era5_surface_YYYYMMDDHH.npy  (synthetic demo data, committed)
├── docs/
│   └── ROADMAP.md
├── scripts/
│   ├── create_synthetic_era5.py
│   ├── build_manifest.py
│   └── run_inference_demo.py
├── src/earth2_recipes/
│   ├── model.py     ForecastModel ABC, PersistenceModel, NoisyPersistenceModel
│   ├── metrics.py   rmse, mae, bias, anomaly_correlation, write_metrics_csv
│   ├── manifests.py ManifestRecord, build_manifest_records, write_manifest_jsonl
│   ├── plotting.py  plot_forecast_comparison
│   └── utils.py     load_yaml, resolve_configured_path, set_random_seed
└── tests/
    ├── test_metrics.py
    ├── test_manifests.py
    └── test_model.py
```

---

## Why this repo exists

This project answers a portfolio question: how would a clean, standalone
weather ML starter look if it were aligned with the Earth-2 / PhysicsNeMo
ecosystem from day one?

It is not a fork of `physicsnemo`, a replacement for upstream NVIDIA tooling,
or a giant experiment framework. It is a compact scaffold for recipes,
configs, manifests, demos, and evaluation utilities that can evolve alongside
upstream NVIDIA tools.

**Project principles**

- *Earth-2 aligned* — naming and workflow shape reflect the NVIDIA weather ecosystem
- *Recipe-first* — configs describe experiments, datapipes, training, and inference separately
- *Upstream-aware* — `physicsnemo/` is a local reference, not the place to add custom code
- *Portfolio-ready* — structure is easy to explain in a review, interview, or demo

---

## Roadmap

- Connect configs to real PhysicsNeMo training recipes
- Wire Earth2Studio pretrained inference into the demo path
- Expand `ForecastModel.predict` signature with `variables` and `timestamp`
  to match the Earth2Studio native call convention
- Multi-lead evaluation (6h, 12h, 24h) with ACC on Z500 as the primary metric

See [docs/ROADMAP.md](docs/ROADMAP.md) for details.
