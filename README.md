# earth2-physicsnemo-weather-recipes

`earth2-physicsnemo-weather-recipes` is a lightweight portfolio starter for weather and climate machine learning workflows built around the NVIDIA Earth-2 and PhysicsNeMo ecosystem.

This repository is intentionally separate from custom end-to-end data engineering work. The goal here is to show an NVIDIA-native, recipe-style project layout that is easy to understand, easy to extend, and realistic enough to support:

- ERA5-oriented dataset manifest generation
- config-driven datapipe and experiment definitions
- a path toward PhysicsNeMo training recipes
- Earth2Studio pretrained-model inference demos
- benchmark-style forecast evaluation and plotting

The first version is deliberately small. It focuses on project shape, naming, and portfolio clarity rather than trying to clone an upstream framework.

## Why This Repo Exists

This project is designed to answer a simple portfolio question:

How would a clean, standalone weather ML starter look if it were aligned with the Earth-2 / PhysicsNeMo ecosystem from day one?

It is not:

- a fork of `physicsnemo`
- a replacement for upstream NVIDIA tooling
- a generic climate data pipeline repo
- a giant experiment framework

Instead, it is a compact scaffold for recipes, configs, manifests, demos, and evaluation utilities that can evolve alongside upstream NVIDIA tools.

## Project Principles

- `Earth-2 aligned`: naming and workflow shape reflect the NVIDIA weather ecosystem
- `Recipe-first`: configs describe experiments, datapipes, training, and inference separately
- `Portfolio-ready`: the structure is easy to explain in a review, interview, or demo
- `Lightweight`: only a small amount of starter code is included
- `Upstream-aware`: `physicsnemo/` is treated as a local reference, not as the place to add custom code

## Repository Layout

```text
earth2-physicsnemo-weather-recipes/
├── conf/
│   ├── datapipe/
│   ├── experiment/
│   ├── inference/
│   ├── paths/
│   └── training/
├── docs/
├── scripts/
└── src/earth2_recipes/
```

## Included in This Starter

### Configs

The `conf/` tree separates concerns so an experiment can reference independent path, datapipe, training, and inference configs. This keeps the repo close to a recipe-driven workflow without introducing heavyweight orchestration too early.

### Manifest Builder

`scripts/build_manifest.py` scans a local ERA5-style data directory using glob patterns from YAML config and writes a JSONL manifest. This is the seam where a more formal datapipe or catalog integration can grow later.

### Inference Demo

`scripts/run_inference_demo.py` creates a minimal deterministic forecast demo. In the starter version, it defaults to a synthetic persistence-style baseline so the repo remains runnable before any Earth2Studio dependency is fully integrated.

### Evaluation Utilities

The library includes simple forecast metrics and plotting helpers for side-by-side benchmark-style comparisons. These are intentionally small, readable, and easy to swap out once a stronger evaluation stack is in place.

## Quickstart

Create and activate an environment, then install the project in editable mode:

```bash
cd earth2-physicsnemo-weather-recipes
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Build a manifest from local ERA5-style files:

```bash
python scripts/build_manifest.py \
  --datapipe-config conf/datapipe/era5_surface_manifest.yaml \
  --paths-config conf/paths/local.yaml
```

Run the starter inference demo:

```bash
python scripts/run_inference_demo.py \
  --inference-config conf/inference/earth2studio_deterministic_demo.yaml \
  --paths-config conf/paths/local.yaml
```

## Roadmap Themes

- add a richer ERA5 manifest schema with variable- and lead-time-aware metadata
- connect configs to real PhysicsNeMo training recipes
- wire Earth2Studio pretrained inference into the demo path
- expand evaluation to support multi-lead benchmarks and skill scores
- add notebook or report artifacts for polished portfolio presentation

See [docs/ROADMAP.md](/Users/catherineravenscroft/git_personal/climate_biogeochem/earth2-physicsnemo-weather-recipes/docs/ROADMAP.md) for the next steps.

## Notes on Upstream References

This repository assumes local access to upstream tooling for reference and future integration work, but it does not place custom code inside those upstream repos.

- `physicsnemo/` remains an upstream reference clone
- this repo is the standalone place for portfolio-ready Earth-2 / PhysicsNeMo recipe work

## Current Status

This is a starter scaffold, not a finished benchmark suite. The current files are structured to be credible, readable, and easy to extend, with clear TODO markers where deeper Earth2Studio and PhysicsNeMo integration should happen next.
