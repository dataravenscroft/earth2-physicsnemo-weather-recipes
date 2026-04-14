# Roadmap

## Purpose

This roadmap keeps the repository focused on a clean Earth-2 / PhysicsNeMo portfolio narrative:

1. data manifests and datapipes
2. recipe-style training and inference configs
3. benchmark-oriented evaluation
4. optional upstream integration when the project is ready

## Near-Term

### 1. Strengthen ERA5 manifest generation

- add variable-level metadata, pressure levels, and temporal coverage summaries
- support train, validation, and test split logic driven by config
- add validation checks for missing timestamps and duplicated samples

### 2. Improve datapipe realism

- represent common ERA5 access patterns such as rolling windows and lead times
- document expected tensor shapes and channel conventions
- align config names more closely with PhysicsNeMo recipe terminology

### 3. Make the inference path Earth2Studio-aware

- replace the synthetic baseline fallback with a real pretrained-model demo
- add configuration for model name, initial conditions, and lead time schedules
- capture outputs in a reproducible artifact directory

## Mid-Term

### 4. Add benchmark-style evaluation recipes

- support multiple deterministic metrics across forecast horizons
- compare a persistence baseline against model outputs
- save tabular summaries and figure assets for portfolio presentation

### 5. Introduce a first training recipe wrapper

- connect the config tree to an actual PhysicsNeMo training entrypoint
- keep the repo thin by wrapping upstream tools rather than copying internals
- document what is repo-owned versus upstream-owned

## Longer-Term

### 6. Add polished demonstration assets

- notebook or report views for forecast case studies
- example experiment cards summarizing inputs, configs, and outcomes
- optional CI checks for config validity and small unit tests

## Non-Goals for Early Versions

- cloning large pieces of upstream PhysicsNeMo
- building a full orchestration framework before a few strong recipes exist
- mixing this repository with unrelated custom pipeline code
