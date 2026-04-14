#!/usr/bin/env python3
"""Build a simple ERA5-style JSONL manifest from local files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earth2_recipes.manifests import build_manifest_records, write_manifest_jsonl
from earth2_recipes.utils import ensure_directory, load_yaml, resolve_configured_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datapipe-config",
        default="conf/datapipe/era5_surface_manifest.yaml",
        help="Path to the datapipe YAML config.",
    )
    parser.add_argument(
        "--paths-config",
        default="conf/paths/local.yaml",
        help="Path to the local paths YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datapipe_cfg = load_yaml(REPO_ROOT / args.datapipe_config)
    paths_cfg = load_yaml(REPO_ROOT / args.paths_config)

    source_cfg = datapipe_cfg.get("source", {})
    manifest_cfg = datapipe_cfg.get("manifest", {})
    time_cfg = datapipe_cfg.get("time", {})

    data_root_key = source_cfg.get("data_root_key", "era5_root")
    file_glob = source_cfg.get("file_glob", "**/*.nc")
    allow_empty = bool(source_cfg.get("allow_empty", False))
    data_root = resolve_configured_path(paths_cfg, data_root_key, REPO_ROOT)

    output_dir_key = manifest_cfg.get("output_dir_key", "manifest_root")
    output_dir = resolve_configured_path(paths_cfg, output_dir_key, REPO_ROOT)
    output_path = output_dir / manifest_cfg.get("output_filename", "manifest.jsonl")
    ensure_directory(output_dir)

    records = build_manifest_records(
        data_root=data_root,
        file_glob=file_glob,
        dataset_name=datapipe_cfg.get("dataset", {}).get("name", "era5"),
        variables=datapipe_cfg.get("variables", []),
        split_years={
            "train": time_cfg.get("train_years", []),
            "validation": time_cfg.get("validation_years", []),
            "test": time_cfg.get("test_years", []),
        },
    )

    if not records and not allow_empty:
        raise FileNotFoundError(
            f"No files matched {file_glob!r} under {data_root}. "
            "Set source.allow_empty=true to write an empty starter manifest."
        )

    write_manifest_jsonl(output_path, records)

    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Dataset root: {data_root}")
    print(f"Variables: {', '.join(datapipe_cfg.get('variables', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
