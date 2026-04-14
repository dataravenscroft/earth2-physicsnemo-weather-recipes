"""Manifest helpers for ERA5-style weather datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    path: str
    split: str
    timestamp: str | None
    dataset: str
    variables: list[str]


def infer_timestamp_from_path(path: Path) -> str | None:
    """Best-effort timestamp inference from a filename stem."""
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    for fmt, width in (("%Y%m%d%H", 10), ("%Y%m%d", 8)):
        if len(digits) >= width:
            try:
                parsed = datetime.strptime(digits[:width], fmt)
                return parsed.isoformat()
            except ValueError:
                continue
    return None


def assign_split(timestamp: str | None, split_years: dict[str, list[int]]) -> str:
    """Assign a train/validation/test split using configured years."""
    if not timestamp:
        return "unassigned"

    year = datetime.fromisoformat(timestamp).year
    for split, years in split_years.items():
        if year in years:
            return split
    return "holdout"


def build_manifest_records(
    data_root: Path,
    file_glob: str,
    dataset_name: str,
    variables: list[str],
    split_years: dict[str, list[int]],
) -> list[ManifestRecord]:
    """Scan a directory tree and return manifest records."""
    records: list[ManifestRecord] = []
    for file_path in sorted(data_root.glob(file_glob)):
        if not file_path.is_file():
            continue

        timestamp = infer_timestamp_from_path(file_path)
        record = ManifestRecord(
            sample_id=file_path.stem,
            path=str(file_path.resolve()),
            split=assign_split(timestamp, split_years),
            timestamp=timestamp,
            dataset=dataset_name,
            variables=variables,
        )
        records.append(record)
    return records


def write_manifest_jsonl(output_path: Path, records: Iterable[ManifestRecord]) -> None:
    """Write manifest records as newline-delimited JSON."""
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")
