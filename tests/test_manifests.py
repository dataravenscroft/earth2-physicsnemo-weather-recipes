"""Tests for manifest helpers in earth2_recipes.manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from earth2_recipes.manifests import (
    ManifestRecord,
    assign_split,
    build_manifest_records,
    infer_timestamp_from_path,
    write_manifest_jsonl,
)


SPLIT_YEARS = {
    "train": [2018, 2019, 2020],
    "validation": [2021],
    "test": [2022],
}


# --- infer_timestamp_from_path ---

def test_timestamp_yyyymmddhh():
    p = Path("era5_2020010106.nc")
    assert infer_timestamp_from_path(p) == "2020-01-01T06:00:00"


def test_timestamp_yyyymmdd():
    p = Path("era5_20200101.nc")
    assert infer_timestamp_from_path(p) == "2020-01-01T00:00:00"


def test_timestamp_no_digits_returns_none():
    p = Path("readme.nc")
    assert infer_timestamp_from_path(p) is None


def test_timestamp_short_digits_returns_none():
    p = Path("v42.nc")
    assert infer_timestamp_from_path(p) is None


# --- assign_split ---

def test_assign_train():
    assert assign_split("2019-07-01T00:00:00", SPLIT_YEARS) == "train"


def test_assign_validation():
    assert assign_split("2021-01-01T00:00:00", SPLIT_YEARS) == "validation"


def test_assign_test():
    assert assign_split("2022-06-15T00:00:00", SPLIT_YEARS) == "test"


def test_assign_holdout():
    assert assign_split("2023-01-01T00:00:00", SPLIT_YEARS) == "holdout"


def test_assign_none_timestamp():
    assert assign_split(None, SPLIT_YEARS) == "unassigned"


# --- build_manifest_records ---

def test_build_manifest_finds_files(tmp_path: Path):
    (tmp_path / "era5_2020010100.nc").touch()
    (tmp_path / "era5_2021010100.nc").touch()

    records = build_manifest_records(
        data_root=tmp_path,
        file_glob="*.nc",
        dataset_name="era5_test",
        variables=["z500", "t850"],
        split_years=SPLIT_YEARS,
    )

    assert len(records) == 2
    splits = {r.split for r in records}
    assert splits == {"train", "validation"}


def test_build_manifest_empty_dir(tmp_path: Path):
    records = build_manifest_records(
        data_root=tmp_path,
        file_glob="*.nc",
        dataset_name="era5_test",
        variables=[],
        split_years=SPLIT_YEARS,
    )
    assert records == []


def test_build_manifest_record_fields(tmp_path: Path):
    (tmp_path / "era5_2022060100.nc").touch()
    records = build_manifest_records(
        data_root=tmp_path,
        file_glob="*.nc",
        dataset_name="era5_test",
        variables=["z500"],
        split_years=SPLIT_YEARS,
    )
    r = records[0]
    assert r.dataset == "era5_test"
    assert r.variables == ["z500"]
    assert r.split == "test"
    assert r.timestamp == "2022-06-01T00:00:00"
    assert r.sample_id == "era5_2022060100"


# --- write_manifest_jsonl ---

def test_write_manifest_roundtrip(tmp_path: Path):
    record = ManifestRecord(
        sample_id="era5_2020010100",
        path="/data/era5_2020010100.nc",
        split="train",
        timestamp="2020-01-01T00:00:00",
        dataset="era5_test",
        variables=["z500", "t850"],
    )
    out = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(out, [record])

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["sample_id"] == "era5_2020010100"
    assert parsed["split"] == "train"
    assert parsed["variables"] == ["z500", "t850"]


def test_write_manifest_empty(tmp_path: Path):
    out = tmp_path / "empty.jsonl"
    write_manifest_jsonl(out, [])
    assert out.read_text() == ""
