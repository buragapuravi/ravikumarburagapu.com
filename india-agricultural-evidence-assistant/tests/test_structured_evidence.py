from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.structured_evidence import (
    event_attendance_summary,
    filter_records,
    load_des_apy_records,
    load_des_district_records,
    load_extension_indicators,
    source_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_extension_indicators_have_verified_summary_hashes_and_reconcile():
    rows = load_extension_indicators()
    bloomesdale = event_attendance_summary(
        rows,
        "ICAR-CIARI Bloomsdale rice field day",
    )
    assert bloomesdale["participants_total"] == 43
    assert bloomesdale["participants_women_reported"] == 32
    assert bloomesdale["participants_men_reported"] == 11
    assert bloomesdale["women_share_percent"] == 74.4
    assert bloomesdale["source_id"] == "icar-ciari-bloomsdale-rice-field-day-2026"


def test_structured_data_filters_match_exact_dimensions():
    rows = load_extension_indicators()
    matches = filter_records(
        rows,
        state="meghalaya",
        crop="pulses",
        indicator="participating_villages",
    )
    assert len(matches) == 1
    assert matches[0]["value"] == "5"


def test_des_apy_records_preserve_dimensions_units_and_row_hash():
    records = load_des_apy_records()
    wheat = filter_records(
        records,
        state="Punjab",
        district="Ludhiana",
        crop="Wheat",
        season="Rabi",
    )
    assert {row["metric"] for row in wheat} == {"area", "production", "yield"}
    assert {row["agricultural_year"] for row in wheat} == {"2022-2023"}
    assert next(row for row in wheat if row["metric"] == "yield")["value"] == "4.59"
    assert all(row["source_record_sha256"] for row in wheat)


def test_des_apy_records_reject_modified_source_row_hash(tmp_path):
    source = PROJECT_ROOT / "data/structured/des_apy_observations.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[0]["source_row_canonical"] += "|modified"
    target = tmp_path / "des.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Source report row hash mismatch"):
        load_des_apy_records(target)


def test_des_apy_records_reject_values_that_disagree_with_reported_row(tmp_path):
    source = PROJECT_ROOT / "data/structured/des_apy_observations.csv"
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows[0]["value"] = "244701"
    target = tmp_path / "des.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="DES report value"):
        load_des_apy_records(target)


def test_structured_registry_distinguishes_ingested_and_candidate_sources():
    registry = source_registry()
    des = next(item for item in registry if item["source_id"] == "des-apy-portal")
    candidate = next(item for item in registry if item["source_id"] == "icar-crida-contingency-plans")
    assert des["status"] == "partially-imported"
    assert candidate["status"] == "candidate-not-ingested"


def test_des_district_extract_contains_verified_punjab_wheat_comparison_rows():
    rows = load_des_district_records()
    assert len(rows) == 23
    ludhiana = next(row for row in rows if row["district"] == "Ludhiana")
    assert ludhiana["yield_tonne_per_hectare"] == "4.59"
    assert all(len(row["source_record_sha256"]) == 64 for row in rows)


def test_des_district_extract_rejects_modified_source_row(tmp_path):
    source = PROJECT_ROOT / "data/structured/des_punjab_wheat_districts_2022_2023.csv"
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows[0]["source_row_canonical"] += "|modified"
    target = tmp_path / "districts.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="DES district row hash mismatch"):
        load_des_district_records(target)
