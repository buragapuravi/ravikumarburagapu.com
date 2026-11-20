from __future__ import annotations

import csv
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.config import ROOT

_EVENT_COLUMNS = {
    "record_id",
    "indicator",
    "value",
    "unit",
    "category",
    "event_name",
    "event_date",
    "organization",
    "crop",
    "state",
    "district",
    "source_id",
    "source_title",
    "source_url",
    "source_issued_date",
    "source_summary_path",
    "source_summary_sha256",
    "extraction_method",
    "page_reference",
    "temporal_scope",
    "caveat",
}
_APY_COLUMNS = {
    "record_id",
    "metric",
    "value",
    "unit",
    "crop",
    "state",
    "district",
    "season",
    "agricultural_year",
    "estimate_status",
    "source_id",
    "source_title",
    "source_url",
    "query_filters",
    "retrieved_at",
    "extraction_method",
    "source_record_sha256",
    "source_row_canonical",
    "temporal_scope",
    "caveat",
}
_APY_DISTRICT_COLUMNS = {
    "record_id",
    "crop",
    "state",
    "district",
    "season",
    "agricultural_year",
    "area_hectares",
    "production_tonnes",
    "yield_tonne_per_hectare",
    "estimate_status",
    "source_id",
    "source_title",
    "source_url",
    "query_filters",
    "retrieved_at",
    "extraction_method",
    "source_record_sha256",
    "source_row_canonical",
    "temporal_scope",
    "caveat",
}


def _read_csv(
    path: Path,
    required_columns: set[str],
    value_column: str = "value",
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"{path.name} is missing required structured-data columns.")
            rows = list(reader)
    except OSError as error:
        raise RuntimeError(f"Could not read structured dataset {path}: {error}") from error
    if not rows:
        raise ValueError(f"Structured dataset is empty: {path}")
    identifiers = [row["record_id"].strip() for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path.name} must contain non-empty, unique record_id values.")
    for row in rows:
        try:
            value = Decimal(row[value_column])
        except (InvalidOperation, KeyError) as error:
            raise ValueError(f"{path.name} contains a non-numeric value.") from error
        if not value.is_finite() or value < 0:
            raise ValueError(f"{path.name} values must be finite and non-negative.")
        if any(not row.get(column, "").strip() for column in required_columns - {"page_reference"}):
            raise ValueError(f"{path.name} contains an empty required provenance or dimension field.")
    return rows


def load_extension_indicators(
    path: Path | None = None,
) -> list[dict[str, str]]:
    dataset_path = path or ROOT / "data/structured/icar_extension_indicators.csv"
    rows = _read_csv(dataset_path, _EVENT_COLUMNS)
    project_root = ROOT.resolve()
    for row in rows:
        summary_path = (ROOT / row["source_summary_path"]).resolve()
        if not summary_path.is_relative_to(project_root):
            raise ValueError("A source summary path escapes the project data directory.")
        try:
            digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(f"Could not verify source summary {summary_path}: {error}") from error
        if digest != row["source_summary_sha256"]:
            raise ValueError(f"Source summary hash mismatch for record {row['record_id']}.")
    return rows


def load_des_apy_records(path: Path | None = None) -> list[dict[str, str]]:
    dataset_path = path or ROOT / "data/structured/des_apy_observations.csv"
    rows = _read_csv(dataset_path, _APY_COLUMNS)
    measures = {"area": 3, "production": 4, "yield": 5}
    groups: dict[tuple[str, str, str, str, str], dict[str, Decimal]] = {}
    for row in rows:
        digest = hashlib.sha256(row["source_row_canonical"].encode("utf-8")).hexdigest()
        if digest != row["source_record_sha256"]:
            raise ValueError(f"Source report row hash mismatch for record {row['record_id']}.")
        source_fields = row["source_row_canonical"].split("|")
        if len(source_fields) != 6 or row["metric"] not in measures:
            raise ValueError(f"Invalid DES report row schema for record {row['record_id']}.")
        reported_state = re.sub(r"^\d+\.\s*", "", source_fields[0]).casefold()
        reported_district = re.sub(r"^\d+\.\s*", "", source_fields[1]).casefold()
        reported_year = source_fields[2].replace(" ", "").casefold()
        if (
            reported_state != row["state"].casefold()
            or reported_district != row["district"].casefold()
            or reported_year != row["agricultural_year"].replace(" ", "").casefold()
        ):
            raise ValueError(f"DES report dimensions do not match record {row['record_id']}.")
        try:
            reported_value = Decimal(source_fields[measures[row["metric"]]].replace(",", ""))
            normalized_value = Decimal(row["value"])
        except InvalidOperation as error:
            raise ValueError(f"Invalid DES numeric value in record {row['record_id']}.") from error
        if reported_value != normalized_value:
            raise ValueError(f"DES report value does not match record {row['record_id']}.")
        group_key = (
            row["state"],
            row["district"],
            row["crop"],
            row["season"],
            row["agricultural_year"],
        )
        group = groups.setdefault(group_key, {})
        if row["metric"] in group:
            raise ValueError(f"Duplicate DES measure {row['metric']} for {group_key}.")
        group[row["metric"]] = normalized_value
    for dimensions, group in groups.items():
        if set(group) != set(measures):
            raise ValueError(f"DES record group is missing area, production, or yield: {dimensions}.")
        calculated_yield = group["production"] / group["area"]
        if abs(calculated_yield - group["yield"]) > Decimal("0.01"):
            raise ValueError(f"DES yield does not reconcile with area and production: {dimensions}.")
    return rows


def load_des_district_records(path: Path | None = None) -> list[dict[str, str]]:
    dataset_path = path or ROOT / "data/structured/des_punjab_wheat_districts_2022_2023.csv"
    rows = _read_csv(dataset_path, _APY_DISTRICT_COLUMNS, "area_hectares")
    districts: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        canonical = row["source_row_canonical"]
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != row["source_record_sha256"]:
            raise ValueError(f"DES district row hash mismatch for record {row['record_id']}.")
        fields = canonical.split("|")
        if len(fields) != 6:
            raise ValueError(f"Invalid DES district row schema for record {row['record_id']}.")
        reported_state = re.sub(r"^\d+\.\s*", "", fields[0]).casefold()
        reported_district = re.sub(r"^\d+\.\s*", "", fields[1]).casefold()
        if (
            reported_state != row["state"].casefold()
            or reported_district != row["district"].casefold()
            or fields[2].replace(" ", "").casefold()
            != row["agricultural_year"].replace(" ", "").casefold()
        ):
            raise ValueError(f"DES report dimensions do not match record {row['record_id']}.")
        try:
            area, production, reported_yield = (
                Decimal(row["area_hectares"]),
                Decimal(row["production_tonnes"]),
                Decimal(row["yield_tonne_per_hectare"]),
            )
            source_values = [
                Decimal(value.replace(",", "")) for value in fields[3:]
            ]
        except InvalidOperation as error:
            raise ValueError(f"Invalid DES district measure in record {row['record_id']}.") from error
        if source_values != [area, production, reported_yield]:
            raise ValueError(f"DES district values do not match the source row for {row['record_id']}.")
        if any(not value.is_finite() or value < 0 for value in (area, production, reported_yield)):
            raise ValueError(f"DES district measures must be finite and non-negative: {row['record_id']}.")
        dimensions = (
            row["state"].casefold(),
            row["district"].casefold(),
            row["crop"].casefold(),
            row["season"].casefold(),
            row["agricultural_year"].casefold(),
        )
        if dimensions in districts:
            raise ValueError(f"Duplicate DES district dimensions: {dimensions}.")
        districts.add(dimensions)
        if area <= 0 or abs(production / area - reported_yield) > Decimal("0.01"):
            raise ValueError(f"DES yield does not reconcile with area and production: {dimensions}.")
    return rows


def filter_records(
    rows: list[dict[str, str]],
    **filters: str,
) -> list[dict[str, str]]:
    unknown = set(filters) - (set(rows[0]) if rows else set())
    if unknown:
        raise ValueError(f"Unsupported structured-data filters: {', '.join(sorted(unknown))}")
    normalized = {key: value.strip().casefold() for key, value in filters.items() if value.strip()}
    return [
        row
        for row in rows
        if all(row.get(key, "").strip().casefold() == value for key, value in normalized.items())
    ]


def event_attendance_summary(
    rows: list[dict[str, str]],
    event_name: str,
) -> dict[str, Any]:
    event_rows = filter_records(rows, event_name=event_name)
    if not event_rows:
        raise ValueError(f"No structured indicators found for event: {event_name}")
    values = {row["category"]: Decimal(row["value"]) for row in event_rows}
    if "total" not in values:
        raise ValueError(f"No source-reported total is available for event: {event_name}")
    if values["total"] <= 0 or len(values) != len(event_rows):
        raise ValueError(f"Attendance categories must be unique and totals positive for event: {event_name}")
    summary: dict[str, Any] = {
        "event_name": event_name,
        "participants_total": int(values["total"]),
        "source_id": event_rows[0]["source_id"],
        "source_title": event_rows[0]["source_title"],
        "source_url": event_rows[0]["source_url"],
        "caveat": event_rows[0]["caveat"],
    }
    if "men" in values and "women" in values:
        category_total = values["men"] + values["women"]
        if category_total != values["total"]:
            raise ValueError(f"Participant categories do not reconcile for event: {event_name}")
        summary["participants_men_reported"] = int(values["men"])
        summary["participants_women_reported"] = int(values["women"])
        summary["women_share_percent"] = round(
            float(values["women"] / values["total"] * 100),
            1,
        )
    return summary


def source_registry(path: Path | None = None) -> list[dict[str, str]]:
    registry_path = path or ROOT / "data/sources/source_registry.json"
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read official-source registry {registry_path}: {error}") from error
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("The official-source registry must contain a sources list.")
    required = {"source_id", "organization", "title", "url", "modality", "status"}
    for item in sources:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("Every official-source registry entry must include provenance fields.")
    return sources
