from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping

from .config import ScenarioConfig, normalize_scenario_name
from .surrogate import SurrogateParams


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def load_workflow_config(path: Path | None = None) -> dict[str, Any]:
    config = load_json(path or CONFIG_DIR / "eti_workflow.json")
    scenarios = [normalize_scenario_name(value) for value in config["scenario_order"]]
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("scenario_order contains duplicate canonical scenario names.")
    config["scenario_order"] = scenarios
    config["calibration_scenarios"] = [
        normalize_scenario_name(value) for value in config["calibration_scenarios"]
    ]
    config["validation_scenarios"] = [
        normalize_scenario_name(value) for value in config["validation_scenarios"]
    ]
    return config


def load_scenario_record(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    canonical = normalize_scenario_name(name)
    record = load_json((config_dir or CONFIG_DIR) / f"{canonical}.json")
    if normalize_scenario_name(record.get("name", "")) != canonical:
        raise ValueError(f"Scenario file name and record name disagree for {canonical!r}.")
    record["name"] = canonical
    return record


def load_scenario_config(name: str, config_dir: Path | None = None) -> ScenarioConfig:
    record = load_scenario_record(name, config_dir=config_dir)
    allowed = {field.name for field in fields(ScenarioConfig)}
    values = {key: value for key, value in record.items() if key in allowed}
    return ScenarioConfig(**values)


def load_all_scenario_records(
    workflow: Mapping[str, Any] | None = None,
    config_dir: Path | None = None,
) -> list[dict[str, Any]]:
    active = dict(workflow or load_workflow_config())
    return [load_scenario_record(name, config_dir=config_dir) for name in active["scenario_order"]]


def load_surrogate_parameter_record(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or CONFIG_DIR / "surrogate_parameters.json")


def load_surrogate_params(path: Path | None = None) -> SurrogateParams:
    record = load_surrogate_parameter_record(path)
    parameter_fields = {field.name for field in fields(SurrogateParams)}
    values: dict[str, Any] = {}
    for name, metadata in record.get("parameters", {}).items():
        if name not in parameter_fields:
            raise ValueError(f"Unknown surrogate parameter in configuration: {name}")
        values[name] = metadata["value"] if isinstance(metadata, dict) else metadata
    bounds = record.get("state_bounds", {})
    if "minimum" in bounds:
        values["min_vals"] = tuple(float(value) for value in bounds["minimum"])
    if "maximum" in bounds:
        values["max_vals"] = tuple(float(value) for value in bounds["maximum"])
    return SurrogateParams(**values)


def require_frozen_validation_parameters(path: Path | None = None) -> dict[str, Any]:
    record = load_surrogate_parameter_record(path)
    calibration_run = record.get("calibration_run")
    if (
        not bool(record.get("frozen_before_validation"))
        or not isinstance(calibration_run, dict)
        or bool(calibration_run.get("validation_results_used_for_tuning", True))
    ):
        raise RuntimeError(
            "The active surrogate parameters are not a completed baseline-only frozen calibration. "
            "Run scripts/run_baseline_calibration.py --activate before manuscript validation."
        )
    return record


def parameter_provenance_rows(path: Path | None = None) -> list[dict[str, Any]]:
    record = load_surrogate_parameter_record(path)
    rows: list[dict[str, Any]] = []
    for code_name, metadata in record.get("parameters", {}).items():
        row = dict(metadata)
        row["code_name"] = code_name
        row.setdefault("symbol", code_name)
        row.setdefault("description", "See surrogate equation definition.")
        row.setdefault("source_or_rationale", row.get("source_type", "TODO_source_required"))
        row.setdefault(
            "calibrated_or_fixed",
            "calibrated" if row.get("source_type") == "calibrated to baseline BSM1" else "fixed",
        )
        row.setdefault("calibration_dataset", record.get("calibration_dataset"))
        rows.append(row)
    return rows
