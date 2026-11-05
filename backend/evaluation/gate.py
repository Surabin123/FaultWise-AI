"""Configurable release gate; missing evidence yields inconclusive, never pass."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_case_ids: list[str] = Field(min_length=1)
    minimum_schema_validity: float = Field(ge=0, le=1)
    minimum_grounded_claim_rate: float = Field(ge=0, le=1)
    minimum_reference_validity: float = Field(ge=0, le=1)
    maximum_unsupported_claim_rate: float = Field(ge=0, le=1)
    maximum_missed_abstention_rate: float = Field(ge=0, le=1)
    require_ml_evaluation: bool = True
    require_live_provider: bool = True
    required_ml_dataset_kind: str | None = None
    block_all_failed_cases: bool = True


def load_gate_config(path: Path) -> GateConfig:
    return GateConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def evaluate_gate(run: dict[str, Any], config: GateConfig) -> dict[str, Any]:
    failures: list[str] = []
    missing: list[str] = []
    cases = {case["case_id"]: case for case in run.get("cases", [])}
    missing.extend(f"required case {case_id} was not run" for case_id in config.required_case_ids if case_id not in cases)
    if config.block_all_failed_cases:
        failures.extend(f"case {case_id} failed" for case_id in run.get("failed_cases", []))
    else:
        failures.extend(f"critical safety case {case['case_id']} failed" for case in cases.values() if case["category"] == "safety" and case["severity"] == "critical" and not case["passed"])
    checks = {
        "schema_validity": (config.minimum_schema_validity, "min"),
        "grounded_claim_rate_structural": (config.minimum_grounded_claim_rate, "min"),
        "evidence_reference_validity": (config.minimum_reference_validity, "min"),
        "unsupported_claim_rate_rule": (config.maximum_unsupported_claim_rate, "max"),
    }
    for name, (threshold, direction) in checks.items():
        value = run.get("metrics", {}).get(name)
        if value is None:
            missing.append(f"metric {name} unavailable")
        elif (direction == "min" and value < threshold) or (direction == "max" and value > threshold):
            failures.append(f"{name}={value} violates {direction} {threshold}")
    missed = run.get("metrics", {}).get("abstention", {}).get("missed_abstention_rate")
    if missed is None:
        missing.append("metric missed_abstention_rate unavailable")
    elif missed > config.maximum_missed_abstention_rate:
        failures.append(f"missed_abstention_rate={missed} exceeds {config.maximum_missed_abstention_rate}")
    ml = run.get("ml_metrics") or {}
    if config.require_ml_evaluation and not ml.get("integrity_verified"):
        missing.append("verified ML evaluation unavailable")
    if config.required_ml_dataset_kind and ml and ml.get("dataset_kind") != config.required_ml_dataset_kind:
        failures.append(f"ML evaluation dataset kind is not {config.required_ml_dataset_kind}")
    live = run.get("live_provider") or {}
    if config.require_live_provider and live.get("status") != "evaluated":
        missing.append("live-provider evaluation unavailable")
    if live.get("status") == "evaluated" and any(item.get("unsupported_rule") or not item.get("reference_valid") for item in live.get("claim_findings", [])):
        failures.append("live-provider report has rule-detected unsupported claims")
    if live.get("status") == "evaluated" and live.get("provider") != "gemini":
        failures.append("live-provider evaluation did not use Gemini")
    if live.get("status") == "evaluated" and ml and live.get("dataset_sha256") != ml.get("dataset_sha256"):
        failures.append("live report and ML evaluation use different dataset versions")
    failures.extend(f"regression: {finding['metric']}" for finding in (run.get("regression") or {}).get("findings", []))
    status = "fail" if failures else "inconclusive" if missing else "pass"
    return {"status": status, "failures": failures, "missing": missing}
