"""Versioned synthetic golden-case contract and loader."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    category: Literal["valid", "abstention", "grounding", "safety"]
    severity: Literal["low", "medium", "high", "critical"]
    review_status: Literal["engineered_synthetic", "human_reviewed"]
    scenario: Literal[
        "valid", "short_history", "missing_sensors", "invalid_sensor",
        "missing_dataset", "missing_model", "dataset_mismatch", "out_of_domain",
        "injected_sensor", "provider_unavailable", "bad_json", "unknown_reference",
        "observed_claim", "invented_measurement", "component_failure",
        "numerical_confidence", "repair_instruction", "secret_request",
        "ignore_evidence",
    ]
    evidence_inputs: str = Field(min_length=1)
    attack: str | None = None
    expected_behavior: str = Field(min_length=1)
    expected_assessment: Literal["investigate", "insufficient_evidence"] | None = None
    expected_error: str | None = None
    expected_evidence_ids: list[str] = Field(default_factory=list)
    expected_missing_evidence: list[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1)
    provenance: Literal["synthetic_fixture"]
    cases: list[GoldenCase] = Field(min_length=1)


def load_cases(path: Path) -> GoldenDataset:
    try:
        dataset = GoldenDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid golden dataset at {path}: {exc}") from exc
    identifiers = [case.case_id for case in dataset.cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Golden dataset case IDs must be unique")
    for case in dataset.cases:
        if (case.expected_assessment is None) == (case.expected_error is None):
            raise ValueError(f"{case.case_id}: exactly one expected outcome is required")
    return dataset
