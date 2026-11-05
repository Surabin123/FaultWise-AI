"""Diagnostic evidence and report contracts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    id: str
    type: Literal["observation", "prediction", "anomaly", "data_quality"]
    source: str
    value: Any
    unit: str | None = None
    cycle: int | None = None
    interpretation: str
    limitations: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=5)
    kind: Literal["hypothesis"]
    scope: Literal["sensor_behavior", "component_condition"]
    evidence_ids: list[str] = Field(min_length=1)


class GeneratedContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concerns: list[Claim] = Field(min_length=1)
    recommended_investigation: list[Claim] = Field(min_length=1)
    uncertainty: str = Field(min_length=10)
    limitations: list[str] = Field(min_length=1)


class DiagnosticReport(BaseModel):
    report_id: str
    engine_id: int
    dataset_variant: str
    dataset_sha256: str | None
    model_id: str | None
    prompt_version: str
    provider: str | None
    provider_model: str | None
    generated_at_utc: datetime
    latency_ms: float
    token_usage: dict[str, int] | None
    validation_status: Literal["passed", "abstained"]
    assessment: Literal["actionable", "investigate", "insufficient_evidence"]
    executive_summary: str
    observed_evidence: list[EvidenceItem]
    ml_predictions: list[EvidenceItem]
    potential_concerns: list[Claim]
    recommended_investigation: list[Claim]
    uncertainty: str
    limitations: list[str]
    missing_evidence: list[str]


class ReportRequest(BaseModel):
    persist: bool = True
