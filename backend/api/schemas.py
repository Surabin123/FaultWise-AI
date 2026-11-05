"""Typed HTTP request and response contracts."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetVariant(str, Enum):
    FD001 = "FD001"
    FD002 = "FD002"
    FD003 = "FD003"
    FD004 = "FD004"


class DatasetState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class DatasetSummary(BaseModel):
    variant: DatasetVariant
    status: DatasetState
    engine_count: int | None = None
    row_count: int | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary]


class DatasetMetadata(DatasetSummary):
    operating_settings: list[str]
    sensors: list[str]
    cycle_range: list[int] | None = None
    label_source: str = "derived from complete run-to-failure training trajectories"


class EngineListResponse(BaseModel):
    variant: DatasetVariant
    engine_ids: list[int]
    count: int


class EngineCycle(BaseModel):
    cycle: int
    operating_settings: dict[str, float | None]
    sensors: dict[str, float | None]


class EngineTrajectoryResponse(BaseModel):
    variant: DatasetVariant
    engine_id: int
    total_cycles: int
    offset: int
    cycles: list[EngineCycle]


class ModelState(str, Enum):
    LOADED = "loaded"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class ModelStatus(BaseModel):
    status: ModelState
    model_id: str | None = None
    variant: DatasetVariant | None = None
    dataset_kind: str | None = None
    trained_at_utc: datetime | None = None


class PredictionEvidence(BaseModel):
    source: Literal["observed_sensor_measurements"] = "observed_sensor_measurements"
    cycle: int
    readings: dict[str, float | None]
    missing_sensors: list[str]


class PredictionResponse(BaseModel):
    engine_id: int
    dataset_variant: DatasetVariant
    predicted_rul: float = Field(ge=0)
    unit: Literal["cycles"] = "cycles"
    model_id: str
    model_dataset_kind: str
    generated_at_utc: datetime
    status: Literal["predicted"]
    evidence: PredictionEvidence
    limitations: list[str]


class SensorAnomaly(BaseModel):
    sensor: str
    baseline_median: float
    recent_median: float
    robust_z_score: float | None = None
    unusual: bool
    explanation: str


class AnomalyResponse(BaseModel):
    engine_id: int
    dataset_variant: DatasetVariant
    status: Literal["analyzed", "insufficient_evidence"]
    cycles_analyzed: int
    baseline_cycles: int
    recent_cycles: int
    sensors_analyzed: int
    method: str
    method_version: str
    score_definition: str
    unusual_sensor_behavior: list[SensorAnomaly]
    skipped_sensors: list[str]
    limitations: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "faultwise-api"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dataset: DatasetSummary
    model: ModelStatus


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
