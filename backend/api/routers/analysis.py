"""RUL prediction and deterministic anomaly-analysis endpoints."""

from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Request
from fastapi import Path as ApiPath
from typing import Annotated

from faultwise_ml.schema import SENSOR_COLUMNS

from ..schemas import (
    AnomalyResponse,
    DatasetVariant,
    PredictionEvidence,
    PredictionResponse,
)
from ..services.anomalies import analyze_trajectory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/datasets/{variant}/engines/{engine_id}", tags=["analysis"])


@router.post("/prediction", response_model=PredictionResponse)
def predict_rul(variant: DatasetVariant, engine_id: Annotated[int, ApiPath(ge=1)], request: Request) -> PredictionResponse:
    dataset_service = request.app.state.dataset_service
    trajectory = dataset_service.trajectory(variant, engine_id)
    prediction, bundle = request.app.state.model_service.predict_latest(trajectory, variant)
    latest = trajectory.iloc[-1]
    readings = {
        sensor: _optional_float(latest[sensor]) for sensor in SENSOR_COLUMNS
    }
    config = bundle["config"]
    return PredictionResponse(
        engine_id=engine_id,
        dataset_variant=variant,
        predicted_rul=prediction,
        model_id=request.app.state.model_service.model_id(bundle),
        model_dataset_kind=config.get("dataset_kind", "unknown"),
        generated_at_utc=datetime.now(timezone.utc),
        status="predicted",
        evidence=PredictionEvidence(
            cycle=int(latest["cycle"]),
            readings=readings,
            missing_sensors=[sensor for sensor, value in readings.items() if value is None],
        ),
        limitations=[
            "RUL is a model estimate, not a confirmed failure time.",
            "The persisted Phase 1 pipeline and its feature ordering are used without refitting.",
            "The model's held-out row-level metrics are not an official C-MAPSS benchmark score.",
        ],
    )


@router.post("/anomaly-analysis", response_model=AnomalyResponse)
def anomaly_analysis(variant: DatasetVariant, engine_id: Annotated[int, ApiPath(ge=1)], request: Request) -> AnomalyResponse:
    trajectory = request.app.state.dataset_service.trajectory(variant, engine_id)
    result = analyze_trajectory(trajectory, engine_id=engine_id, variant=variant)
    logger.info(
        "Anomaly analysis completed for %s engine %s: status=%s flagged=%d",
        variant.value,
        engine_id,
        result.status,
        len(result.unusual_sensor_behavior),
    )
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None
