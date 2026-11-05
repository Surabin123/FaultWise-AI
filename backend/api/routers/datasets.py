"""Dataset inventory, engine listing, and trajectory endpoints."""

from fastapi import APIRouter, Query, Request
from typing import Annotated
from fastapi import Path as ApiPath

from faultwise_ml.schema import SENSOR_COLUMNS, SETTING_COLUMNS

from ..schemas import (
    DatasetListResponse,
    DatasetMetadata,
    DatasetSummary,
    DatasetVariant,
    EngineCycle,
    EngineListResponse,
    EngineTrajectoryResponse,
)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.get("", response_model=DatasetListResponse)
def list_datasets(request: Request) -> DatasetListResponse:
    return DatasetListResponse(datasets=request.app.state.dataset_service.list_datasets())


@router.get("/{variant}", response_model=DatasetMetadata)
def dataset_metadata(variant: DatasetVariant, request: Request) -> DatasetMetadata:
    return request.app.state.dataset_service.metadata(variant)


@router.get("/{variant}/engines", response_model=EngineListResponse)
def list_engines(variant: DatasetVariant, request: Request) -> EngineListResponse:
    engine_ids = request.app.state.dataset_service.engine_ids(variant)
    return EngineListResponse(variant=variant, engine_ids=engine_ids, count=len(engine_ids))


@router.get("/{variant}/engines/{engine_id}/trajectory", response_model=EngineTrajectoryResponse)
def engine_trajectory(
    variant: DatasetVariant,
    engine_id: Annotated[int, ApiPath(ge=1)],
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> EngineTrajectoryResponse:
    frame = request.app.state.dataset_service.trajectory(variant, engine_id)
    page = frame.iloc[offset : offset + limit]
    cycles = [
        EngineCycle(
            cycle=int(row["cycle"]),
            operating_settings={
                name: _optional_float(row[name]) for name in SETTING_COLUMNS
            },
            sensors={name: _optional_float(row[name]) for name in SENSOR_COLUMNS},
        )
        for _, row in page.iterrows()
    ]
    return EngineTrajectoryResponse(
        variant=variant,
        engine_id=engine_id,
        total_cycles=len(frame),
        offset=offset,
        cycles=cycles,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None
