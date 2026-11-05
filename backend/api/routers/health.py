"""Process health and dependency readiness endpoints."""

from fastapi import APIRouter, Request, Response, status

from ..schemas import DatasetVariant, HealthResponse, ModelStatus, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return process health without implying the model or dataset is ready."""
    return HealthResponse()


@router.get("/api/v1/model", response_model=ModelStatus, tags=["model"])
def model_status(request: Request) -> ModelStatus:
    """Expose whether the configured Phase 1 artifact can be loaded."""
    return request.app.state.model_service.status()


@router.get("/ready", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}})
def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Readiness requires both the configured dataset and compatible model."""
    variant = DatasetVariant(request.app.state.settings.default_variant)
    dataset = request.app.state.dataset_service.summary(variant)
    model: ModelStatus = request.app.state.model_service.status()
    ready = dataset.status.value == "available" and model.status.value == "loaded" and model.variant == variant
    result = ReadinessResponse(status="ready" if ready else "not_ready", dataset=dataset, model=model)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
