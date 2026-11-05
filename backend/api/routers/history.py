"""Read-only evaluation, model-metric, provider, and report history endpoints."""

from fastapi import APIRouter, HTTPException, Request

from backend.evaluation.ml import load_ml_metrics

router = APIRouter(prefix="/api/v1", tags=["history"])


@router.get("/provider")
def provider_status(request: Request) -> dict:
    provider = request.app.state.report_service.provider
    return {"configured": provider is not None, "provider": provider.name if provider else None, "model": provider.model if provider else None}


@router.get("/model/evaluation")
def model_evaluation(request: Request) -> dict:
    path = request.app.state.settings.ml_metrics_path
    if not path.is_file():
        raise HTTPException(503, detail="Saved ML evaluation is unavailable.")
    try:
        return load_ml_metrics(path)
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(503, detail="Saved ML evaluation failed validation.") from None


@router.get("/evaluations")
def list_evaluations(request: Request) -> dict:
    return {"evaluations": request.app.state.history_store.evaluations()}


@router.get("/evaluations/compare")
def compare_evaluations(request: Request, previous: str, current: str) -> dict:
    try:
        result = request.app.state.history_store.compare(previous, current)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(404, detail="Evaluation run not found.")
    return result


@router.get("/evaluations/{run_id}")
def get_evaluation(request: Request, run_id: str) -> dict:
    try:
        result = request.app.state.history_store.evaluation(run_id)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(404, detail="Evaluation run not found.")
    return result


@router.get("/reports")
def list_reports(request: Request) -> dict:
    return {"reports": request.app.state.history_store.reports()}


@router.get("/reports/{report_id}")
def get_report(request: Request, report_id: str) -> dict:
    try:
        result = request.app.state.history_store.report(report_id)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(404, detail="Report not found.")
    return result.model_dump(mode="json")
