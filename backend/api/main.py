"""FastAPI application exposing real Phase 1 data and model functionality."""

import logging
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import Settings
from .errors import ApiError
from .routers import analysis, datasets, health, reports, history
from .reports.provider import GeminiProvider
from .reports.service import ReportService
from .schemas import ErrorBody, ErrorResponse
from .services.datasets import DatasetService
from .services.models import ModelService
from .services.history import HistoryStore
from .schemas import DatasetVariant
from .observability import install_observability


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_environment()
    if configured.default_variant not in {variant.value for variant in DatasetVariant}:
        raise ValueError("FAULTWISE_DEFAULT_VARIANT must be FD001, FD002, FD003, or FD004")
    logging.basicConfig(level=logging.INFO)
    application = FastAPI(
        title="FaultWise AI API",
        description="Dataset access and RUL inference backed by the Phase 1 C-MAPSS ML package.",
        version=__version__,
    )
    install_observability(application)
    if configured.allowed_origins:
        application.add_middleware(CORSMiddleware, allow_origins=list(configured.allowed_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    application.state.settings = configured
    application.state.dataset_service = DatasetService(configured.data_dir)
    application.state.model_service = ModelService(configured.model_path)
    provider = GeminiProvider(configured.gemini_api_key, configured.gemini_model) if configured.llm_provider == "gemini" and configured.gemini_api_key else None
    application.state.report_service = ReportService(application.state.dataset_service, application.state.model_service, provider, configured.report_dir)
    application.state.history_store = HistoryStore(configured.evaluation_dir, configured.report_dir)
    application.include_router(health.router)
    application.include_router(datasets.router)
    application.include_router(analysis.router)
    application.include_router(reports.router)
    application.include_router(history.router)

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    application.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @application.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @application.get("/app-config.js", include_in_schema=False)
    def app_config() -> Response:
        value = json.dumps({"apiBase": configured.public_api_base})
        return Response(f"window.FAULTWISE_CONFIG = {value};", media_type="application/javascript", headers={"Cache-Control": "no-store"})

    @application.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        body = ErrorResponse(error=ErrorBody(code=exc.code, message=exc.message))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def request_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(error=ErrorBody(code="invalid_request", message="Request validation failed."))
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = str(exc.detail) if isinstance(exc.detail, str) else "The requested resource was not found."
        body = ErrorResponse(error=ErrorBody(code=f"http_{exc.status_code}", message=message))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    return application


app = create_app()
