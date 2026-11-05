"""Diagnostic report endpoint."""

from typing import Annotated

from fastapi import APIRouter, Path as ApiPath, Request

from ..reports.schemas import DiagnosticReport, ReportRequest
from ..schemas import DatasetVariant

router = APIRouter(prefix="/api/v1/datasets/{variant}/engines/{engine_id}", tags=["reports"])


@router.post("/diagnostic-report", response_model=DiagnosticReport)
def diagnostic_report(
    variant: DatasetVariant,
    engine_id: Annotated[int, ApiPath(ge=1)],
    request: Request,
    options: ReportRequest | None = None,
) -> DiagnosticReport:
    return request.app.state.report_service.create(variant, engine_id, persist=options.persist if options else True)
