"""Read locally persisted evaluations and reports without fabricating history."""

import json
from pathlib import Path
from uuid import UUID
from typing import Any

from pydantic import ValidationError

from backend.api.reports.schemas import DiagnosticReport
from backend.evaluation.regression import compare


class HistoryStore:
    def __init__(self, evaluation_dir: Path, report_dir: Path) -> None:
        self.evaluation_dir = evaluation_dir
        self.report_dir = report_dir

    @staticmethod
    def _id(value: str) -> str:
        try:
            return str(UUID(value))
        except ValueError as exc:
            raise ValueError("An evaluation or report ID must be a UUID") from exc

    def evaluation(self, run_id: str) -> dict[str, Any] | None:
        path = self.evaluation_dir / f"evaluation-{self._id(run_id)}.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) and value.get("run_id") == run_id else None

    def evaluations(self) -> list[dict[str, Any]]:
        if not self.evaluation_dir.is_dir():
            return []
        items = []
        for path in self.evaluation_dir.glob("evaluation-*.json"):
            run_id = path.stem.removeprefix("evaluation-")
            try:
                value = self.evaluation(run_id)
            except ValueError:
                continue
            if value is not None:
                metrics = value.get("metrics") or {}
                cases = value.get("cases") or []
                items.append({
                    "run_id": run_id,
                    "timestamp_utc": value.get("timestamp_utc"),
                    "evaluation_dataset_version": value.get("evaluation_dataset_version"),
                    "model_version": value.get("model_version"),
                    "prompt_version": value.get("prompt_version"),
                    "overall_status": value.get("overall_status"),
                    "case_count": len(cases),
                    "passed_count": sum(bool(case.get("passed")) for case in cases if isinstance(case, dict)),
                    "failed_count": len(value.get("failed_cases") or []),
                    "unsupported_claim_rate_rule": metrics.get("unsupported_claim_rate_rule"),
                    "live_provider_status": (value.get("live_provider") or {}).get("status"),
                })
        return sorted(items, key=lambda item: item["timestamp_utc"] or "", reverse=True)

    def compare(self, previous: str, current: str) -> dict[str, Any] | None:
        left, right = self.evaluation(previous), self.evaluation(current)
        return compare(left, right) if left is not None and right is not None else None

    def report(self, report_id: str) -> DiagnosticReport | None:
        path = self.report_dir / f"{self._id(report_id)}.json"
        if not path.is_file():
            return None
        try:
            report = DiagnosticReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None
        return report if report.report_id == report_id else None

    def reports(self) -> list[dict[str, Any]]:
        if not self.report_dir.is_dir():
            return []
        items = []
        for path in self.report_dir.glob("*.json"):
            try:
                report = self.report(path.stem)
            except ValueError:
                continue
            if report:
                items.append({
                    "report_id": report.report_id, "engine_id": report.engine_id,
                    "dataset_variant": report.dataset_variant,
                    "generated_at_utc": report.generated_at_utc.isoformat(),
                    "assessment": report.assessment, "provider": report.provider,
                })
        return sorted(items, key=lambda item: item["generated_at_utc"], reverse=True)
