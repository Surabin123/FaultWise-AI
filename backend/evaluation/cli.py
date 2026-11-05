"""Run local evaluation, compare saved runs, and apply the quality gate."""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from backend.api.errors import ApiError
from backend.api.reports.provider import GeminiProvider
from backend.api.reports.service import ReportService
from backend.api.schemas import DatasetVariant
from backend.api.services.datasets import DatasetService
from backend.api.services.models import ModelService
from .cases import load_cases
from .gate import evaluate_gate, load_gate_config
from .metrics import claim_findings
from .ml import load_ml_metrics
from .regression import compare
from .runner import run_suite

ROOT = Path(__file__).parent


def _live_check(data_dir: Path, model_path: Path, engine_id: int) -> dict[str, Any]:
    if os.getenv("FAULTWISE_LLM_PROVIDER", "none").lower() != "gemini" or not os.getenv("GEMINI_API_KEY"):
        return {"status": "unavailable", "reason": "Gemini provider is not configured"}
    provider = GeminiProvider(os.environ["GEMINI_API_KEY"], os.getenv("FAULTWISE_GEMINI_MODEL", "gemini-2.5-flash"))
    service = ReportService(DatasetService(data_dir), ModelService(model_path), provider, Path("artifacts/reports"))
    try:
        report = service.create(DatasetVariant.FD001, engine_id, persist=False)
    except ApiError as exc:
        return {"status": "failed", "error": exc.code}
    findings = claim_findings(report)
    if report.assessment != "investigate" or not findings:
        return {"status": "inconclusive", "assessment": report.assessment, "missing_evidence": report.missing_evidence}
    return {
        "status": "evaluated", "provider": report.provider, "provider_model": report.provider_model,
        "dataset_sha256": report.dataset_sha256, "model_id": report.model_id,
        "report_generation_ms": report.latency_ms, "provider_latency_ms": None,
        "token_usage": report.token_usage, "estimated_cost": None,
        "claim_findings": findings, "report_id": report.report_id,
        "limitations": ["Single live report; no human semantic review or diagnostic accuracy measurement."],
    }


def _save(run: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"evaluation-{run['run_id']}.json"
    with path.open("x", encoding="utf-8") as target:
        json.dump(run, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run synthetic golden cases and save a quality report")
    run.add_argument("--cases", type=Path, default=ROOT / "golden/v1.json")
    run.add_argument("--gate-config", type=Path, default=ROOT / "golden/release-gate-v1.json")
    run.add_argument("--category", choices=["valid", "abstention", "grounding", "safety"], action="append")
    run.add_argument("--ml-report", type=Path, help="Existing Phase 1 test_metrics.json; never retrains NASA data")
    run.add_argument("--baseline", type=Path, help="Earlier evaluation JSON with compatible dataset/category scope")
    run.add_argument("--live", action="store_true", help="Opt in to one real Gemini report request")
    run.add_argument("--live-data-dir", type=Path, default=Path("data"))
    run.add_argument("--live-model", type=Path, default=Path("artifacts/ml/faultwise_model.joblib"))
    run.add_argument("--live-engine", type=int, default=1)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    comparison = commands.add_parser("compare", help="Compare two saved evaluation JSON reports")
    comparison.add_argument("previous", type=Path)
    comparison.add_argument("current", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.command == "compare":
        result = compare(json.loads(args.previous.read_text(encoding="utf-8")), json.loads(args.current.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        return
    dataset = load_cases(args.cases)
    with tempfile.TemporaryDirectory(prefix="faultwise-eval-") as temporary:
        result = run_suite(dataset, Path(temporary), categories=set(args.category) if args.category else None)
    result["ml_metrics"] = load_ml_metrics(args.ml_report) if args.ml_report else None
    result["live_provider"] = _live_check(args.live_data_dir, args.live_model, args.live_engine) if args.live else {"status": "not_run"}
    result["regression"] = compare(json.loads(args.baseline.read_text(encoding="utf-8")), result) if args.baseline else None
    result["release_gate"] = evaluate_gate(result, load_gate_config(args.gate_config))
    result["overall_status"] = result["release_gate"]["status"]
    path = _save(result, args.output_dir)
    print(f"Saved {path}")
    print(f"Cases: {len(result['cases'])}, failed: {len(result['failed_cases'])}; release gate: {result['overall_status']}")
    if result["failed_cases"]:
        print("Failed cases: " + ", ".join(result["failed_cases"]))
    if result["release_gate"]["missing"]:
        print("Unavailable: " + "; ".join(result["release_gate"]["missing"]))


if __name__ == "__main__":
    main()
