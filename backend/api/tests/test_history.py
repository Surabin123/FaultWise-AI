"""Read-only dashboard contracts over actual saved artifacts."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.api.config import Settings
from backend.api.main import create_app
from backend.api.reports.schemas import DiagnosticReport


def _run(run_id: str, *, failures: list[str]) -> dict:
    return {
        "run_id": run_id, "timestamp_utc": "2026-11-05T00:00:00+00:00",
        "evaluation_dataset_version": "synthetic-golden-v1", "evaluation_dataset_provenance": "synthetic_fixture",
        "model_version": "synthetic-fixture-model", "prompt_version": "diagnostic-v3",
        "overall_status": "fail" if failures else "inconclusive",
        "evaluation_config": {"categories": "all"},
        "cases": [{"case_id":"case-a","passed":not failures}],
        "failed_cases": failures,
        "metrics": {"unsupported_claim_rate_rule": 0.0},
        "live_provider": {"status":"not_run"},
    }


def test_dashboard_history_and_provider_endpoints(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluation"; evaluations.mkdir()
    reports = tmp_path / "reports"; reports.mkdir()
    old, new = str(uuid4()), str(uuid4())
    (evaluations / f"evaluation-{old}.json").write_text(json.dumps(_run(old,failures=["case-a"])),encoding="utf-8")
    (evaluations / f"evaluation-{new}.json").write_text(json.dumps(_run(new,failures=[])),encoding="utf-8")
    (evaluations / "evaluation-corrupt.json").write_text("{broken",encoding="utf-8")
    report_id = str(uuid4())
    report = DiagnosticReport(
        report_id=report_id, engine_id=1, dataset_variant="FD001", dataset_sha256="synthetic-hash",
        model_id="fixture-model", prompt_version="diagnostic-v3", provider=None, provider_model=None,
        generated_at_utc=datetime.now(timezone.utc), latency_ms=1.0, token_usage=None,
        validation_status="abstained", assessment="insufficient_evidence",
        executive_summary="Evidence insufficient.", observed_evidence=[], ml_predictions=[],
        potential_concerns=[], recommended_investigation=[], uncertainty="Evidence unavailable.",
        limitations=["Synthetic test report."], missing_evidence=["trajectory"],
    )
    (reports / f"{report_id}.json").write_text(report.model_dump_json(),encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data", model_path=tmp_path / "missing.joblib",
                        evaluation_dir=evaluations, report_dir=reports, ml_metrics_path=tmp_path / "missing.json", public_api_base="/testing")
    client = TestClient(create_app(settings))
    assert client.get("/").status_code == 200
    assert "Engine explorer" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert '"apiBase": "/testing"' in client.get("/app-config.js").text
    assert client.get("/api/v1/provider").json() == {"configured": False, "provider": None, "model": None}
    listed = client.get("/api/v1/evaluations").json()["evaluations"]
    assert len(listed) == 2
    assert {item["run_id"] for item in listed} == {old,new}
    assert client.get(f"/api/v1/evaluations/{old}").json()["failed_cases"] == ["case-a"]
    compared = client.get("/api/v1/evaluations/compare", params={"previous":old,"current":new}).json()
    assert compared["resolved_cases"] == ["case-a"]
    assert client.get("/api/v1/evaluations/invalid").status_code == 422
    assert client.get(f"/api/v1/evaluations/{uuid4()}").status_code == 404
    assert client.get("/api/v1/reports").json()["reports"][0]["report_id"] == report_id
    assert client.get(f"/api/v1/reports/{report_id}").json()["assessment"] == "insufficient_evidence"
    assert client.get("/api/v1/reports/invalid").status_code == 422
    assert client.get("/api/v1/model/evaluation").status_code == 503


def test_model_evaluation_exposes_verified_saved_metrics(tmp_path: Path) -> None:
    data = tmp_path / "train_FD001.txt"; data.write_text("synthetic",encoding="utf-8")
    saved = tmp_path / "test_metrics.json"
    saved.write_text(json.dumps({
        "dataset":"Synthetic fixture","dataset_source":str(data),
        "dataset_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
        "config":{"dataset_kind":"synthetic_fixture","variant":"FD001","seed":42},
        "trained_at_utc":"2026-11-05T00:00:00+00:00",
        "training_units":[1],"test_units":[2],"test_protocol":"engine-level holdout",
        "official_benchmark_metrics":False,
        "test_metrics":{"mae":1.0,"rmse":1.2,"phm08_score_sum":2.0,"n_observations":4},
    }),encoding="utf-8")
    client = TestClient(create_app(Settings(data_dir=tmp_path,model_path=tmp_path / "missing.joblib",ml_metrics_path=saved)))
    result = client.get("/api/v1/model/evaluation")
    assert result.status_code == 200
    assert result.json()["integrity_verified"] is True
    assert result.json()["dataset_kind"] == "synthetic_fixture"
    assert "dataset_source" not in result.json()
