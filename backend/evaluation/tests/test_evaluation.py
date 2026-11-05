"""Deterministic Phase 4 contracts and real-service adversarial evaluation."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.evaluation.cases import load_cases
from backend.evaluation.gate import GateConfig, evaluate_gate, load_gate_config
from backend.evaluation.metrics import abstention_metrics, ratio
from backend.evaluation.ml import load_ml_metrics
from backend.evaluation.regression import compare
from backend.evaluation.runner import run_suite

ROOT = Path(__file__).parents[1]
CASES = ROOT / "golden/v1.json"
GATE = ROOT / "golden/release-gate-v1.json"


def test_golden_dataset_loading_and_validation(tmp_path: Path) -> None:
    dataset = load_cases(CASES)
    assert dataset.provenance == "synthetic_fixture"
    assert len(dataset.cases) == 19
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    raw["cases"].append(raw["cases"][0])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_cases(duplicate)
    raw["cases"][-1]["case_id"] = "new"
    raw["cases"][-1]["severity"] = "unreviewed"
    duplicate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid golden dataset"):
        load_cases(duplicate)


def test_zero_denominators_and_abstention_definitions() -> None:
    assert ratio(0, 0) is None
    assert abstention_metrics([], []) == {
        "precision": None, "recall": None, "false_abstention_rate": None,
        "missed_abstention_rate": None,
    }
    result = abstention_metrics([True, True, False, False], [True, False, True, False])
    assert result == {
        "precision": 0.5, "recall": 0.5,
        "false_abstention_rate": 0.5, "missed_abstention_rate": 0.5,
    }


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return run_suite(load_cases(CASES), tmp_path_factory.mktemp("phase4"))


def test_runner_exercises_actual_service_and_writes_machine_readable_report(evaluation: dict, tmp_path: Path) -> None:
    assert evaluation["evaluation_dataset_provenance"] == "synthetic_fixture"
    assert evaluation["fixture_training"]["synthetic_data"] is True
    assert evaluation["metrics"]["denominators"] == {"cases": 19, "accepted_reports": 1, "claims": 2, "abstention_cases": 8}
    assert evaluation["metrics"]["unsupported_claim_rate_rule"] == 0.0
    assert evaluation["metrics"]["schema_validity"] == 1.0
    assert evaluation["failed_cases"] == []
    by_id = {case["case_id"]: case for case in evaluation["cases"]}
    assert by_id["valid_sensor_anomaly"]["report"]["ml_predictions"]
    assert by_id["valid_sensor_anomaly"]["claim_findings"][0]["reference_valid"]
    assert by_id["missing_latest_sensors"]["provider_called"] is False
    assert by_id["instruction_in_sensor_field"]["actual_error"] == "dataset_invalid"
    assert by_id["unknown_evidence_reference"]["actual_error"] == "report_validation_failed"
    assert by_id["unsupported_component_failure"]["actual_error"] == "report_validation_failed"
    assert by_id["secret_exfiltration_request"]["actual_error"] == "report_validation_failed"
    assert by_id["ignore_evidence_instruction"]["actual_error"] == "report_validation_failed"
    assert "supported operating conditions" in by_id["unsupported_operating_condition"]["report"]["missing_evidence"]
    output = tmp_path / "result.json"
    output.write_text(json.dumps(evaluation, allow_nan=False), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["failed_cases"] == evaluation["failed_cases"]


def test_release_gate_fail_inconclusive_and_pass(evaluation: dict) -> None:
    config = load_gate_config(GATE)
    current = evaluate_gate(evaluation, config)
    assert current["status"] == "inconclusive"
    assert "live-provider evaluation unavailable" in current["missing"]
    clean = deepcopy(evaluation)
    clean["failed_cases"] = []
    for case in clean["cases"]:
        case["passed"] = True
    clean["metrics"].update({"grounded_claim_rate_structural": 1.0, "unsupported_claim_rate_rule": 0.0})
    assert evaluate_gate(clean, config)["status"] == "inconclusive"
    clean["ml_metrics"] = {"integrity_verified": True, "dataset_kind": "nasa_cmapss", "dataset_sha256": "same"}
    clean["live_provider"] = {"status": "evaluated", "provider": "gemini", "dataset_sha256": "same"}
    assert evaluate_gate(clean, config)["status"] == "pass"
    clean["live_provider"]["claim_findings"] = [{"unsupported_rule": True, "reference_valid": True}]
    assert evaluate_gate(clean, config)["status"] == "fail"
    clean["live_provider"]["claim_findings"] = []
    clean["ml_metrics"]["dataset_kind"] = "synthetic_fixture"
    assert evaluate_gate(clean, config)["status"] == "fail"
    clean["ml_metrics"]["dataset_kind"] = "nasa_cmapss"
    clean["failed_cases"] = ["secret_exfiltration_request"]
    assert evaluate_gate(clean, config)["status"] == "fail"
    clean["failed_cases"] = []
    clean["metrics"]["unsupported_claim_rate_rule"] = None
    assert evaluate_gate(clean, config)["status"] == "inconclusive"
    with pytest.raises(ValidationError):
        GateConfig(required_case_ids=["x"], minimum_schema_validity=2,
                   minimum_grounded_claim_rate=1, minimum_reference_validity=1,
                   maximum_unsupported_claim_rate=0, maximum_missed_abstention_rate=0)


def test_regression_and_compatibility(evaluation: dict) -> None:
    previous = deepcopy(evaluation)
    current = deepcopy(evaluation)
    current["failed_cases"] = ["secret_exfiltration_request"]
    current["metrics"]["unsupported_claim_rate_rule"] = 0.5
    difference = compare(previous, current)
    names = [item["metric"] for item in difference["findings"]]
    assert "unsupported_claim_rate_rule" in names
    assert names.count("new_failed_case") == 1
    current["evaluation_dataset_version"] = "different"
    with pytest.raises(ValueError, match="different golden"):
        compare(previous, current)


def test_saved_ml_report_integrity_is_not_recomputed_on_new_split(tmp_path: Path) -> None:
    source = tmp_path / "train_FD001.txt"
    source.write_text("synthetic source", encoding="utf-8")
    import hashlib
    report = {
        "dataset": "Synthetic fixture", "dataset_source": str(source),
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config": {"dataset_kind": "synthetic_fixture", "variant": "FD001", "seed": 42}, "trained_at_utc": "2026-01-01T00:00:00+00:00",
        "training_units": [1], "test_units": [2], "test_protocol": "engine-level holdout",
        "official_benchmark_metrics": False,
        "test_metrics": {"mae": 1.0, "rmse": 1.0, "phm08_score_sum": 2.0, "n_observations": 2},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_ml_metrics(path)["integrity_verified"] is True
    report["test_units"] = [1]
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_ml_metrics(path)["integrity_verified"] is False
