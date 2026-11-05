from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from faultwise_ml.data import load_cmapss
from faultwise_ml.prediction import load_artifact, predict_latest_per_unit
from faultwise_ml.schema import SENSOR_COLUMNS
from faultwise_ml.training import train_baseline

from backend.api.config import Settings
from backend.api.main import create_app

FIXTURE = Path(__file__).parents[3] / "packages" / "ml" / "tests" / "fixtures" / "synthetic_fd001.txt"


@pytest.fixture
def client_without_model(tmp_path: Path) -> TestClient:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_FD001.txt").write_bytes(FIXTURE.read_bytes())
    settings = Settings(data_dir=data_dir, model_path=tmp_path / "missing.joblib")
    return TestClient(create_app(settings))


@pytest.fixture
def client_with_fixture_model(tmp_path: Path) -> tuple[TestClient, Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "train_FD001.txt"
    dataset_path.write_bytes(FIXTURE.read_bytes())
    model_dir = tmp_path / "model"
    train_baseline(
        dataset_path,
        model_dir,
        variant="FD001",
        dataset_kind="synthetic_fixture",
        seed=42,
        test_fraction=0.25,
    )
    model_path = model_dir / "faultwise_model.joblib"
    settings = Settings(data_dir=data_dir, model_path=model_path)
    return TestClient(create_app(settings)), dataset_path, model_path


def test_health_is_independent_of_dataset_and_model(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(data_dir=tmp_path / "none", model_path=tmp_path / "none.joblib")))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "faultwise-api"}


def test_dataset_availability_and_readiness_when_prerequisites_missing(
    tmp_path: Path, client_without_model: TestClient
) -> None:
    listed = client_without_model.get("/api/v1/datasets")
    assert listed.status_code == 200
    states = {item["variant"]: item["status"] for item in listed.json()["datasets"]}
    assert states == {"FD001": "available", "FD002": "unavailable", "FD003": "unavailable", "FD004": "unavailable"}
    ready = client_without_model.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert ready.json()["model"]["status"] == "unavailable"
    assert client_without_model.get("/api/v1/datasets/FD002").status_code == 503


def test_dataset_metadata_engine_list_trajectory_and_bad_engine(client_without_model: TestClient) -> None:
    metadata = client_without_model.get("/api/v1/datasets/FD001").json()
    assert metadata["status"] == "available"
    assert metadata["engine_count"] == 4
    assert len(metadata["sensors"]) == 21
    assert metadata["cycle_range"] == [1, 8]
    engines = client_without_model.get("/api/v1/datasets/FD001/engines").json()
    assert engines == {"variant": "FD001", "engine_ids": [1, 2, 3, 4], "count": 4}
    trajectory = client_without_model.get("/api/v1/datasets/FD001/engines/1/trajectory?limit=2").json()
    assert trajectory["total_cycles"] == 8
    assert trajectory["offset"] == 0
    assert len(trajectory["cycles"]) == 2
    assert "sensor_1" in trajectory["cycles"][0]["sensors"]
    missing_engine = client_without_model.get("/api/v1/datasets/FD001/engines/999/trajectory")
    assert missing_engine.status_code == 404
    assert missing_engine.json()["error"]["code"] == "engine_not_found"
    assert client_without_model.get("/api/v1/datasets/FD001/engines/0/trajectory").status_code == 422


def test_invalid_dataset_is_distinguished_from_unavailable(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_FD001.txt").write_text("bad rows\n", encoding="utf-8")
    client = TestClient(create_app(Settings(data_dir=data_dir, model_path=tmp_path / "none.joblib")))
    items = {item["variant"]: item for item in client.get("/api/v1/datasets").json()["datasets"]}
    assert items["FD001"]["status"] == "invalid"
    assert items["FD001"]["row_count"] is None
    assert items["FD002"]["status"] == "unavailable"


def test_missing_model_returns_explicit_error(client_without_model: TestClient) -> None:
    assert client_without_model.get("/api/v1/model").json()["status"] == "unavailable"
    response = client_without_model.post("/api/v1/datasets/FD001/engines/1/prediction")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_phase1_artifact_prediction_uses_pipeline_and_latest_sensor_evidence(
    client_with_fixture_model: tuple[TestClient, Path, Path]
) -> None:
    client, dataset_path, model_path = client_with_fixture_model
    status = client.get("/api/v1/model").json()
    assert status["status"] == "loaded"
    assert status["dataset_kind"] == "synthetic_fixture"
    assert client.get("/ready").status_code == 200
    response = client.post("/api/v1/datasets/FD001/engines/2/prediction")
    assert response.status_code == 200
    result = response.json()
    frame = load_cmapss(dataset_path, variant="FD001")
    engine = frame.loc[frame.unit == 2]
    expected = float(predict_latest_per_unit(load_artifact(model_path)["model"], engine).iloc[0].predicted_rul)
    assert result["status"] == "predicted"
    assert result["engine_id"] == 2
    assert result["dataset_variant"] == "FD001"
    assert result["model_dataset_kind"] == "synthetic_fixture"
    assert result["predicted_rul"] == pytest.approx(expected)
    assert result["evidence"]["cycle"] == 8
    assert set(result["evidence"]["readings"]) == set(SENSOR_COLUMNS)
    assert result["unit"] == "cycles"


def test_model_variant_mismatch_is_explicit(client_with_fixture_model: tuple[TestClient, Path, Path]) -> None:
    client, dataset_path, model_path = client_with_fixture_model
    (dataset_path.parent / "train_FD002.txt").write_bytes(dataset_path.read_bytes())
    response = client.post("/api/v1/datasets/FD002/engines/1/prediction")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_variant_mismatch"


def test_anomaly_analysis_flags_unusual_behavior_and_short_history_is_insufficient(
    client_without_model: TestClient, tmp_path: Path
) -> None:
    short = client_without_model.post("/api/v1/datasets/FD001/engines/1/anomaly-analysis").json()
    assert short["status"] == "insufficient_evidence"
    assert short["unusual_sensor_behavior"] == []

    data_dir = tmp_path / "long_data"
    data_dir.mkdir()
    source_frame = load_cmapss(FIXTURE, variant="FD001")
    rows = source_frame.loc[source_frame.unit == 1].iloc[:7].copy()
    template = rows.iloc[-1].copy()
    for cycle in range(8, 13):
        row = template.copy()
        row["cycle"] = cycle
        for sensor in SENSOR_COLUMNS:
            row[sensor] = float(template[sensor]) + (cycle - 7) * 0.1
        row["sensor_1"] = 1000.0 + cycle
        rows.loc[len(rows)] = row
    rows.to_csv(data_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    client = TestClient(create_app(Settings(data_dir=data_dir, model_path=tmp_path / "none.joblib")))
    result = client.post("/api/v1/datasets/FD001/engines/1/anomaly-analysis").json()
    assert result["status"] == "analyzed"
    flagged = {item["sensor"] for item in result["unusual_sensor_behavior"]}
    assert "sensor_1" in flagged
    sensor = next(item for item in result["unusual_sensor_behavior"] if item["sensor"] == "sensor_1")
    assert sensor["robust_z_score"] > 3.5
    assert "not a probability" in result["score_definition"]


def test_corrupt_model_artifact_is_not_reported_as_ready(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_FD001.txt").write_bytes(FIXTURE.read_bytes())
    model_path = tmp_path / "broken.joblib"
    model_path.write_bytes(b"not a joblib file")
    client = TestClient(create_app(Settings(data_dir=data_dir, model_path=model_path)))
    assert client.get("/api/v1/model").json()["status"] == "invalid"
    assert client.get("/ready").status_code == 503
    prediction = client.post("/api/v1/datasets/FD001/engines/1/prediction")
    assert prediction.status_code == 503
    assert prediction.json()["error"]["code"] == "model_artifact_invalid"
