"""Operational HTTP metrics avoid unbounded paths and sensitive request details."""

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.config import Settings


def test_http_metrics_and_health(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "data", model_path=tmp_path / "missing.joblib"))
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/missing-path?token=private-value").status_code == 404
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'route="/health",status="200"' in metrics.text
    assert 'route="/ready",status="503"' in metrics.text
    assert 'route="unmatched",status="404"' in metrics.text
    assert "private-value" not in metrics.text


def test_metrics_are_scoped_to_each_app(tmp_path):
    first = TestClient(create_app(Settings(data_dir=tmp_path, model_path=tmp_path / "missing")))
    second = TestClient(create_app(Settings(data_dir=tmp_path, model_path=tmp_path / "missing")))
    first.get("/health")
    assert 'route="/health"' in first.get("/metrics").text
    assert 'route="/health"' not in second.get("/metrics").text
