"""HTTP happy path, schema validation (422), info and outage (503)."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True
    assert r.json()["model_version"] == "wine-logreg-test-0.0.1"


def test_prediction(client, example):
    r = client.post("/predict", json=example)
    assert r.status_code == 200, r.text
    obj = r.json()
    assert obj["predicted_class"] in [0, 1]
    assert obj["label"] in ["хорошее", "остальное"]
    assert 0 <= obj["probability"] <= 1
    assert obj["threshold"] == 0.65
    assert obj["model_version"] == "wine-logreg-test-0.0.1"


@pytest.mark.parametrize("changed", [
    {"alcohol": -1}, {"alcohol": 50}, {"pH": 20},
    {"тип_вина": "розовое"}, {"fixed acidity": -0.1},
    {"free sulfur dioxide": 999},
])
def test_bad_fields_return_422(client, example, changed):
    assert client.post("/predict", json=dict(example, **changed)).status_code == 422


def test_missing_field_422(client, example):
    data = dict(example)
    data.pop("alcohol")
    assert client.post("/predict", json=data).status_code == 422


def test_wrong_type_422(client, example):
    assert client.post("/predict", json=dict(example, alcohol="not a number")).status_code == 422


def test_extra_field_422(client, example):
    assert client.post("/predict", json=dict(example, quality=9)).status_code == 422


def test_so2_constraint_422(client, example):
    assert client.post("/predict", json=dict(example, **{"free sulfur dioxide": 100, "total sulfur dioxide": 20})).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["threshold"] == 0.65
    assert info["demo_mode"] is True
    assert len(info["features"]) == 12
    assert "precision" in info["metrics"]


def test_swagger(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_unloaded_model_returns_503(monkeypatch, tmp_path, artifacts, example):
    monkeypatch.setenv("WINE_ARTIFACTS_DIR", str(tmp_path / "nonexistent"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json=example).status_code == 503
        assert client.get("/model-info").status_code == 503
