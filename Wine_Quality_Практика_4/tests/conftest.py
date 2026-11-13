"""Keep original deliverable artifacts intact. Train offline fixture independently."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from train import run
from app.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    path = tmp_path_factory.mktemp("wine_test_artifacts")
    run(ROOT / "data", path, demo=True, model_version="wine-logreg-test-0.0.1")
    return path


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("WINE_ARTIFACTS_DIR", str(artifacts))
    with TestClient(app) as tc:
        yield tc


@pytest.fixture()
def example(artifacts):
    import json
    return json.loads((artifacts / "etalon.json").read_text(encoding="utf-8"))[0]["features"]
