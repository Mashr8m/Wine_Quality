"""Pipeline persistence, full metadata, etalon drift and training reproducibility."""
import json
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from train import FEATURES, load_data, run


def test_three_artifacts_exist(artifacts):
    for filename in ["model.joblib", "meta.json", "etalon.json"]:
        assert (artifacts / filename).is_file(), filename


def test_saved_object_is_full_pipeline(artifacts):
    model = joblib.load(artifacts / "model.joblib")
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "classifier"]


def test_metadata(artifacts):
    meta = json.loads((artifacts / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == "wine-logreg-test-0.0.1"
    assert meta["threshold"] == 0.65
    assert len(meta["data_sha256"]) == 64
    assert meta["demo_mode"] is True
    assert meta["split"]["random_state"] == 42
    assert meta["data_counts"]["train_rows"] + meta["data_counts"]["validation_rows"] == meta["data_counts"]["unique_rows"]
    assert set(["precision", "recall", "f1", "roc_auc", "pr_auc"]).issubset(meta["metrics"])


def test_etalon_predictions(artifacts):
    model = joblib.load(artifacts / "model.joblib")
    examples = json.loads((artifacts / "etalon.json").read_text(encoding="utf-8"))
    assert len(examples) == 5
    for entry in examples:
        assert set(entry["features"]) == set(FEATURES)
        probability = float(model.predict_proba(pd.DataFrame([entry["features"]]))[0, 1])
        assert probability == pytest.approx(entry["probability"], abs=1e-7)
        assert int(probability >= 0.65) == entry["predicted_class"]


def test_full_train_default_requires_uci_data_in_offline_mode(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path, demo=False, offline=True)


def test_retrain_and_rollback_backup(tmp_path):
    data_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"
    output = tmp_path / "artifacts"
    run(data_dir, output, demo=True, model_version="wine-logreg-1.0.0")
    run(data_dir, output, demo=True, model_version="wine-logreg-1.1.0", save_previous=True)
    old = output / "previous" / "wine-logreg-1.0.0"
    assert json.loads((old / "meta.json").read_text(encoding="utf-8"))["model_version"] == "wine-logreg-1.0.0"
    assert (old / "model.joblib").exists()
    assert json.loads((output / "meta.json").read_text(encoding="utf-8"))["model_version"] == "wine-logreg-1.1.0"


def test_rollback_changes_active_artifacts(tmp_path):
    from pathlib import Path
    from scripts.rollback_model import rollback
    data_dir = Path(__file__).resolve().parents[1] / "data"
    path = tmp_path / "artifacts"
    run(data_dir, path, demo=True, model_version="wine-logreg-1.0.0")
    run(data_dir, path, demo=True, model_version="wine-logreg-1.1.0", save_previous=True)
    result = rollback(path, "wine-logreg-1.0.0")
    assert result["model_version"] == "wine-logreg-1.0.0"
    assert json.loads((path / "meta.json").read_text(encoding="utf-8"))["model_version"] == "wine-logreg-1.0.0"
    assert (path / "previous" / "wine-logreg-1.1.0" / "model.joblib").exists()
