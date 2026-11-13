"""Train UCI Wine Quality binary classifier. Default: original UCI full data.

Usage:
    python train.py             # downloads original UCI files if missing
    python train.py --offline   # requires both original CSV files in data/
    python train.py --demo      # explicit offline test using small actual UCI sample
    python train.py --version wine-logreg-1.1.0 --save-previous
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
FEATURES_NUM = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
FEATURES_CAT = ["тип_вина"]
FEATURES = FEATURES_NUM + FEATURES_CAT
THRESHOLD = 0.65
RANDOM_STATE = 42
MODEL_VERSION = "wine-logreg-1.0.0"


def download_uci(data_dir: Path) -> None:
    """Download original UCI archive; don't silently substitute other data."""
    data_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(UCI_ZIP, headers={"User-Agent": "WineQualityStudy/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Не удалось скачать UCI Wine Quality. Скачайте winequality-red.csv и "
            "winequality-white.csv из UCI в папку data/; или для офлайн-проверки "
            "используйте 'python train.py --demo'."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for name in ("winequality-red.csv", "winequality-white.csv"):
            if name not in names:
                raise ValueError(f"Файл {name} отсутствует в архиве UCI")
            content = archive.read(names[name])
            (data_dir / name).write_bytes(content)


def load_data(data_dir: Path = DATA_DIR, demo: bool = False, offline: bool = False):
    if demo:
        path = data_dir / "demo_wine.csv"
        data = pd.read_csv(path, sep=";")
        origins = [path]
    else:
        red_path, white_path = data_dir / "winequality-red.csv", data_dir / "winequality-white.csv"
        if not (red_path.exists() and white_path.exists()):
            if offline:
                raise FileNotFoundError("Для --offline нужны data/winequality-red.csv и data/winequality-white.csv")
            download_uci(data_dir)
        red = pd.read_csv(red_path, sep=";")
        white = pd.read_csv(white_path, sep=";")
        if (len(red), len(white)) != (1599, 4898):
            raise ValueError("Размеры исходных данных UCI неверны: ожидается (1599, 4898)")
        red["тип_вина"] = "красное"
        white["тип_вина"] = "белое"
        data = pd.concat([red, white], ignore_index=True)
        origins = [red_path, white_path]
    expected = set(FEATURES + ["quality"])
    if not expected.issubset(data.columns):
        raise ValueError(f"Не хватает колонок: {sorted(expected - set(data.columns))}")
    if data["quality"].isna().any() or data[FEATURES].isna().any().any():
        raise ValueError("В исходных данных обнаружены явные пропуски: проверьте источник")
    if not data["тип_вина"].isin(["белое", "красное"]).all():
        raise ValueError("Некорректное значение в колонке тип_вина")
    # Same cleanup as practice 3, BEFORE the split, to prevent leakage.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    data["good"] = (data["quality"] >= 7).astype(int)
    return data, origins, before


def build_model():
    preparation = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES_NUM),
        ("wine_type", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES_CAT),
    ])
    return Pipeline([
        ("preprocess", preparation),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def sha256_of_files(paths):
    result = hashlib.sha256()
    for path in paths:
        result.update(path.name.encode("utf-8"))
        result.update(path.read_bytes())
    return result.hexdigest()


def train_model(data, threshold: float = THRESHOLD):
    stratify = data["good"].astype(str) + "_" + data["тип_вина"]
    x_train, x_val, y_train, y_val = train_test_split(
        data[FEATURES], data["good"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=stratify,
    )
    model = build_model()
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_val)[:, 1]
    predicted = (probability >= threshold).astype(int)
    metrics = {
        "precision": round(float(precision_score(y_val, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, probability)), 4),
        "pr_auc": round(float(average_precision_score(y_val, probability)), 4),
        "selected_for_tasting": int(predicted.sum()),
        "true_positives": int(((predicted == 1) & (y_val.to_numpy() == 1)).sum()),
        "false_positives": int(((predicted == 1) & (y_val.to_numpy() == 0)).sum()),
        "false_negatives": int(((predicted == 0) & (y_val.to_numpy() == 1)).sum()),
        "true_negatives": int(((predicted == 0) & (y_val.to_numpy() == 0)).sum()),
    }
    return model, x_train, x_val, y_train, y_val, metrics


def create_etalon(model, x_val: pd.DataFrame, threshold: float):
    """Five deterministic samples; compare after serialization to detect drift."""
    examples = []
    for _, row in x_val.head(5).iterrows():
        features = {name: (str(row[name]) if name == "тип_вина" else float(row[name]))
                    for name in FEATURES}
        p = float(model.predict_proba(pd.DataFrame([features]))[0, 1])
        examples.append({
            "features": features,
            "probability": round(p, 8),
            "predicted_class": int(p >= threshold),
        })
    return examples


def run(data_dir=DATA_DIR, artifacts_dir=ARTIFACTS_DIR, *, demo=False, offline=False,
        model_version=MODEL_VERSION, threshold=THRESHOLD, save_previous=False):
    if not 0 < threshold < 1:
        raise ValueError("Порог должен быть между 0 и 1")
    data_dir, artifacts_dir = Path(data_dir), Path(artifacts_dir)
    data, origins, n_original = load_data(data_dir, demo=demo, offline=offline)
    model, x_train, x_val, y_train, y_val, metrics = train_model(data, threshold)
    fingerprint = sha256_of_files(origins)
    meta = {
        "model_version": model_version,
        "data_version": f"{'DEMO' if demo else 'UCI-186'}-sha256:{fingerprint[:16]}",
        "data_sha256": fingerprint,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "Wine Quality (UCI, CC BY 4.0)",
        "dataset_url": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "demo_mode": bool(demo),
        "algorithm": "LogisticRegression(class_weight='balanced', random_state=42)",
        "sklearn_version": sklearn.__version__,
        "target": "quality >= 7 (1=хорошее, 0=остальные)",
        "features": FEATURES,
        "threshold": threshold,
        "metrics": metrics,
        "data_counts": {
            "raw_rows": n_original, "duplicates_removed": n_original - len(data),
            "unique_rows": len(data), "train_rows": len(x_train), "validation_rows": len(x_val),
            "positive_train": int(y_train.sum()), "positive_validation": int(y_val.sum()),
        },
        "split": {"strategy": "stratified quality_good + wine_type", "test_size": 0.2, "random_state": 42},
        "business_goal": {"min_recall": 0.60, "min_precision": 0.40},
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if save_previous and (artifacts_dir / "model.joblib").exists():
        old_meta = json.loads((artifacts_dir / "meta.json").read_text(encoding="utf-8"))
        previous = artifacts_dir / "previous" / old_meta["model_version"]
        previous.mkdir(parents=True, exist_ok=True)
        for name in ("model.joblib", "meta.json", "etalon.json"):
            if (artifacts_dir / name).exists():
                shutil.copy2(artifacts_dir / name, previous / name)
        print(f"Предыдущая версия сохранена: {previous}")
    joblib.dump(model, artifacts_dir / "model.joblib", compress=3)
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifacts_dir / "etalon.json").write_text(
        json.dumps(create_etalon(model, x_val, threshold), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model_version": model_version, "demo_mode": demo, "data_counts": meta["data_counts"],
                      "metrics": metrics}, ensure_ascii=False, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description="Обучение Wine Quality и экспорт артефактов")
    parser.add_argument("--demo", action="store_true", help="Только офлайн-проверка, НЕ итоговая модель")
    parser.add_argument("--offline", action="store_true", help="Не скачивать данные")
    parser.add_argument("--version", default=MODEL_VERSION)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--save-previous", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.artifacts_dir, demo=args.demo, offline=args.offline,
        model_version=args.version, threshold=args.threshold, save_previous=args.save_previous)


if __name__ == "__main__":
    main()
