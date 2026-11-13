"""FastAPI /health /predict /model-info. Run: python -m uvicorn app.main:app --port 8077"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import Health, ModelInfo, Prediction, WineFeatures

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
STATE: dict = {"model": None, "meta": None, "error": None}


def load_artifacts():
    root = Path(os.environ.get("WINE_ARTIFACTS_DIR", str(DEFAULT_ARTIFACT_DIR)))
    try:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        model = joblib.load(root / "model.joblib")
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("Артефакт модели не поддерживает predict_proba")
    except Exception as exc:
        STATE.update({"model": None, "meta": None, "error": str(exc)})
    else:
        STATE.update({"model": model, "meta": meta, "error": None})


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.update({"model": None, "meta": None, "error": None})


app = FastAPI(title="Wine Quality — модель как сервис (группа 9)", version="1.0.0",
              description="Вероятность quality ≥ 7 для красного и белого вина на основе 11 измерений",
              lifespan=lifespan)


@app.get("/health", response_model=Health, tags=["Служебные"])
def health():
    ready = STATE["model"] is not None and STATE["meta"] is not None
    return Health(status="ok" if ready else "degraded", model_loaded=ready,
                  model_version=STATE["meta"]["model_version"] if ready else "unknown")


@app.post("/predict", response_model=Prediction, tags=["Прогноз"],
          responses={422: {"description": "Неверный запрос"}, 503: {"description": "Модель не загружена"}})
def predict(features: WineFeatures):
    if STATE["model"] is None or STATE["meta"] is None:
        raise HTTPException(status_code=503, detail="Модель недоступна: сначала выполните python train.py")
    frame = pd.DataFrame([features.model_dump(by_alias=True)])
    probability = float(STATE["model"].predict_proba(frame)[0, 1])
    threshold = float(STATE["meta"]["threshold"])
    predicted = int(probability >= threshold)
    return Prediction(predicted_class=predicted, label="хорошее" if predicted else "остальное",
                      probability=round(probability, 6), threshold=threshold,
                      model_version=STATE["meta"]["model_version"])


@app.get("/model-info", response_model=ModelInfo, tags=["Служебные"],
         responses={503: {"description": "Метаданные недоступны"}})
def model_info():
    if STATE["meta"] is None:
        raise HTTPException(status_code=503, detail="Метаданные модели недоступны")
    return ModelInfo(**STATE["meta"])
