from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raman_classifier.inference import ModelPredictor


class PredictionRequest(BaseModel):
    spectrum: list[float] = Field(..., description="1D spectral intensity values.")
    axis: list[float] | None = Field(default=None, description="Optional axis or wavenumber vector.")
    top_k: int = Field(default=3, ge=1, le=10)


app = FastAPI(title="Raman Spectral Classifier API", version="0.1.0")
_predictor: ModelPredictor | None = None


@app.on_event("startup")
def load_predictor() -> None:
    global _predictor
    experiment_dir = os.environ.get("RAMAN_EXPERIMENT_DIR")
    if not experiment_dir:
        return
    _predictor = ModelPredictor.from_experiment_dir(experiment_dir)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": "yes" if _predictor is not None else "no"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Set RAMAN_EXPERIMENT_DIR before starting the API.",
        )
    return _predictor.predict_spectrum(payload.spectrum, axis=payload.axis, top_k=payload.top_k)
