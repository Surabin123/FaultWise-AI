"""FaultWise AI API. Diagnostic text is generated from measured inputs only."""
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .features import SENSOR_COLS, feature_names

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("FAULTWISE_MODEL_DIR", "artifacts"))
app = FastAPI(title="FaultWise AI", description="Evidence-calibrated maintenance decision support prototype", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


class UnitHistory(BaseModel):
    unit_id: str = Field(min_length=1, max_length=80)
    readings: list[dict[str, float]] = Field(min_length=5, max_length=1000)


def load_bundle():
    path = MODEL_DIR / "faultwise.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/health")
def health():
    bundle = load_bundle()
    return {"status": "ok", "model_loaded": bundle is not None, "metrics": bundle["metrics"] if bundle else None}


@app.post("/api/analyze")
def analyze(history: UnitHistory):
    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(503, "No trained model. Train the C-MAPSS baseline and set FAULTWISE_MODEL_DIR.")
    try:
        values = np.array([[float(row[s]) for s in SENSOR_COLS] for row in history.readings], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise HTTPException(422, "Each reading must provide numeric sensor_1 through sensor_21")
    if not np.isfinite(values).all():
        raise HTTPException(422, "Readings must contain finite numeric sensor values")
    trail = values[-5:]
    means = trail.mean(axis=0)
    x = np.array([means], dtype=float)
    pred = max(0.0, float(bundle["model"].predict(x)[0]))
    uncertainty = float(bundle["residual_p90"])
    low, high = max(0.0, pred - uncertainty), pred + uncertainty
    metrics = bundle["metrics"]
    abstain = (high - low) > 60
    # Flag only if the observed latest value is far from the training sensor distribution.
    scaler_mean = getattr(bundle["model"], "_faultwise_sensor_mean", None)
    scaler_std = getattr(bundle["model"], "_faultwise_sensor_std", None)
    anomalies = []
    if scaler_mean is not None and scaler_std is not None:
        z = np.abs((values[-1] - scaler_mean) / np.maximum(scaler_std, 1e-8))
        anomalies = [{"sensor": SENSOR_COLS[i], "z_score": round(float(z[i]), 2)} for i in np.where(z >= 3)[0]]
    return {
        "unit_id": history.unit_id,
        "prediction": {"rul_cycles": round(pred, 1), "interval_90": [round(low, 1), round(high, 1)], "holdout_mae": round(metrics["mae"], 2), "holdout_rmse": round(metrics["rmse"], 2)},
        "evidence": {"observations": len(values), "sensors_reviewed": 21, "window": "mean of latest 5 readings", "out_of_distribution_sensors": anomalies},
        "recommendation": "INVESTIGATE" if anomalies else ("REVIEW MODEL UNCERTAINTY" if abstain else "MONITOR"),
        "abstain": abstain,
        "limitations": ["RUL interval uses empirical holdout residuals and may under-cover under distribution shift.", "Sensor flags use a 3-sigma training-set threshold.", "This prototype is not approved for operational maintenance decisions."],
        "explanation": "The estimate is based on the current sensor window and a unit-level holdout baseline. Sensor anomaly flags are statistical alerts, not confirmed faults.",
    }
