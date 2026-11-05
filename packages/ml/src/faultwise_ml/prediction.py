"""Persisted model loading and RUL prediction helpers."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import validate_frame
from .preprocessing import feature_frame


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a FaultWise joblib bundle and check its minimal schema."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")
    bundle = joblib.load(artifact_path)
    required = {"model", "config", "feature_columns"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid FaultWise model artifact; required keys are {sorted(required)}")
    return bundle


def predict_rows(model: Any, frame: pd.DataFrame, *, rul_cap: int = 125) -> np.ndarray:
    """Predict capped non-negative RUL for each validated observation row."""
    validated = validate_frame(frame)
    predictions = np.asarray(model.predict(feature_frame(validated)), dtype=float)
    if predictions.shape != (len(validated),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid RUL predictions")
    return np.clip(predictions, 0.0, float(rul_cap))


def predict_latest_per_unit(model: Any, frame: pd.DataFrame, *, rul_cap: int = 125) -> pd.DataFrame:
    """Predict once per unit from its latest available sensor observation."""
    validated = validate_frame(frame)
    latest = validated.groupby("unit", sort=True, as_index=False).tail(1).reset_index(drop=True)
    predictions = np.asarray(model.predict(feature_frame(latest)), dtype=float)
    if predictions.shape != (len(latest),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid RUL predictions")
    latest["predicted_rul"] = np.clip(predictions, 0.0, float(rul_cap))
    return latest[["unit", "cycle", "predicted_rul"]]
