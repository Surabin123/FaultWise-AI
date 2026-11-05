"""Check latest operating settings against verified training-engine ranges."""

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from faultwise_ml.data import DatasetValidationError, load_cmapss
from faultwise_ml.schema import SETTING_COLUMNS


def _verified_bounds(bundle: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    report = bundle.get("training_report", {})
    bounds = report.get("training_setting_bounds")
    if bounds is None:  # Backward-compatible path for Phase 1 artifacts saved before bounds existed.
        source = report.get("dataset_source")
        digest = report.get("dataset_sha256")
        units = report.get("training_units")
        variant = bundle.get("config", {}).get("variant")
        if not source or not digest or not units or not variant:
            return None
        path = Path(source)
        if not path.is_file():
            return None
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None
        if actual != digest:
            return None
        try:
            frame = load_cmapss(path, variant=variant)
        except (OSError, UnicodeError, DatasetValidationError):
            return None
        training = frame.loc[frame.unit.isin(units)]
        if training.empty or set(training.unit.unique()) != set(units):
            return None
        bounds = {}
        for setting in SETTING_COLUMNS:
            values = training[setting].dropna()
            bounds[setting] = None if values.empty else {
                "minimum": float(values.min()), "maximum": float(values.max())
            }
    if not isinstance(bounds, dict):
        return None
    verified: dict[str, dict[str, float]] = {}
    for setting in SETTING_COLUMNS:
        entry = bounds.get(setting)
        if not isinstance(entry, dict):
            return None
        try:
            minimum, maximum = float(entry["minimum"]), float(entry["maximum"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            return None
        verified[setting] = {"minimum": minimum, "maximum": maximum}
    return verified


def operating_coverage(latest: pd.Series, bundle: dict[str, Any]) -> tuple[str, list[str]]:
    """Return status and out-of-range setting names; extrema are empirical, not safety limits."""
    bounds = _verified_bounds(bundle)
    if bounds is None:
        return "unavailable", []
    outside: list[str] = []
    for setting in SETTING_COLUMNS:
        value = latest[setting]
        if pd.isna(value) or not math.isfinite(float(value)):
            outside.append(setting)
        elif not bounds[setting]["minimum"] <= float(value) <= bounds[setting]["maximum"]:
            outside.append(setting)
    return ("outside_training_range" if outside else "within_training_range"), outside
