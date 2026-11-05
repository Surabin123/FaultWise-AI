"""Regression metrics and the PHM08 asymmetric prognostics score."""

from collections.abc import Sequence
import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _validated_arrays(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same nonzero length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite values")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("RUL values must be non-negative")
    return actual, predicted


def nasa_prognostics_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return the PHM08 C-MAPSS asymmetric score (sum, lower is better).

    With ``d = prediction - truth``, early/underestimates (d < 0) use
    exp(-d/13)-1 and late/overestimates (d >= 0) use exp(d/10)-1.
    """
    actual, predicted = _validated_arrays(y_true, y_pred)
    errors = predicted - actual
    terms = np.empty_like(errors)
    early = errors < 0
    with np.errstate(over="ignore"):
        terms[early] = np.expm1(-errors[early] / 13.0)
        terms[~early] = np.expm1(errors[~early] / 10.0)
    score = float(np.sum(terms, dtype=np.float64))
    if not math.isfinite(score):
        raise ValueError("PHM08 score overflowed for the supplied prediction errors")
    return score


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int]:
    """Return MAE, RMSE, and summed PHM08 score for aligned RUL observations."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return {
        "n_observations": int(actual.size),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "phm08_score_sum": nasa_prognostics_score(actual, predicted),
    }
