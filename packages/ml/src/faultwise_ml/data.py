"""Strict loading and validation for NASA C-MAPSS whitespace text files."""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, INPUT_COLUMNS, SENSOR_COLUMNS

_MISSING_TOKENS: Final = {"nan", "na", "null", "none"}


class DatasetValidationError(ValueError):
    """Raised when C-MAPSS input does not satisfy the declared data contract."""


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed C-MAPSS frame and return it sorted by engine and cycle.

    Missing feature values are permitted and later imputed by a train-fitted
    median imputer. Missing identifiers, infinities, non-numeric values,
    duplicate cycles, extra/missing columns, or invalid identifiers are errors.
    """
    expected = list(INPUT_COLUMNS)
    actual = list(frame.columns)
    if actual != expected:
        missing = [name for name in expected if name not in actual]
        unexpected = [name for name in actual if name not in expected]
        raise DatasetValidationError(
            f"Unexpected C-MAPSS columns/order; missing={missing}, unexpected={unexpected}. "
            "Expected the canonical 26-column positional schema."
        )
    if frame.empty:
        raise DatasetValidationError("Dataset contains no observations")

    out = frame.copy()
    for column in expected:
        try:
            out[column] = pd.to_numeric(out[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(f"Column {column!r} contains a non-numeric value") from exc

    identifiers = out[["unit", "cycle"]].to_numpy(dtype=float)
    if not np.isfinite(identifiers).all():
        raise DatasetValidationError("unit and cycle identifiers must be finite and non-missing")
    if (identifiers <= 0).any() or not np.equal(identifiers, np.floor(identifiers)).all():
        raise DatasetValidationError("unit and cycle identifiers must be positive integers")

    features = out[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if np.isinf(features).any():
        raise DatasetValidationError("operating settings and sensors may be finite or missing, not infinite")
    sensor_values = out[list(SENSOR_COLUMNS)].to_numpy(dtype=float)
    if (sensor_values < 0).any():
        raise DatasetValidationError("C-MAPSS sensor measurements must be non-negative or missing")
    if out.duplicated(["unit", "cycle"]).any():
        duplicate = out.loc[out.duplicated(["unit", "cycle"], keep=False), ["unit", "cycle"]].iloc[0]
        raise DatasetValidationError(
            f"Duplicate observation for unit={int(duplicate.unit)}, cycle={int(duplicate.cycle)}"
        )
    counts = out.groupby("unit")["cycle"].size()
    if (counts < 2).any():
        units = counts[counts < 2].index.astype(int).tolist()
        raise DatasetValidationError(f"Each engine needs at least two cycles; invalid units={units}")

    out[["unit", "cycle"]] = out[["unit", "cycle"]].astype("int64")
    return out.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)


def _parse_numeric(token: str, *, row: int, column: str) -> float:
    if token.strip().lower() in _MISSING_TOKENS:
        return float("nan")
    try:
        value = float(token)
    except ValueError as exc:
        raise DatasetValidationError(
            f"Malformed row {row}: {column} value {token!r} is not numeric"
        ) from exc
    if np.isinf(value):
        raise DatasetValidationError(f"Malformed row {row}: {column} must not be infinite")
    return value


def load_cmapss(path: str | Path, *, variant: str) -> pd.DataFrame:
    """Load one NASA C-MAPSS ``train_FDxxx.txt`` or ``test_FDxxx.txt`` file.

    The source files have no header: unit, cycle, three settings, then 21
    sensors. Blank lines are ignored. Every nonblank line must have exactly 26
    whitespace-separated fields. ``NaN``/``NA``/``null`` feature tokens are
    retained as missing values for train-only median imputation.
    """
    normalized_variant = variant.upper()
    if normalized_variant not in {"FD001", "FD002", "FD003", "FD004"}:
        raise DatasetValidationError(f"Unsupported C-MAPSS variant {variant!r}; use FD001–FD004")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"C-MAPSS data file does not exist: {source}")

    rows: list[list[float]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != len(INPUT_COLUMNS):
                raise DatasetValidationError(
                    f"Malformed row {line_number}: expected {len(INPUT_COLUMNS)} values, found {len(tokens)}"
                )
            rows.append([
                _parse_numeric(token, row=line_number, column=column)
                for token, column in zip(tokens, INPUT_COLUMNS, strict=True)
            ])
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    return validate_frame(frame)
