"""RUL target creation and leakage-safe estimator preprocessing."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .schema import FEATURE_COLUMNS


def add_capped_rul(frame: pd.DataFrame, *, cap: int = 125) -> pd.Series:
    """Create piecewise-linear RUL labels for complete run-to-failure units."""
    if cap <= 0:
        raise ValueError("RUL cap must be a positive integer")
    terminal_cycles = frame.groupby("unit")["cycle"].transform("max")
    return (terminal_cycles - frame["cycle"]).clip(upper=cap).astype(float)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only observed settings/sensors; identifiers and labels are excluded."""
    return frame.loc[:, list(FEATURE_COLUMNS)].copy()


def build_preprocessor() -> Pipeline:
    """Build a train-fitted numeric imputer; fit only inside model training."""
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
