"""Shared C-MAPSS parsing and feature construction."""
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = ID_COLS + SETTING_COLS + SENSOR_COLS


def read_cmapss(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
    frame = frame.dropna(axis=1, how="all")
    required = set(ALL_COLS)
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Expected a non-empty NASA C-MAPSS whitespace-delimited file")
    return frame


def add_labels(frame: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    out = frame.copy()
    end = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (end - out["cycle"]).clip(upper=cap)
    return out


def feature_names(window: int = 5) -> list[str]:
    return [f"{s}_mean_{window}" for s in SENSOR_COLS]


def make_features(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Trailing rolling means at each unit-cycle; no future values are used."""
    pieces = []
    for _, unit in frame.groupby("unit", sort=False):
        unit = unit.sort_values("cycle")
        x = unit[SENSOR_COLS].rolling(window, min_periods=1).mean()
        x.columns = feature_names(window)
        x["unit"] = unit["unit"].to_numpy()
        x["cycle"] = unit["cycle"].to_numpy()
        pieces.append(x)
    return pd.concat(pieces, ignore_index=True)
