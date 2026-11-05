"""Reproducible engine-level split, training, evaluation, and artifact saving."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .data import load_cmapss
from .metrics import regression_metrics
from .model import build_model
from .preprocessing import add_capped_rul, feature_frame
from .schema import FEATURE_COLUMNS, SETTING_COLUMNS


@dataclass(frozen=True)
class TrainingConfig:
    variant: str
    dataset_kind: str
    seed: int = 42
    test_fraction: float = 0.2
    rul_cap: int = 125
    estimator: str = "HistGradientBoostingRegressor"
    max_iter: int = 180
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    l2_regularization: float = 1.0


def split_by_engine(
    frame: pd.DataFrame, *, test_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole engine trajectories into train/test; indices never share units."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    units = frame["unit"].nunique()
    if units < 2:
        raise ValueError("At least two distinct engines are required for an engine-level split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_indices, test_indices = next(
        splitter.split(frame, groups=frame["unit"].to_numpy())
    )
    train_units = set(frame.iloc[train_indices]["unit"].unique())
    test_units = set(frame.iloc[test_indices]["unit"].unique())
    if not train_units or not test_units or train_units & test_units:
        raise RuntimeError("Engine-level split invariant failed")
    return train_indices, test_indices


def train_baseline(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
    dataset_kind: str,
    seed: int = 42,
    test_fraction: float = 0.2,
    rul_cap: int = 125,
) -> dict[str, Any]:
    """Train on complete units, evaluate held-out units, and save model/config/metrics.

    The test score is summed over all cycle rows from held-out complete
    trajectories. It is not the official one-terminal-prediction-per-engine
    PHM08 test protocol and is labeled accordingly in the saved report.
    """
    normalized_variant = variant.upper()
    if dataset_kind not in {"nasa_cmapss", "synthetic_fixture"}:
        raise ValueError("dataset_kind must be 'nasa_cmapss' or 'synthetic_fixture'")
    if dataset_kind == "synthetic_fixture" and normalized_variant != "FD001":
        raise ValueError("the bundled synthetic fixture represents FD001 schema only")
    config = TrainingConfig(
        variant=normalized_variant,
        dataset_kind=dataset_kind,
        seed=seed,
        test_fraction=test_fraction,
        rul_cap=rul_cap,
    )
    frame = load_cmapss(data_path, variant=normalized_variant)
    targets = add_capped_rul(frame, cap=rul_cap)
    train_indices, test_indices = split_by_engine(
        frame, test_fraction=test_fraction, seed=seed
    )
    x = feature_frame(frame)
    y = targets.to_numpy(dtype=float)
    model = build_model(seed=seed)

    # The imputer is inside the pipeline, so fit sees training rows only.
    model.fit(x.iloc[train_indices], y[train_indices])
    test_predictions = np.clip(
        model.predict(x.iloc[test_indices]), 0.0, float(rul_cap)
    )
    metrics = regression_metrics(y[test_indices], test_predictions)
    train_units = sorted(frame.iloc[train_indices]["unit"].unique().astype(int).tolist())
    test_units = sorted(frame.iloc[test_indices]["unit"].unique().astype(int).tolist())
    training_setting_bounds: dict[str, dict[str, float] | None] = {}
    for setting in SETTING_COLUMNS:
        values = x.iloc[train_indices][setting].dropna()
        training_setting_bounds[setting] = None if values.empty else {
            "minimum": float(values.min()), "maximum": float(values.max()),
        }
    report: dict[str, Any] = {
        "dataset": "NASA C-MAPSS" if dataset_kind == "nasa_cmapss" else "Synthetic C-MAPSS-shaped test fixture",
        "variant": normalized_variant,
        "dataset_source": str(Path(data_path).resolve()),
        "dataset_sha256": _sha256(data_path),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "test_protocol": "engine-level holdout; metrics over every cycle row of held-out complete trajectories",
        "test_units": test_units,
        "training_units": train_units,
        "training_setting_bounds": training_setting_bounds,
        "test_metrics": metrics,
        "synthetic_data": dataset_kind == "synthetic_fixture",
        "official_benchmark_metrics": False,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "config": asdict(config),
        "feature_columns": list(FEATURE_COLUMNS),
        "rul_cap": rul_cap,
        "training_report": report,
    }
    joblib.dump(bundle, destination / "faultwise_model.joblib")
    _write_json(destination / "training_config.json", asdict(config))
    _write_json(destination / "test_metrics.json", report)
    return report


def _sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="train_FDxxx.txt")
    parser.add_argument("--variant", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    parser.add_argument(
        "--data-kind",
        choices=["nasa_cmapss", "synthetic_fixture"],
        required=True,
        help="Explicit input provenance; fixture metrics are never NASA benchmark results.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/ml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--rul-cap", type=int, default=125)
    args = parser.parse_args()
    report = train_baseline(
        args.data,
        args.output,
        variant=args.variant,
        dataset_kind=args.data_kind,
        seed=args.seed,
        test_fraction=args.test_fraction,
        rul_cap=args.rul_cap,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
