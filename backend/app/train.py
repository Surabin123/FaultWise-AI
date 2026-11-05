"""Train and evaluate the C-MAPSS baseline with an engine-level split."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from .features import add_labels, feature_names, make_features, read_cmapss


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y_true
    terms = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    return float(np.sum(terms))


def train(data: Path, output: Path, seed: int = 42) -> dict:
    raw = add_labels(read_cmapss(data))
    features = make_features(raw)
    dataset = raw.merge(features, on=["unit", "cycle"], validate="one_to_one").sort_values(["unit", "cycle"])
    x = dataset[feature_names()]
    y = dataset["rul"].to_numpy()
    groups = dataset["unit"].to_numpy()
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(split.split(x, y, groups))
    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    model.fit(x.iloc[tr], y[tr])
    model._faultwise_sensor_mean = dataset.iloc[tr][[f"sensor_{i}" for i in range(1, 22)]].mean().to_numpy()
    model._faultwise_sensor_std = dataset.iloc[tr][[f"sensor_{i}" for i in range(1, 22)]].std().fillna(0).to_numpy()
    pred = np.clip(model.predict(x.iloc[va]), 0, None)
    residuals = np.abs(y[va] - pred)
    metrics = {
        "holdout_units": int(np.unique(groups[va]).size),
        "holdout_rows": int(len(va)),
        "mae": float(mean_absolute_error(y[va], pred)),
        "rmse": float(np.sqrt(mean_squared_error(y[va], pred))),
        "nasa_score": nasa_score(y[va], pred),
        "absolute_error_p90": float(np.quantile(residuals, 0.90)),
        "split_seed": seed,
        "rul_cap": 125,
        "window": 5,
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names(), "residual_p90": metrics["absolute_error_p90"], "metrics": metrics}, output / "faultwise.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.seed), indent=2))
