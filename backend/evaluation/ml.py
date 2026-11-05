"""Read and verify saved Phase 1 metrics without changing the evaluation split."""

import hashlib
import json
from pathlib import Path
from typing import Any

from faultwise_ml import __version__ as ml_version


def load_ml_metrics(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    train_units = set(report["training_units"])
    test_units = set(report["test_units"])
    source = Path(report["dataset_source"])
    expected_hash = report["dataset_sha256"]
    integrity = bool(train_units and test_units and not train_units & test_units)
    if source.is_file():
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        integrity = integrity and actual_hash == expected_hash
    else:
        integrity = False
    metrics = report["test_metrics"]
    for key in ("mae", "rmse", "phm08_score_sum", "n_observations"):
        if key not in metrics:
            raise ValueError(f"Saved ML report lacks {key}")
    return {
        "dataset": report["dataset"], "dataset_kind": report["config"]["dataset_kind"],
        "model_id": f"faultwise-ml/{ml_version}/{report['config']['variant']}/seed-{report['config']['seed']}",
        "dataset_sha256": expected_hash, "test_protocol": report["test_protocol"],
        "model_config": report["config"], "trained_at_utc": report["trained_at_utc"],
        "train_engines": len(train_units), "held_out_engines": len(test_units),
        "integrity_verified": integrity, "official_benchmark_metrics": report["official_benchmark_metrics"],
        **{key: metrics[key] for key in ("mae", "rmse", "phm08_score_sum", "n_observations")},
    }
