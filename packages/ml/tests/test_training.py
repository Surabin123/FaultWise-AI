from pathlib import Path

import numpy as np
import pytest

from faultwise_ml.data import load_cmapss
from faultwise_ml.model import build_model
from faultwise_ml.prediction import load_artifact, predict_latest_per_unit, predict_rows
from faultwise_ml.preprocessing import add_capped_rul, feature_frame
from faultwise_ml.training import split_by_engine, train_baseline

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_fd001.txt"


def test_engine_split_has_no_unit_leakage_and_is_deterministic() -> None:
    frame = load_cmapss(FIXTURE, variant="FD001")
    first_train, first_test = split_by_engine(frame, test_fraction=0.25, seed=7)
    second_train, second_test = split_by_engine(frame, test_fraction=0.25, seed=7)
    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_test, second_test)
    assert set(frame.iloc[first_train].unit).isdisjoint(set(frame.iloc[first_test].unit))


def test_pipeline_imputer_fits_only_on_training_rows() -> None:
    frame = load_cmapss(FIXTURE, variant="FD001")
    train_indices, test_indices = split_by_engine(frame, test_fraction=0.25, seed=7)
    x = feature_frame(frame)
    test_unit = frame.iloc[test_indices].unit.iloc[0]
    test_row = frame.index[frame.unit == test_unit][0]
    x.loc[test_row, "sensor_1"] = 1_000_000.0
    model = build_model(seed=7)
    model.fit(x.iloc[train_indices], add_capped_rul(frame).iloc[train_indices])
    fitted_median = model.named_steps["preprocess"].named_steps["imputer"].statistics_[3]
    expected = np.nanmedian(x.iloc[train_indices]["sensor_1"].to_numpy())
    assert fitted_median == pytest.approx(expected)
    assert fitted_median < 1_000_000.0


def test_training_persists_reproducible_artifact_and_marks_fixture(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    report = train_baseline(
        FIXTURE, output, variant="FD001", dataset_kind="synthetic_fixture", seed=7,
        test_fraction=0.25,
    )
    assert report["synthetic_data"] is True
    assert report["official_benchmark_metrics"] is False
    assert "Synthetic" in report["dataset"]
    assert report["dataset_sha256"]
    assert (output / "faultwise_model.joblib").is_file()
    assert (output / "training_config.json").is_file()
    assert (output / "test_metrics.json").is_file()
    bundle = load_artifact(output / "faultwise_model.joblib")
    frame = load_cmapss(FIXTURE, variant="FD001")
    predictions = predict_rows(bundle["model"], frame)
    latest = predict_latest_per_unit(bundle["model"], frame)
    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()
    assert predictions.min() >= 0
    assert predictions.max() <= 125
    assert len(latest) == frame.unit.nunique()


def test_training_is_deterministic_on_fixture(tmp_path: Path) -> None:
    first = train_baseline(FIXTURE, tmp_path / "one", variant="FD001", dataset_kind="synthetic_fixture", seed=12)
    second = train_baseline(FIXTURE, tmp_path / "two", variant="FD001", dataset_kind="synthetic_fixture", seed=12)
    assert first["test_metrics"] == second["test_metrics"]
    assert first["training_units"] == second["training_units"]
    assert first["test_units"] == second["test_units"]


def test_operating_bounds_use_training_engines_only(tmp_path: Path) -> None:
    frame = load_cmapss(FIXTURE, variant="FD001")
    _, test_indices = split_by_engine(frame, test_fraction=0.25, seed=7)
    held_out_unit = int(frame.iloc[test_indices].unit.iloc[0])
    frame.loc[frame.unit == held_out_unit, "setting_1"] = 1e9
    source = tmp_path / "synthetic_modified.txt"
    frame.to_csv(source, sep=" ", header=False, index=False)
    report = train_baseline(source, tmp_path / "artifact", variant="FD001", dataset_kind="synthetic_fixture", seed=7, test_fraction=0.25)
    assert report["training_setting_bounds"]["setting_1"]["maximum"] < 1e9
    assert held_out_unit in report["test_units"]
