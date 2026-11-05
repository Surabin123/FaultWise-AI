from pathlib import Path

import pytest

from faultwise_ml.data import DatasetValidationError, load_cmapss

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_fd001.txt"


def test_loads_and_sorts_synthetic_fixture() -> None:
    frame = load_cmapss(FIXTURE, variant="FD001")
    assert len(frame) == 32
    assert frame["unit"].nunique() == 4
    assert frame.groupby("unit")["cycle"].apply(list).iloc[0] == list(range(1, 9))


def test_missing_sensor_value_is_preserved_for_train_only_imputation(tmp_path: Path) -> None:
    source = FIXTURE.read_text(encoding="utf-8").replace("101.100", "NaN", 1)
    path = tmp_path / "train_FD001.txt"
    path.write_text(source, encoding="utf-8")
    frame = load_cmapss(path, variant="FD001")
    assert frame["sensor_1"].isna().sum() == 1


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("1 1 0.1", "expected 26 values"),
        ("1 1 x " + " ".join(["1"] * 23), "not numeric"),
        ("1 1 0.1 0.2 0.3 " + " ".join(["inf"] + ["1"] * 20), "must not be infinite"),
        ("1 1 0.1 0.2 0.3 " + " ".join(["-1"] + ["1"] * 20), "must be non-negative"),
    ],
)
def test_rejects_malformed_rows_and_invalid_feature_values(
    tmp_path: Path, replacement: str, message: str
) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(replacement + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        load_cmapss(path, variant="FD001")


def test_rejects_unsupported_variant_and_duplicate_observation(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Unsupported"):
        load_cmapss(FIXTURE, variant="FD005")
    first = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "duplicate.txt"
    path.write_text(first + "\n" + first + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Duplicate observation"):
        load_cmapss(path, variant="FD001")
