from pathlib import Path

import numpy as np

from faultwise_ml.data import load_cmapss
from faultwise_ml.preprocessing import add_capped_rul, feature_frame
from faultwise_ml.schema import FEATURE_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_fd001.txt"


def test_rul_labels_and_cap_are_per_engine() -> None:
    frame = load_cmapss(FIXTURE, variant="FD001")
    labels = add_capped_rul(frame, cap=4)
    assert labels.iloc[0] == 4
    assert labels.iloc[3] == 4
    assert labels.iloc[7] == 0
    assert labels.iloc[8] == 4


def test_features_exclude_ids_and_targets() -> None:
    frame = load_cmapss(FIXTURE, variant="FD001")
    features = feature_frame(frame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert "unit" not in features and "cycle" not in features and "rul" not in features
    assert np.isfinite(features.to_numpy()).all()
