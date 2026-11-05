import math

import pytest

from faultwise_ml.metrics import nasa_prognostics_score, regression_metrics


def test_phm08_score_known_zero_error() -> None:
    assert nasa_prognostics_score([10], [10]) == 0.0


def test_phm08_score_known_early_and_late_errors() -> None:
    expected = math.e - 1.0
    assert nasa_prognostics_score([13], [0]) == pytest.approx(expected)
    assert nasa_prognostics_score([0], [10]) == pytest.approx(expected)


def test_metrics_mae_and_rmse_known_example() -> None:
    result = regression_metrics([0, 10], [3, 6])
    assert result["n_observations"] == 2
    assert result["mae"] == pytest.approx(3.5)
    assert result["rmse"] == pytest.approx(math.sqrt(12.5))


@pytest.mark.parametrize("truth,prediction", [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([-1], [0])])
def test_metrics_reject_invalid_inputs(truth: list[float], prediction: list[float]) -> None:
    with pytest.raises(ValueError):
        nasa_prognostics_score(truth, prediction)
