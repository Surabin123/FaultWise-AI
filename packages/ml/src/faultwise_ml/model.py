"""Small, deterministic scikit-learn RUL baseline."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_model(*, seed: int = 42) -> Pipeline:
    """Create an imputer + histogram gradient boosting regression pipeline."""
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )),
    ])
