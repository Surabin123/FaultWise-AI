"""Cached access to the persisted Phase 1 pipeline and inference helpers."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from faultwise_ml import __version__ as ml_version
from faultwise_ml.prediction import load_artifact, predict_latest_per_unit
from faultwise_ml.schema import FEATURE_COLUMNS
import pandas as pd

from ..errors import ModelArtifactInvalid, ModelInferenceFailed, ModelUnavailable, ModelVariantMismatch
from ..schemas import DatasetVariant, ModelState, ModelStatus

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, artifact_path: Path) -> None:
        self._artifact_path = artifact_path
        self._bundle: dict[str, Any] | None = None
        self._loaded_mtime_ns: int | None = None
        self._load_error = False

    def _bundle_or_raise(self) -> dict[str, Any]:
        if not self._artifact_path.is_file():
            self._bundle = None
            self._loaded_mtime_ns = None
            self._load_error = False
            raise ModelUnavailable()
        mtime_ns = self._artifact_path.stat().st_mtime_ns
        if self._bundle is not None and self._loaded_mtime_ns == mtime_ns:
            return self._bundle
        if self._load_error and self._loaded_mtime_ns == mtime_ns:
            raise ModelArtifactInvalid()
        try:
            bundle = load_artifact(self._artifact_path)
            if bundle["feature_columns"] != list(FEATURE_COLUMNS):
                raise ValueError("artifact feature order does not match the Phase 1 schema")
            if not isinstance(bundle.get("config"), dict) or bundle["config"].get("variant") not in {
                variant.value for variant in DatasetVariant
            }:
                raise ValueError("artifact training configuration is missing a supported variant")
        except Exception as exc:
            self._bundle = None
            self._loaded_mtime_ns = mtime_ns
            self._load_error = True
            logger.exception("Could not load configured FaultWise model artifact")
            raise ModelArtifactInvalid() from exc
        self._bundle = bundle
        self._loaded_mtime_ns = mtime_ns
        self._load_error = False
        logger.info("Loaded FaultWise model artifact for %s", bundle["config"]["variant"])
        return bundle

    def status(self) -> ModelStatus:
        if not self._artifact_path.is_file():
            self._bundle = None
            self._loaded_mtime_ns = None
            return ModelStatus(status=ModelState.UNAVAILABLE)
        try:
            bundle = self._bundle_or_raise()
        except ModelArtifactInvalid:
            return ModelStatus(status=ModelState.INVALID)
        config = bundle["config"]
        report = bundle.get("training_report", {})
        trained_at = report.get("trained_at_utc")
        if isinstance(trained_at, str):
            try:
                trained_at = datetime.fromisoformat(trained_at)
            except ValueError:
                trained_at = None
        return ModelStatus(
            status=ModelState.LOADED,
            model_id=self.model_id(bundle),
            variant=DatasetVariant(config["variant"]),
            dataset_kind=config.get("dataset_kind", "unknown"),
            trained_at_utc=trained_at,
        )

    def predict_latest(self, frame: pd.DataFrame, variant: DatasetVariant) -> tuple[float, dict[str, Any]]:
        bundle = self._bundle_or_raise()
        trained_variant = bundle["config"]["variant"]
        if trained_variant != variant.value:
            raise ModelVariantMismatch(variant.value, trained_variant)
        try:
            latest = predict_latest_per_unit(
                bundle["model"], frame, rul_cap=int(bundle.get("rul_cap", 125))
            )
            if len(latest) != 1:
                raise ValueError("engine trajectory did not produce exactly one endpoint prediction")
        except Exception as exc:
            logger.exception("RUL inference failed for a validated engine trajectory")
            raise ModelInferenceFailed() from exc
        return float(latest.iloc[0]["predicted_rul"]), bundle

    @staticmethod
    def model_id(bundle: dict[str, Any]) -> str:
        config = bundle["config"]
        return f"faultwise-ml/{ml_version}/{config['variant']}/seed-{config.get('seed', 'unknown')}"
