"""Local C-MAPSS dataset operations backed by the Phase 1 loader."""

import logging
import hashlib
from pathlib import Path

import pandas as pd

from faultwise_ml.data import DatasetValidationError, load_cmapss
from faultwise_ml.schema import SENSOR_COLUMNS, SETTING_COLUMNS

from ..errors import DatasetInvalid, DatasetUnavailable, EngineNotFound
from ..schemas import DatasetMetadata, DatasetState, DatasetSummary, DatasetVariant

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _path(self, variant: DatasetVariant) -> Path:
        return self._data_dir / f"train_{variant.value}.txt"

    def sha256(self, variant: DatasetVariant) -> str:
        path = self._path(variant)
        if not path.is_file():
            raise DatasetUnavailable(variant.value)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load(self, variant: DatasetVariant) -> pd.DataFrame:
        path = self._path(variant)
        if not path.is_file():
            raise DatasetUnavailable(variant.value)
        try:
            return load_cmapss(path, variant=variant.value)
        except (DatasetValidationError, OSError, UnicodeError) as exc:
            logger.warning("Dataset %s failed loading/validation: %s", variant.value, exc)
            raise DatasetInvalid(variant.value) from exc

    def summary(self, variant: DatasetVariant) -> DatasetSummary:
        if not self._path(variant).is_file():
            return DatasetSummary(variant=variant, status=DatasetState.UNAVAILABLE)
        try:
            frame = self._load(variant)
        except DatasetInvalid:
            return DatasetSummary(variant=variant, status=DatasetState.INVALID)
        return DatasetSummary(
            variant=variant,
            status=DatasetState.AVAILABLE,
            engine_count=int(frame["unit"].nunique()),
            row_count=len(frame),
        )

    def list_datasets(self) -> list[DatasetSummary]:
        return [self.summary(variant) for variant in DatasetVariant]

    def metadata(self, variant: DatasetVariant) -> DatasetMetadata:
        frame = self._load(variant)
        return DatasetMetadata(
            variant=variant,
            status=DatasetState.AVAILABLE,
            engine_count=int(frame["unit"].nunique()),
            row_count=len(frame),
            operating_settings=list(SETTING_COLUMNS),
            sensors=list(SENSOR_COLUMNS),
            cycle_range=[int(frame["cycle"].min()), int(frame["cycle"].max())],
        )

    def engine_ids(self, variant: DatasetVariant) -> list[int]:
        frame = self._load(variant)
        return sorted(frame["unit"].unique().astype(int).tolist())

    def trajectory(self, variant: DatasetVariant, engine_id: int) -> pd.DataFrame:
        frame = self._load(variant)
        trajectory = frame.loc[frame["unit"] == engine_id].copy()
        if trajectory.empty:
            raise EngineNotFound(engine_id, variant.value)
        return trajectory.reset_index(drop=True)
