"""Environment-backed API settings."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_path: Path
    default_variant: str = "FD001"
    llm_provider: str = "none"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    report_dir: Path = Path("artifacts/reports")
    evaluation_dir: Path = Path("artifacts/evaluation")
    ml_metrics_path: Path = Path("artifacts/ml/test_metrics.json")
    public_api_base: str = ""
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("FAULTWISE_DATA_DIR", "data")),
            model_path=Path(os.getenv("FAULTWISE_MODEL_PATH", "artifacts/ml/faultwise_model.joblib")),
            default_variant=os.getenv("FAULTWISE_DEFAULT_VARIANT", "FD001").upper(),
            llm_provider=os.getenv("FAULTWISE_LLM_PROVIDER", "none").lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("FAULTWISE_GEMINI_MODEL", "gemini-2.5-flash"),
            report_dir=Path(os.getenv("FAULTWISE_REPORT_DIR", "artifacts/reports")),
            evaluation_dir=Path(os.getenv("FAULTWISE_EVALUATION_DIR", "artifacts/evaluation")),
            ml_metrics_path=Path(os.getenv("FAULTWISE_ML_METRICS_PATH", "artifacts/ml/test_metrics.json")),
            public_api_base=os.getenv("FAULTWISE_PUBLIC_API_BASE", "").rstrip("/"),
            allowed_origins=tuple(origin.strip() for origin in os.getenv("FAULTWISE_ALLOWED_ORIGINS", "").split(",") if origin.strip()),
        )
