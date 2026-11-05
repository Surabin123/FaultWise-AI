"""Safe, consistent API errors and domain exceptions."""


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class DatasetUnavailable(ApiError):
    def __init__(self, variant: str) -> None:
        super().__init__(503, "dataset_unavailable", f"Dataset {variant} is not installed.")


class DatasetInvalid(ApiError):
    def __init__(self, variant: str) -> None:
        super().__init__(503, "dataset_invalid", f"Dataset {variant} is present but failed validation.")


class EngineNotFound(ApiError):
    def __init__(self, engine_id: int, variant: str) -> None:
        super().__init__(404, "engine_not_found", f"Engine {engine_id} is not present in {variant}.")


class ModelUnavailable(ApiError):
    def __init__(self) -> None:
        super().__init__(503, "model_unavailable", "A trained model artifact is not available.")


class ModelArtifactInvalid(ApiError):
    def __init__(self) -> None:
        super().__init__(503, "model_artifact_invalid", "The configured model artifact could not be loaded.")


class ModelVariantMismatch(ApiError):
    def __init__(self, requested: str, trained: str) -> None:
        super().__init__(409, "model_variant_mismatch", f"The loaded model is for {trained}, not {requested}.")


class ModelInferenceFailed(ApiError):
    def __init__(self) -> None:
        super().__init__(500, "model_inference_failed", "The model could not produce a valid prediction.")


class ProviderUnavailable(ApiError):
    def __init__(self) -> None:
        super().__init__(503, "provider_unavailable", "A diagnostic report provider is not configured.")


class ProviderTimeout(ApiError):
    def __init__(self) -> None:
        super().__init__(504, "provider_timeout", "The diagnostic report provider timed out.")


class ReportGenerationFailed(ApiError):
    def __init__(self) -> None:
        super().__init__(502, "report_generation_failed", "The provider returned an unusable diagnostic report.")


class ReportValidationFailed(ApiError):
    def __init__(self) -> None:
        super().__init__(502, "report_validation_failed", "The diagnostic report failed evidence validation.")


class ReportStorageFailed(ApiError):
    def __init__(self) -> None:
        super().__init__(500, "report_storage_failed", "The validated diagnostic report could not be saved.")
