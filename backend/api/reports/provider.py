"""Small Gemini REST adapter; tests inject a deterministic fake provider."""

from dataclasses import dataclass
from typing import Protocol

import httpx

from ..errors import ProviderTimeout, ReportGenerationFailed


@dataclass
class ProviderResult:
    text: str
    token_usage: dict[str, int] | None


class ReportProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str) -> ProviderResult: ...


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> ProviderResult:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        try:
            response = httpx.post(
                url,
                headers={"x-goog-api-key": self._api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
                },
                timeout=25.0,
            )
            response.raise_for_status()
            body = response.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            usage = body.get("usageMetadata", {})
            tokens = {key: int(usage[key]) for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount") if key in usage}
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty provider content")
            return ProviderResult(text=text, token_usage=tokens or None)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout() from exc
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ReportGenerationFailed() from exc
