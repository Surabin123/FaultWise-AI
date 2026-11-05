"""Small in-process HTTP counters with bounded route labels."""

import logging
from threading import Lock
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("faultwise.http")


class HttpMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._values: dict[tuple[str, str, int], tuple[int, float]] = {}

    def record(self, method: str, route: str, status: int, seconds: float) -> None:
        key = (method, route, status)
        with self._lock:
            count, total = self._values.get(key, (0, 0.0))
            self._values[key] = (count + 1, total + seconds)

    def render(self) -> str:
        lines = [
            "# HELP faultwise_http_requests_total HTTP requests by route and status.",
            "# TYPE faultwise_http_requests_total counter",
            "# HELP faultwise_http_request_duration_seconds_sum Total request duration by route and status.",
            "# TYPE faultwise_http_request_duration_seconds_sum counter",
        ]
        with self._lock:
            values = sorted(self._values.items())
        for (method, route, status), (count, total) in values:
            labels = f'method="{method}",route="{route}",status="{status}"'
            lines.append(f"faultwise_http_requests_total{{{labels}}} {count}")
            lines.append(f"faultwise_http_request_duration_seconds_sum{{{labels}}} {total:.6f}")
        return "\n".join(lines) + "\n"


class ObservabilityMiddleware:
    def __init__(self, app: ASGIApp, metrics: HttpMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        status = 500

        async def send_with_status(message):  # type: ignore[no-untyped-def]
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            seconds = perf_counter() - started
            route = scope.get("route")
            route_name = getattr(route, "path", "unmatched")
            method = scope["method"]
            self.metrics.record(method, route_name, status, seconds)
            logger.info("request method=%s route=%s status=%d duration_ms=%.2f", method, route_name, status, seconds * 1000)


def install_observability(application: FastAPI) -> None:
    metrics = HttpMetrics()
    application.state.http_metrics = metrics
    application.add_middleware(ObservabilityMiddleware, metrics=metrics)

    @application.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
