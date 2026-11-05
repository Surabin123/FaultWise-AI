"""Compare only compatible evaluations, with explicit metric direction and deltas."""

from typing import Any

_RULES = {
    "schema_validity": ("decrease", 0.0),
    "evidence_reference_validity": ("decrease", 0.0),
    "grounded_claim_rate_structural": ("decrease", 0.0),
    "unsupported_claim_rate_rule": ("increase", 0.0),
    "abstention.missed_abstention_rate": ("increase", 0.0),
    "ml_metrics.mae": ("increase", 1.0),
    "ml_metrics.rmse": ("increase", 1.0),
    "ml_metrics.phm08_score_sum": ("increase", 100.0),
}


def _get(run: dict[str, Any], path: str) -> float | None:
    node: Any = run if path.startswith("ml_metrics.") else run.get("metrics", {})
    for key in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return float(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def compare(previous: dict[str, Any], current: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    if previous.get("evaluation_dataset_version") != current.get("evaluation_dataset_version"):
        raise ValueError("Cannot compare different golden dataset versions")
    if previous.get("evaluation_config", {}).get("categories") != current.get("evaluation_config", {}).get("categories"):
        raise ValueError("Cannot compare different category selections")
    failures = set(previous.get("failed_cases", []))
    new_failures = sorted(set(current.get("failed_cases", [])) - failures)
    resolved = sorted(failures - set(current.get("failed_cases", [])))
    findings: list[dict[str, Any]] = []
    for metric, (direction, default_threshold) in _RULES.items():
        if metric.startswith("ml_metrics."):
            old_ml, new_ml = previous.get("ml_metrics"), current.get("ml_metrics")
            if not old_ml or not new_ml or any(old_ml.get(key) != new_ml.get(key) for key in ("dataset_sha256", "test_protocol", "dataset_kind")):
                continue
        old, new = _get(previous, metric), _get(current, metric)
        if old is None or new is None:
            continue
        delta = round(new - old, 8)
        threshold = (thresholds or {}).get(metric, default_threshold)
        if (direction == "increase" and delta > threshold) or (direction == "decrease" and delta < -threshold):
            findings.append({
                "metric": metric, "previous": old, "current": new,
                "difference": delta, "threshold": threshold,
                "affected_cases": new_failures,
                "severity": "critical" if metric in {"unsupported_claim_rate_rule", "abstention.missed_abstention_rate"} else "high",
            })
    for case_id in new_failures:
        case = next((case for case in current.get("cases", []) if case["case_id"] == case_id), None)
        findings.append({
            "metric": "new_failed_case", "previous": False, "current": True,
            "difference": 1, "threshold": 0, "affected_cases": [case_id],
            "severity": case.get("severity", "high") if case else "high",
        })
    return {"previous_run_id": previous.get("run_id"), "current_run_id": current.get("run_id"), "new_failed_cases": new_failures, "resolved_cases": resolved, "findings": findings}
