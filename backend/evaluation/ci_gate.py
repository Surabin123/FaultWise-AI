"""Check deterministic CI evidence while preserving the strict release gate."""

import argparse
import json
from pathlib import Path


def check_ci_result(run: dict) -> tuple[bool, str]:
    gate = run.get("release_gate") or {}
    regression = run.get("regression")
    if gate.get("status") not in {"pass", "fail", "inconclusive"}:
        return False, "release gate result is absent or invalid"
    if not isinstance(regression, dict):
        return False, "compatible baseline comparison is absent"
    if gate["status"] == "fail" or gate.get("failures") or regression.get("findings") or run.get("failed_cases"):
        return False, "quality failure or regression detected"
    if not run.get("cases") or run.get("evaluation_dataset_version") != "faultwise-synthetic-golden-v1":
        return False, "complete versioned golden evaluation is absent"
    expected_missing = {"verified ML evaluation unavailable", "live-provider evaluation unavailable"}
    missing = set(gate.get("missing", []))
    if missing - expected_missing:
        return False, f"unexpected missing checks: {sorted(missing - expected_missing)}"
    if gate["status"] == "inconclusive" and not missing:
        return False, "inconclusive gate has no missing evidence"
    return True, f"deterministic CI passed; full release gate: {gate['status']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    try:
        run = json.loads(args.evaluation.read_text(encoding="utf-8"))
        passed, message = check_ci_result(run)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        passed, message = False, f"invalid evaluation artifact: {exc}"
    print(message)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
