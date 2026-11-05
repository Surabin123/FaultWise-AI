"""Require a completed Phase 4 release gate before publishing."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    try:
        run = json.loads(args.evaluation.read_text(encoding="utf-8"))
        gate = run["release_gate"]
        status = gate["status"]
        if status != "pass" or gate.get("failures") or gate.get("missing"):
            print(f"Release blocked: {status}; failures={gate.get('failures')}; missing={gate.get('missing')}")
            raise SystemExit(1)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Release blocked: invalid evaluation artifact: {exc}")
        raise SystemExit(1) from exc
    print("Release quality gate passed")


if __name__ == "__main__":
    main()
