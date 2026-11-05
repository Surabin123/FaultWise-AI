from backend.evaluation.ci_gate import check_ci_result


def _run():
    return {
        "evaluation_dataset_version": "faultwise-synthetic-golden-v1",
        "cases": [{"case_id": "a", "passed": True}],
        "failed_cases": [],
        "regression": {"findings": []},
        "release_gate": {
            "status": "inconclusive",
            "failures": [],
            "missing": ["verified ML evaluation unavailable", "live-provider evaluation unavailable"],
        },
    }


def test_expected_missing_live_and_ml_allows_development_ci():
    passed, message = check_ci_result(_run())
    assert passed
    assert "inconclusive" in message


def test_regression_blocks_ci():
    run = _run()
    run["regression"]["findings"].append({"metric": "new_failed_case"})
    assert not check_ci_result(run)[0]


def test_unexpected_missing_check_blocks_ci():
    run = _run()
    run["release_gate"]["missing"].append("required case missing")
    assert not check_ci_result(run)[0]


def test_missing_comparison_blocks_ci():
    run = _run()
    run["regression"] = None
    assert not check_ci_result(run)[0]
