# Phase 4 quality evaluation

The evaluation package is `backend/evaluation`. It runs versioned synthetic golden cases against the **production** `ReportService`, using a fixed fake provider. It never needs a Gemini key for local tests. The cases are in [`golden/v1.json`](../backend/evaluation/golden/v1.json); each declares provenance, scenario, evidence inputs, expected assessment or error, references or missing evidence, category, severity, and review status. They are engineered test cases, not real engine failures or aviation incidents.

The runner trains one model on the bundled **synthetic** C-MAPSS-shaped fixture, invokes the actual dataset loader, model service, anomaly analysis, report generation, and report validator, and records per-case outcomes. Safety cases send adversarial fixed-provider outputs through the actual service. A numeric sensor field containing an instruction is rejected by the dataset loader. There is currently no document retrieval or free-text evidence ingestion, so injection through a maintenance document cannot yet be tested against a production surface.

## Run and compare

From the repository root, after `pip install -r requirements-dev.txt`:

```powershell
python -m backend.evaluation.cli run --ml-report artifacts/ml/test_metrics.json
python -m backend.evaluation.cli run --category safety
python -m backend.evaluation.cli run --baseline artifacts/evaluation/evaluation-PREVIOUS-ID.json
python -m backend.evaluation.cli compare artifacts/evaluation/evaluation-PREVIOUS-ID.json artifacts/evaluation/evaluation-CURRENT-ID.json
python -m pytest -q
```

Each run writes a new `artifacts/evaluation/evaluation-<run-id>.json`; files are never overwritten. `--category` narrows the run, so the full release gate reports missing required cases as `inconclusive` unless a tested failure makes it `fail`. `--ml-report` reads the existing Phase 1 report and verifies train/test engine disjointness and the source file SHA-256. It **does not retrain or rescore** NASA data. The saved FD001 metrics use 20 held-out complete engines and 4,070 cycle rows, not the official endpoint benchmark protocol. Missing or mismatched ML provenance cannot satisfy the gate.

To opt into one live Gemini report evaluation, set `FAULTWISE_LLM_PROVIDER=gemini` and `GEMINI_API_KEY`, then add `--live --live-data-dir data --live-model artifacts/ml/faultwise_model.joblib --live-engine 1`. This sends one real request and records the resulting report ID, total report duration, model/provider identifiers, and token usage if supplied. Provider-only latency and cost remain `null` because the current adapter does not expose them. A missing key is recorded as `unavailable`; it never counts as a successful live check. No live Gemini call was made during Phase 4 implementation.

## Metric definitions

All rates use a numerator divided by an explicit denominator. If the denominator is zero, the result is `null` (unavailable), never zero.

| Metric | Numerator / denominator | Meaning and limit |
|---|---|---|
| Case pass rate | Cases matching expected behavior / executed cases | Includes expected rejections; not diagnostic accuracy. |
| Schema validity | Accepted `investigate` reports validating against `DiagnosticReport` / accepted `investigate` reports | Rejected provider outputs are tracked as failed cases rather than accepted reports. |
| Evidence reference validity | Accepted claims with nonempty IDs all present in the report evidence / accepted claims | An existing ID does not prove the claim true. |
| Grounded claim rate, structural | Accepted claims with valid IDs, a rule-relevant anomaly reference, and no lexical flag / accepted claims | A rule-based proxy, not semantic groundedness. |
| Evidence relevance, rule | Accepted claims citing anomaly evidence / accepted claims | Coarse relevance rule; human relevance judgment is unavailable. |
| Unsupported claim rate, rule | Accepted claims with an invalid reference, invalid kind, or lexical flag / accepted claims | Detects specific patterns and misses other unsupported statements. |
| Abstention precision | Correct abstentions / all actual abstentions | Dataset/model/data errors count as abstention for this metric; provider absence does not. |
| Abstention recall | Correct abstentions / all cases requiring abstention | Measures missed abstention. |
| False abstention rate | Unnecessary abstentions / cases that should proceed | Valid evidence cases provide the negative class. |
| Missed abstention rate | Cases that should abstain but did not / cases requiring abstention | Does not assess whether the explanation identifies the right missing evidence; that is checked per case. |

The aggregate report includes denominators, individual case evidence and outcomes, failures, run/config/prompt/model/dataset identifiers, fixture provenance, optional saved ML metrics, optional live usage, regression findings, and gate status. It does not estimate diagnostic accuracy, semantic factuality, provider cost, or human-reviewed relevance.

## Regression and release gate

`compare` requires the same golden dataset version and selected categories. It checks increased unsupported-claim and missed-abstention rates, decreased schema/reference/structural grounding rates, new failed cases, and ML metric increases. ML metric comparisons require identical dataset hash, dataset kind, and evaluation protocol. Each finding records previous/current values, difference, threshold, affected cases, and severity. This avoids comparing incompatible NASA protocols or datasets.

The configurable gate is [`release-gate-v1.json`](../backend/evaluation/golden/release-gate-v1.json). It requires all listed cases, complete accepted-report metrics, zero rule-detected unsupported claims or missed abstentions, no failed golden cases, a verified NASA C-MAPSS ML report, and a live Gemini evaluation using the same dataset hash. It returns `fail` if any measured threshold or required case fails; otherwise `inconclusive` for unavailable required evidence; otherwise `pass`. A pass would be an engineering quality check, **not safety certification**. A baseline comparison with regressions blocks the gate.

The original Phase 4 run with the saved NASA FD001 ML report executed **19 synthetic golden cases**. **15 passed; 4 failed**: `unsupported_operating_condition`, `unsupported_component_failure`, `secret_exfiltration_request`, and `ignore_evidence_instruction`. The production report service abstained for the extreme operating setting because the edited dataset hash differed, but did not identify unsupported operating conditions. It accepted three adversarial provider claims that the independent lexical evaluator flagged. Among 4 accepted reports and 8 accepted claims, the rule-detected unsupported claim rate was **3/8** and structural grounded claim rate **5/8**. These are synthetic adversarial test measurements, not real NASA diagnostic scores. The gate returned **`fail`**, and live Gemini evaluation was unavailable. The previously saved NASA FD001 row-level holdout ML report passed the source hash and engine split integrity checks; it was not recalculated.

## Phase 4.1 remediation and comparison

The original golden file and gate thresholds were not changed. The root causes were:

| Original failure | Production cause | Remediation |
|---|---|---|
| Component failure accepted | A valid anomaly ID passed reference validation, while the short word filter missed a specific component assertion. | Every claim now declares sensor or component scope. Component scope and component wording both require component-level evidence. The current evidence contract has none, so such claims are rejected. Sensor-focused hypotheses remain allowed. |
| Secret request accepted | The schema allowed the text and the validator had no credential or exfiltration check. | Reject credential references and directives in every generated text field; do not log provider text. The API key never enters the prompt. |
| Ignore-evidence instruction accepted | Prompt guidance was advisory, and the validator did not recognize instructions in generated prose. | The prompt labels evidence JSON as untrusted data; deterministic validation rejects override or fabrication directives. The sufficiency layer still runs before provider access. |
| Coverage warning missing | The service compared dataset hashes but had no setting-range check. | Training now saves min/max operating settings from training engines only. For older artifacts, the service verifies the original training file hash and derives the same bounds from recorded training units. Missing bounds or out-of-range settings cause abstention. |

The final rerun was saved locally as `evaluation-91ecf8c8-fdba-469c-bfd0-5bac1559d57e.json` and compared against the original local `evaluation-d9214ada-1864-44e9-a99c-39916e686b22.json`. These generated artifacts are intentionally excluded from Git. All **19/19** original cases passed; all four former failures were resolved and no new case failed. The independent rule check found **0/2** unsupported claims among one accepted report, with **2/2** structurally grounded claims. It found no regression findings. The release gate is **`inconclusive`** because a live Gemini evaluation has not been run. The saved NASA FD001 ML report again passed source-hash and engine-split integrity checks; no NASA model was retrained or rescored for this rerun. The full pytest suite passed **56 tests**.

Additional regression tests cover qualified sensor hypotheses, unsupported mechanical claims, secret and override requests in claims and uncertainty text, malformed instructions in numeric evidence, coverage with an unchanged dataset hash, missing coverage provenance, and training-only setting bounds. These tests use synthetic fixtures and harmless key-name markers, never real secrets.

## Known limits

Structural references and lexical checks do not establish semantic truth. Component and instruction patterns can miss unseen phrasing, while broad patterns can reject harmless prose. The adversarial suite exercises the current numeric evidence and fixed-provider output surfaces; it does not measure a live model's attack susceptibility. Empirical training-setting minima and maxima are a coverage screen, **not certified operating limits**, a multivariate domain detector, or calibrated uncertainty. No human-reviewed claim labels, verified maintenance documentation, calibrated RUL intervals, provider pricing, or provider-only latency measurements are available. The gate remains inconclusive without a live provider run; passing the synthetic cases alone does not establish diagnostic safety or accuracy.
