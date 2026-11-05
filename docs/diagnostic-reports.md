# Phase 3 diagnostic reports

`POST /api/v1/datasets/{variant}/engines/{engine_id}/diagnostic-report` collects the actual Phase 1 RUL estimate, latest observed sensors and operating settings, dataset hash, and Phase 2 anomaly analysis. The report service assigns stable evidence IDs such as `model:rul`, `sensor:sensor_1:cycle:192`, `setting:setting_1:cycle:192`, and `anomaly:sensor_1`. Each item records its source, value, cycle, interpretation, and limitations. The current model has no calibrated confidence interval or SHAP attribution; neither is invented. No verified maintenance document collection or PostgreSQL connection exists, so there are no document citations or database writes.

The sufficiency layer runs before the provider. It abstains when the model/dataset are unavailable, anomaly analysis lacks history, fewer than 18 latest-cycle sensors are observed, the data file hash differs from the model training source, operating settings fall outside verified training-engine ranges, training coverage cannot be verified, or no anomaly corroborates a diagnostic concern. Model or dataset absence uses the existing structured 503 errors. Other insufficiency produces a report with `assessment: insufficient_evidence`, missing evidence, and no LLM call. The current deterministic rule permits `investigate` when corroborating anomaly evidence exists and the provider output passes validation. `actionable` is reserved for a future validated decision rule.

Training now stores per-setting minima and maxima computed from **training engines only**. Older artifacts use their recorded training engine IDs and original dataset source after SHA-256 verification to derive the same ranges. An absent or changed source makes coverage unavailable and causes abstention. Range checks are empirical single-feature screens; they are not manufacturer limits, joint operating-envelope validation, or a confidence interval. A changed input dataset still triggers the separate provenance abstention.

The configurable provider is Gemini's `generateContent` REST endpoint, using the [official Google REST format](https://ai.google.dev/gemini-api/docs/generate-content/text-generation). Set `FAULTWISE_LLM_PROVIDER=gemini`, `GEMINI_API_KEY`, and optionally `FAULTWISE_GEMINI_MODEL` (default `gemini-2.5-flash`). Without a key or provider selection, a sufficient request returns `provider_unavailable` (503). Timeouts return 504; malformed output and failed grounding return 502. The API key is sent in the provider request header and is not logged or stored. Do not put keys in a source file.

The LLM returns only hypotheses, suggested investigations, uncertainty, and limitations in JSON. Every claim has a typed scope (`sensor_behavior` or `component_condition`) and evidence IDs. Observations, predictions, executive summary, and final assessment are assembled by deterministic code. The prompt explicitly identifies evidence JSON as untrusted data. Pydantic rejects wrong types, missing fields, extra fields, and invalid claim kinds or scopes. Validation rejects unknown evidence IDs, concerns without anomaly evidence, unsupported numerical prose, component-specific claims lacking component-level evidence, credential references, and instructions to ignore or fabricate evidence. The current evidence schema has no component-level diagnostic source, so it cannot support a component-failure claim; prose is checked even if the provider labels its scope as sensor behavior. Unsafe provider output fails with 502; raw provider text and credentials are not returned or logged. This remains a limited structural and lexical guard, **not perfect hallucination detection**. A provider can still produce plausible unsupported semantic claims; expert review and live evaluation remain necessary.

Reports record ID, model ID, input dataset SHA-256, prompt version, provider/model, generated timestamp, latency, validation status, token usage when returned, evidence, and contents. By default, valid or abstained reports are saved as JSON under ignored `artifacts/reports/` with an atomic replacement. Supply `{"persist":false}` to skip local storage. No secret or hidden prompt response is stored. Reports cannot be reproduced exactly if the provider changes despite the saved metadata.

Example request:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/datasets/FD001/engines/1/diagnostic-report -H "Content-Type: application/json" -d '{"persist":false}'
```

With the locally trained FD001 model and no configured provider, that request returns HTTP 503:

```json
{"error":{"code":"provider_unavailable","message":"A diagnostic report provider is not configured."}}
```

The test suite uses an explicitly synthetic C-MAPSS-shaped fixture and a fixed fake provider to verify successful reports and adversarial failures. That fixture output is not a NASA diagnostic result. On the locally downloaded NASA FD001 data and saved model, the endpoint collected sufficient evidence but returned `provider_unavailable` because no Gemini key is configured; **no real-data LLM report has been generated**.
