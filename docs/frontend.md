# Phase 5 dashboard

The dashboard is served by the current FastAPI app at `/`. It replaces the preserved prototype's fictional fleet cards and old `/api/*` calls with live `/api/v1/*` requests. Its static files are in `frontend/`; it uses buildless JavaScript modules with JSDoc response types, a central API client, and the existing restrained charcoal and teal visual identity. No frontend credentials are stored.

## Run locally

From the repository root:

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The default API base is the same origin. `FAULTWISE_PUBLIC_API_BASE` changes the URL used by the browser; it is emitted by `/app-config.js` at page load. For a different API origin, set `FAULTWISE_ALLOWED_ORIGINS` on the API to an explicit comma-separated list of frontend origins. `FAULTWISE_DATA_DIR`, `FAULTWISE_MODEL_PATH`, `FAULTWISE_ML_METRICS_PATH`, `FAULTWISE_REPORT_DIR`, and `FAULTWISE_EVALUATION_DIR` locate the local data and saved artifacts. Set `FAULTWISE_LLM_PROVIDER=gemini` and `GEMINI_API_KEY` only if an actual Gemini report request is intended. Keys are never sent to the browser.

## Navigation and data sources

| View | Actual source | Behavior when unavailable |
|---|---|---|
| Overview | `/health`, `/ready`, `/api/v1/datasets`, `/api/v1/model`, `/api/v1/model/evaluation`, `/api/v1/provider`, saved report/evaluation lists | Shows unavailable or not ready, never a fabricated zero. |
| Engine explorer | Dataset metadata, engine list, paginated trajectory | Shows invalid or unavailable dataset/engine error. Sensor names and settings come from metadata. |
| Prediction & anomaly | POST prediction and anomaly-analysis endpoints for selected engine | Shows model mismatch or unavailable errors. RUL is labeled an estimate; anomaly scores are not fault probabilities. |
| Diagnostic reports | POST diagnostic-report; GET saved report list/detail | Shows `provider_unavailable` or abstention explicitly. Evidence ID buttons jump to actual report evidence. |
| AI quality | Latest persisted Phase 4 evaluation | Shows actual gate, synthetic case metrics, safety cases, missing live provider, and limitations. |
| Evaluation history | Persisted evaluation list/detail and server-side compatible-run comparison | Shows empty state or explicit comparison error. |

The new read-only endpoints are `/api/v1/model/evaluation`, `/api/v1/provider`, `/api/v1/evaluations`, `/api/v1/evaluations/{run_id}`, `/api/v1/evaluations/compare?previous=…&current=…`, `/api/v1/reports`, and `/api/v1/reports/{report_id}`. They read existing JSON artifacts; they do not run evaluations or generate report history. Corrupt or absent artifacts are not presented as valid results. The model evaluation endpoint verifies the source SHA-256 and engine split from the saved Phase 1 report before marking integrity verified. The evaluation view identifies synthetic fixture scope and never calls those results NASA diagnostic accuracy.

## Real-data demo

With the locally installed NASA FD001 data and trained artifact, open Engine explorer, select FD001 and engine 1, and inspect its 192 observed cycles. Open Prediction & anomaly and run RUL prediction to see the current model estimate and source cycle; this value is a live inference, not a fixed screenshot value. Run anomaly analysis to inspect descriptive sensor deviation and limitations. Open AI quality to read the saved 19-case synthetic evaluation and its `inconclusive` gate. A sufficient diagnostic report request currently returns `provider_unavailable` because no Gemini key is configured. The report list remains empty until an actual report is saved. The saved FD001 MAE/RMSE are row-level holdout results, not an official NASA endpoint benchmark.

## Tests

```powershell
py -3.12 -m pytest -q
cd frontend
npm ci
npm test
npx playwright install chromium
npm run test:e2e
```

Node unit tests verify API paths, errors, pagination, and unavailable formatting. Playwright uses clearly synthetic route fixtures to cover the main workflow, mobile missing dependencies, evidence-reference navigation, run comparison, and provider failure without Gemini credentials. It starts and stops its own local FastAPI server. No browser test fixture is shown in production. The local visual check also loaded real FD001 engine 1 trajectory and a saved-model prediction through the dashboard.

## Limits

There is no authentication or certified maintenance workflow. The dashboard intentionally omits fleet-wide RUL summaries because no aggregate prediction endpoint exists. Report history is local JSON only; no reports were available during the Phase 5 real-data check. Provider latency/cost, calibrated prediction uncertainty, semantic claim review, and live Gemini evaluation are unavailable. The release gate remains `inconclusive` until required live checks are performed. The UI does not turn anomaly flags or generated hypotheses into confirmed faults.
