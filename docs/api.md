# Phase 2 API

The API is a local FastAPI service that calls the Phase 1 loader and persisted scikit-learn pipeline. It does not create or substitute sample production data.

## Setup and configuration

Install runtime dependencies and the local ML package, then run the service:

```powershell
py -3.12 -m pip install -r requirements-api.txt -e packages/ml
$env:FAULTWISE_DATA_DIR = "D:\AI-P\data"
$env:FAULTWISE_MODEL_PATH = "D:\AI-P\artifacts\ml\faultwise_model.joblib"
$env:FAULTWISE_DEFAULT_VARIANT = "FD001"
py -3.12 -m uvicorn backend.api.main:app --reload
```

Environment variables:

| Variable | Default | Use |
|---|---|---|
| `FAULTWISE_DATA_DIR` | `data` | Directory containing local `train_FD001.txt` through `train_FD004.txt` files |
| `FAULTWISE_MODEL_PATH` | `artifacts/ml/faultwise_model.joblib` | Phase 1 joblib artifact path |
| `FAULTWISE_DEFAULT_VARIANT` | `FD001` | Dataset/model variant required for `/ready` |

Train an artifact with the Phase 1 command documented in [training.md](training.md). Set `--data-kind nasa_cmapss` for a real NASA source file. `synthetic_fixture` is reserved for isolated tests and the API reports its provenance.

Interactive OpenAPI docs are available at `/docs`; the raw schema is at `/openapi.json`.

## Endpoint summary

| Method | Path | Result |
|---|---|---|
| `GET` | `/health` | Process is responding; independent of data/model availability |
| `GET` | `/ready` | 200 only when default-variant dataset and matching model are available; otherwise 503 |
| `GET` | `/api/v1/datasets` | Availability state for FD001–FD004 |
| `GET` | `/api/v1/datasets/{variant}` | Actual dataset metadata or a structured unavailable/invalid error |
| `GET` | `/api/v1/datasets/{variant}/engines` | Engine IDs read from the selected local data file |
| `GET` | `/api/v1/datasets/{variant}/engines/{engine_id}/trajectory?offset=0&limit=200` | Paginated observed settings and sensor records |
| `GET` | `/api/v1/model` | Loaded, unavailable, or invalid artifact state |
| `POST` | `/api/v1/datasets/{variant}/engines/{engine_id}/prediction` | RUL prediction from the loaded Phase 1 pipeline and latest real row |
| `POST` | `/api/v1/datasets/{variant}/engines/{engine_id}/anomaly-analysis` | Deterministic within-engine sensor deviation analysis |

Variants are `FD001`–`FD004`; engine IDs must be positive integers present in the selected dataset. Missing datasets return `dataset_unavailable`; malformed files are `dataset_invalid`; unknown engines return 404; missing/bad models return explicit 503 errors; model and dataset variant mismatch returns 409. All errors use `{"error":{"code":"...","message":"..."}}` and do not expose machine paths or stack traces.

Example requests:

```powershell
curl http://127.0.0.1:8000/api/v1/datasets
curl "http://127.0.0.1:8000/api/v1/datasets/FD001/engines/1/trajectory?limit=5"
curl -X POST http://127.0.0.1:8000/api/v1/datasets/FD001/engines/1/prediction
curl -X POST http://127.0.0.1:8000/api/v1/datasets/FD001/engines/1/anomaly-analysis
```

The prediction response separates `predicted_rul` from observed latest-cycle sensor evidence and includes model ID, model data provenance, and limitations. The anomaly response includes baseline/recent cycle counts, per-sensor robust deviation scores, skipped sensors, and a note that scores are not probabilities or fault diagnoses.

## Status meanings

- **API running:** `/health` returns 200.
- **Dataset available:** dataset inventory reports `available`; `unavailable` means no local file and `invalid` means a file failed Phase 1 validation.
- **Model trained:** a Phase 1 artifact exists and has the expected schema; `/api/v1/model` reports `loaded` only after it is successfully opened.
- **Ready:** `/ready` returns 200 only if the configured default dataset exists and the loaded artifact matches its variant.
- **Prediction generated:** prediction endpoint returns 200 with `status: predicted`; this is not asserted unless a real trained artifact is loaded. Fixture-backed integration tests mark `model_dataset_kind: synthetic_fixture`.

## Anomaly baseline and limitations

Phase 1 did not include an anomaly detector. Phase 2 uses a small descriptive baseline: for each sensor, it compares the median of the latest five cycles with the median and MAD of earlier cycles for that same engine. The signed robust z-score is `(recent median - baseline median) / (1.4826 * MAD)`; absolute score 3.5 or greater is flagged. Less than ten cycles returns `insufficient_evidence`. Zero-MAD changes are flagged without a finite score. This method does not condition on operating settings, is not calibrated as a probability, and does not establish a mechanical fault. The anomaly endpoint performs no maintenance action. See [diagnostic-reports.md](diagnostic-reports.md) for the separate Phase 3 report endpoint.

Phase 5 adds read-only endpoints for saved model metrics, provider availability, diagnostic report history, and evaluation history/comparison. See [frontend.md](frontend.md) for the exact routes and dashboard behavior. The API now serves the buildless dashboard at `/` and static files at `/static/*`.

## Tests

Install `requirements-dev.txt` and run `py -3.12 -m pytest`. API integration tests use the small synthetic Phase 1 fixture and train a temporary fixture artifact; they do not represent NASA dataset availability or real-data model performance. See [dataset.md](dataset.md) for source acquisition and data schema.
