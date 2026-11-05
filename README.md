# FaultWise AI

FaultWise AI is an educational predictive maintenance prototype for NASA C-MAPSS simulated turbofan engines. It estimates remaining useful life (RUL), shows unusual sensor behavior, and produces evidence-linked diagnostic reports when the available data supports them. Its quality pipeline measures rule-based grounding, abstention behavior, and regressions. It is **not certified for aviation or operational maintenance decisions**.

## What is implemented

| Layer | Current implementation |
|---|---|
| ML | `packages/ml`: C-MAPSS FD001–FD004 loader, validation, engine-level split, train-only imputation, scikit-learn baseline, RUL prediction, MAE/RMSE/PHM08 scoring, saved provenance. |
| API | `backend/api`: FastAPI dataset, engine, prediction, anomaly, report, history, health, readiness, and metrics endpoints. |
| Reports | Structured evidence IDs, separate observations/predictions/hypotheses, coverage checks, rule-based grounding, and explicit abstention. Gemini is optional. |
| Quality | Versioned synthetic golden cases, adversarial fixtures, regression comparison, and a strict `pass`/`fail`/`inconclusive` release gate. |
| Dashboard | Buildless HTML/CSS/JavaScript served by FastAPI at `/`; reads the real API. |
| Operations | GitHub Actions tests, Playwright smoke tests, deterministic CI quality check, Docker recipe, and in-process HTTP metrics. |

The browser is served by FastAPI; it does not need Next.js or a separate frontend build. The old `backend/app` prototype is preserved but is not the current API. No PostgreSQL, pgvector, RAG, SHAP, MLflow, or DeepEval integration is currently implemented.

## Data and measured status

Download NASA's [Turbofan Engine Degradation Simulation Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place `train_FD001.txt` in the ignored `data/` directory. [Dataset documentation](docs/dataset.md) describes all supported variants, the 26-column schema, validation, and preprocessing. The tiny committed fixture is **synthetic** and exists for tests only.

An FD001 model was trained locally from the NASA-supplied simulated training trajectories using seed 42 and a 20% split by engine. Its measured internal holdout MAE was **12.2848 cycles** and RMSE **16.8989 cycles** across 4,070 cycle rows from 20 held-out engines. These are **not official NASA test-set benchmark results**. The project does not yet score one endpoint per official test engine. The data and trained model are ignored by Git; see [real-data status](docs/real-data-status.md) for source hashes and protocol details.

The latest deterministic synthetic quality run passed **19/19** golden cases. Its full release gate is **inconclusive** because verified real-data ML evidence was not included in that run and live Gemini evaluation was unavailable. These cases measure engineered report behavior, not real diagnostic accuracy or semantic factuality.

## Local setup

Use Python 3.12 and Node.js 20 or newer. From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Train a local FD001 artifact after downloading the dataset:

```powershell
python -m faultwise_ml.training --data data\train_FD001.txt --variant FD001 --data-kind nasa_cmapss --output artifacts\ml --seed 42
```

Start the API and dashboard:

```powershell
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the dashboard or `/docs` for API documentation. `/health` checks the process; `/ready` returns 503 until a matching dataset and model are installed. Without a configured provider, report requests surface `provider_unavailable` or an abstention rather than inventing a report.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FAULTWISE_DATA_DIR` | `data` | Local C-MAPSS files. |
| `FAULTWISE_MODEL_PATH` | `artifacts/ml/faultwise_model.joblib` | Trained ML artifact. |
| `FAULTWISE_DEFAULT_VARIANT` | `FD001` | Readiness variant. |
| `FAULTWISE_ML_METRICS_PATH` | `artifacts/ml/test_metrics.json` | Saved ML evaluation provenance. |
| `FAULTWISE_REPORT_DIR` | `artifacts/reports` | Generated reports. |
| `FAULTWISE_EVALUATION_DIR` | `artifacts/evaluation` | Quality runs. |
| `FAULTWISE_LLM_PROVIDER` | `none` | Set to `gemini` to enable the optional provider. |
| `GEMINI_API_KEY` | unset | Gemini credential; supply through the environment, never Git. |
| `FAULTWISE_GEMINI_MODEL` | `gemini-2.5-flash` | Provider model name. |
| `FAULTWISE_PUBLIC_API_BASE` | same origin | Browser API base URL. |
| `FAULTWISE_ALLOWED_ORIGINS` | empty | Explicit comma-separated CORS origins. |

Gemini has **not** been exercised in a verified live report run for this project. Provider latency/cost and human-reviewed semantic accuracy remain unavailable.

## Tests and quality gate

```powershell
python -m pytest -q
cd frontend
npm ci
npm test
npx playwright install chromium
npm run test:e2e
```

The frontend has no compile/build step. From the repository root, run the synthetic golden evaluation and inspect its saved JSON:

```powershell
python -m backend.evaluation.cli run --baseline backend/evaluation/baselines/ci-v1.json
python -m backend.evaluation.ci_gate artifacts/evaluation/evaluation-<run-id>.json
python -m backend.evaluation.release_gate artifacts/evaluation/evaluation-<run-id>.json
```

The development CI check allows only the expected missing real ML and live-provider evidence. The strict release command exits nonzero while either is missing. See [evaluation methodology](docs/evaluation.md) and [operations](docs/operations.md).

## Container

With a running Docker engine, `docker compose build` and `docker compose up -d` start the combined API/dashboard at `http://127.0.0.1:8000/`. The image excludes local NASA data, model binaries, reports, and keys; Compose mounts local `data/` and `artifacts/`. See [operations](docs/operations.md) for readiness, metrics, release checks, and deployment limits. No cloud deployment or registry publication has been verified.

## Documentation

- [Training and reproducibility](docs/training.md)
- [API and anomaly behavior](docs/api.md)
- [Diagnostic report contract](docs/diagnostic-reports.md)
- [Dashboard and browser tests](docs/frontend.md)
- [Evaluation and regression rules](docs/evaluation.md)
- [CI, observability, and deployment](docs/operations.md)

Current limitations include an internal row-level holdout rather than an official NASA test-set benchmark, descriptive rather than calibrated anomaly scores, uncalibrated RUL uncertainty, rule-based rather than semantic grounding, local JSON history, no authentication, and no live Gemini verification.
