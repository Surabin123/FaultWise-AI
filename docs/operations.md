# Phase 6: CI, release evidence, and deployment

## CI checks

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual dispatch. It installs Python 3.12, runs the Python tests (including ML loader, split, training, metrics, API, and evaluation tests), runs the synthetic golden suite against the committed compatible baseline, and checks the existing Phase 4 gate. A separate job runs Node unit tests and Chromium Playwright tests. After both jobs succeed, CI builds and saves a Docker image as a short-lived workflow artifact. The workflow does not deploy or publish an image.

The frontend is plain HTML/CSS/JavaScript, so there is no frontend production build, TypeScript check, or linter. `npm ci` checks the committed dependency lock. The Playwright web server starts the real FastAPI application.

The committed `backend/evaluation/baselines/ci-v1.json` records only the deterministic synthetic golden v1 baseline, with 19 passing cases, no local file paths, and no NASA or Gemini result. Each CI run writes its full evaluation JSON to a workflow artifact. New failed cases, metric regressions, critical gate failures, missing golden cases, or an incompatible baseline fail CI. Expected missing verified NASA metrics and live Gemini evidence keep the **full release gate inconclusive** while allowing development CI to pass. An unavailable metric is never interpreted as zero.

Run the checks locally from the repository root:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
cd frontend && npm ci && npm test && npx playwright install chromium && npm run test:e2e
```

Then, from the repository root:

```sh
python -m backend.evaluation.cli run --baseline backend/evaluation/baselines/ci-v1.json
python -m backend.evaluation.ci_gate artifacts/evaluation/evaluation-*.json
```

The `*.json` shell expansion above works on Bash. In PowerShell, supply the latest evaluation file path explicitly.

## Strict release decision

The manual `release-gate.yml` workflow runs the same golden suite with the existing strict release configuration and opts in to a live provider check. It fails unless the saved gate status is exactly `pass`. The default hosted runner has no NASA data/model or Gemini secret, so a normal run will be **inconclusive and blocked**. This is intentional. A release operator must supply a verified NASA metrics report, matching data and model files, and `GEMINI_API_KEY` in an authorized release environment before using this gate as release evidence. The workflow does not provision those files or deploy an application. Do not commit the data, model, reports, or API key to Git.

For an equipped local environment, run:

```sh
FAULTWISE_LLM_PROVIDER=gemini GEMINI_API_KEY=... python -m backend.evaluation.cli run --baseline backend/evaluation/baselines/ci-v1.json --ml-report artifacts/ml/test_metrics.json --live
python -m backend.evaluation.release_gate artifacts/evaluation/evaluation-<run-id>.json
```

The live check evaluates one report and its rule-based claim findings; it does not establish semantic factual accuracy or aviation suitability. A verified NASA report is based on the project holdout protocol, not the official C-MAPSS test benchmark.

## Container and service operation

`Dockerfile` packages the Python backend, ML package, and buildless frontend into one image. It runs as an unprivileged user. The image excludes NASA data, trained models, reports, local keys, tests, and caches. `compose.yaml` mounts local `data/` read only and `artifacts/` for models and generated reports. The compose port is bound to localhost; add a properly configured reverse proxy and access controls before external exposure.

```sh
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
```

`/health` checks process availability. `/ready` returns 503 until the configured dataset and compatible trained model are present. A 503 on an empty installation is expected. `/metrics` exposes in-process Prometheus text counters and total request duration, labeled by HTTP method, route template, and status. Request query strings, IDs, and provider secrets are omitted. The application also writes per-request method, route, status, and duration to logs. These counters reset on restart and do not aggregate across replicas. Protect `/metrics` at the network layer when externally deployed.

For a basic rollback, redeploy a previously tested image and its matching model/data artifact set, then check `/ready`, `/health`, and the evaluation gate. The CI image artifact is retained for seven days; durable release storage and deployment automation are not configured.

## Current limits

No cloud deployment, registry publishing, secret store, external monitoring backend, or service level objective is configured. The manual GitHub release workflow is fail-closed until its evidence is provisioned. Container behavior must be checked on a running Docker engine.
