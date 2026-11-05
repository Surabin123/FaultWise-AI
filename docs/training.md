# Train the Phase 1 baseline

Use Python 3.10 or newer. In the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
python -m faultwise_ml.training --data data\train_FD001.txt --variant FD001 --data-kind nasa_cmapss --output artifacts\ml --seed 42
```

Place the locally extracted NASA `train_FD001.txt` at the indicated path (or substitute its actual location). Repeat for another supported variant by changing both filename and variant. A run prints its measured report and writes:

- `faultwise_model.joblib` — fitted imputer/model pipeline, feature schema, and training report.
- `training_config.json` — explicit estimator/split/label configuration.
- `test_metrics.json` — source hash, split engine IDs, timestamp, protocol, and actual held-out metrics.

The train/test split is deterministic for a fixed file, seed, and configuration. Dataset bytes are fingerprinted with SHA-256. Training does not download data or invent missing benchmark results. Do not submit the synthetic fixture as a benchmark run.

## Test without the NASA download

The tests use the tiny local schema fixture only:

```powershell
python -m pytest
```

The training test invokes `--data-kind synthetic_fixture` internally, and asserts the report does not mark those results as official benchmark metrics. It uses temporary directories, so no synthetic model artifact is left in the project `artifacts/` directory.
