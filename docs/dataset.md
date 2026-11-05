# NASA C-MAPSS dataset

## Acquisition

Download **Turbofan Engine Degradation Simulation Data Set** from NASA's [Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), dataset 6. NASA describes four simulated subsets with different combinations of operating conditions and fault modes and provides the original dataset ZIP from that page. Extract the files locally; do not commit the data archive or extracted data to this repository. The dataset may have its own usage terms; consult NASA's repository before redistribution.

For the initial run, use `train_FD001.txt`. FD001–FD004 are supported by the loader. The official package also contains incomplete `test_FDxxx.txt` trajectories and `RUL_FDxxx.txt` targets; Phase 1 currently trains and validates on complete training trajectories. Evaluating the conventional held-out test files and one-terminal-prediction-per-engine benchmark protocol is a documented follow-up, not currently implemented.

## File schema

The whitespace-delimited files have no header. Each nonblank row must contain exactly 26 values, in this order:

| Positions | Name | Meaning |
|---|---|---|
| 1–2 | `unit`, `cycle` | Positive integer engine identifier and cycle counter |
| 3–5 | `setting_1`–`setting_3` | Operating settings |
| 6–26 | `sensor_1`–`sensor_21` | Sensor measurements |

The source uses positional columns; this project assigns names from the documented schema. Unsupported variants, missing/extra fields, nonnumeric values, infinite values, negative sensor measurements, invalid identifiers, duplicate unit-cycle pairs, and engines with fewer than two observations fail with a validation error. Blank lines are ignored. Explicit `NaN`, `NA`, or `null` in a feature field is retained as missing and imputed by the model's median imputer. Negative finite operating settings are accepted because they are not sensor channels. The imputer is part of the scikit-learn pipeline and is fit only on training engine rows. No arbitrary sensor range clipping or normalization is applied.

## Labels and split policy

For a complete run-to-failure training trajectory, raw RUL at cycle `t` is `last_cycle(unit) - t`; labels are capped at 125 cycles by default. The cap can be changed explicitly. A deterministic `GroupShuffleSplit` partitions engine IDs, so rows from one engine cannot occur in both training and test partitions. Metrics in the generated test report cover every cycle row in held-out complete trajectories. They are internal row-level test results and must not be compared directly with an official benchmark result that scores one endpoint prediction per test engine.

The bundled `packages/ml/tests/fixtures/synthetic_fd001.txt` is a tiny deterministic schema/test fixture. It is not NASA data. Training on it requires `--data-kind synthetic_fixture`; resulting reports are marked `synthetic_data: true` and `official_benchmark_metrics: false`. Never cite fixture metrics as benchmark results.
