# NASA C-MAPSS local data status

Updated 5 November 2026. The original C-MAPSS archive was downloaded from [NASA's Prognostics Center of Excellence repository, dataset 6](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) using its direct ZIP URL:

```text
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

The outer archive contains a nested `CMAPSSData.zip`. Its 12 `train_FDxxx.txt`, `test_FDxxx.txt`, and `RUL_FDxxx.txt` files for FD001–FD004 were extracted to the ignored `data/` directory. Outer archive SHA-256: `c9c5dec12a945a82e8bb4446589d7fb3cc057b5d81fa1a12e25ee9912ad3b2`.

The Phase 1 FD001 model was trained from `data/train_FD001.txt` with seed 42, a 20% engine-level holdout, and a 125-cycle label cap. Input file SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`. The model and full provenance report are in ignored `artifacts/ml/`.

| Measurement | Result | Scope |
|---|---:|---|
| Held-out engines | 20 | Engine-level split of the complete FD001 training trajectories |
| Held-out cycle rows | 4,070 | Every cycle row of those engines |
| MAE | 12.2848 cycles | Row-level holdout |
| RMSE | 16.8989 cycles | Row-level holdout |
| PHM08 score sum | 31,263.6343 | Summed over 4,070 rows |

These are measured row-level holdout results on NASA-supplied simulated engine data. They are **not official C-MAPSS benchmark results**: the official protocol scores one endpoint prediction per engine against `RUL_FD001.txt`, which the current training command does not perform.

The Phase 2 API was checked with the local dataset and artifact. `/ready` returned HTTP 200 with FD001 dataset available (100 engines) and model loaded. A request to `/api/v1/datasets/FD001/engines/1/prediction` returned HTTP 200 and a predicted RUL of 5.8873 cycles using the sensor observation at cycle 192. This verifies real-dataset inference through the saved pipeline; it is one example prediction, not a performance measure.
