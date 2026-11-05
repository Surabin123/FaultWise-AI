# RUL metrics

The regression report contains mean absolute error (MAE), root mean squared error (RMSE), and a summed PHM08 asymmetric score. MAE and RMSE are reported in cycle units. The score has no physical units; lower is better.

## PHM08 asymmetric prognostics score

The implementation uses `d = predicted_RUL - true_RUL` and sums:

```text
exp(-d / 13) - 1, when d < 0 (early / under-prediction)
exp( d / 10) - 1, when d >= 0 (late / over-prediction)
```

This is the asymmetric score used by the PHM 2008 Prognostics Data Challenge. The formula and sign convention are printed in the PHM Society proceedings paper [RUL Estimation for Turbofan Engine Degradation Simulation Data](https://www.papers.phmsociety.org/index.php/phmconf/article/download/2348/1323), which states it is identical to the PHM 2008 challenge score. NASA's repository identifies the related PHM08 challenge data set and its C-MAPSS provenance on the [NASA PCoE repository page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

Known-value checks in `packages/ml/tests/test_metrics.py` verify zero error gives zero score, an error of −13 and +10 each gives `e - 1`, and MAE/RMSE match hand-calculated values. The implemented function is the PHM08 sum. Training's engine-level held-out test partition currently sums over every cycle row of held-out complete trajectories; it is deliberately labeled row-level test evaluation. An official-style C-MAPSS benchmark result must score one prediction per test engine using `RUL_FDxxx.txt`; this project does not yet implement or report that result.

## Provenance and reproducibility

Each training run writes `test_metrics.json` with source path, SHA-256 of the input file, variant, UTC timestamp, train/test engine IDs, configuration, protocol, and measured metrics. `training_config.json` and the joblib artifact are saved alongside it. If run against the synthetic test fixture, the report explicitly marks it synthetic and says it is not an official benchmark metric.
