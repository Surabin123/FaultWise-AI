"""Descriptive within-trajectory robust sensor anomaly baseline."""

import numpy as np
import pandas as pd

from faultwise_ml.schema import SENSOR_COLUMNS

from ..schemas import AnomalyResponse, DatasetVariant, SensorAnomaly

METHOD = "within-engine baseline median/MAD robust deviation"
METHOD_VERSION = "1.0"
MINIMUM_CYCLES = 10
RECENT_CYCLES = 5
ROBUST_Z_THRESHOLD = 3.5


def analyze_trajectory(frame: pd.DataFrame, *, engine_id: int, variant: DatasetVariant) -> AnomalyResponse:
    """Compare each sensor's latest five-cycle median with its earlier history."""
    count = len(frame)
    limitations = [
        "A statistical deviation indicates unusual sensor behavior, not a confirmed mechanical fault.",
        "The baseline does not adjust for operating settings or known sensor-channel regimes.",
        "A robust z-score is a standardized deviation, not a probability of failure.",
    ]
    if count < MINIMUM_CYCLES:
        return AnomalyResponse(
            engine_id=engine_id,
            dataset_variant=variant,
            status="insufficient_evidence",
            cycles_analyzed=count,
            baseline_cycles=0,
            recent_cycles=0,
            sensors_analyzed=0,
            method=METHOD,
            method_version=METHOD_VERSION,
            score_definition=f"absolute robust z-score >= {ROBUST_Z_THRESHOLD}; not a probability",
            unusual_sensor_behavior=[],
            skipped_sensors=list(SENSOR_COLUMNS),
            limitations=limitations + [f"At least {MINIMUM_CYCLES} cycles are required by this baseline."],
        )

    recent_count = min(RECENT_CYCLES, count // 2)
    baseline = frame.iloc[:-recent_count][list(SENSOR_COLUMNS)].to_numpy(dtype=float)
    recent = frame.iloc[-recent_count:][list(SENSOR_COLUMNS)].to_numpy(dtype=float)
    anomalies: list[SensorAnomaly] = []
    skipped: list[str] = []
    analyzed = 0
    for index, sensor in enumerate(SENSOR_COLUMNS):
        baseline_values = baseline[:, index]
        recent_values = recent[:, index]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        recent_values = recent_values[np.isfinite(recent_values)]
        if baseline_values.size < 5 or recent_values.size == 0:
            skipped.append(sensor)
            continue
        baseline_median = float(np.median(baseline_values))
        recent_median = float(np.median(recent_values))
        mad = float(np.median(np.abs(baseline_values - baseline_median)))
        robust_scale = 1.4826 * mad
        difference = recent_median - baseline_median
        if robust_scale > 0:
            score: float | None = difference / robust_scale
            unusual = abs(score) >= ROBUST_Z_THRESHOLD
            explanation = (
                f"Recent median differs from the earlier baseline by {score:.2f} robust standard deviations."
                if unusual else "Recent median remains within the configured robust-deviation threshold."
            )
        elif difference == 0:
            score = 0.0
            unusual = False
            explanation = "Recent median matches a zero-MAD historical baseline."
        else:
            score = None
            unusual = True
            explanation = "Recent median differs from a zero-MAD historical baseline; a finite robust score is undefined."
        analyzed += 1
        if unusual:
            anomalies.append(SensorAnomaly(
                sensor=sensor,
                baseline_median=baseline_median,
                recent_median=recent_median,
                robust_z_score=score,
                unusual=True,
                explanation=explanation,
            ))

    return AnomalyResponse(
        engine_id=engine_id,
        dataset_variant=variant,
        status="analyzed" if analyzed else "insufficient_evidence",
        cycles_analyzed=count,
        baseline_cycles=len(baseline),
        recent_cycles=recent_count,
        sensors_analyzed=analyzed,
        method=METHOD,
        method_version=METHOD_VERSION,
        score_definition=f"signed (recent median - baseline median) / (1.4826 * MAD); threshold |score| >= {ROBUST_Z_THRESHOLD}; not a probability",
        unusual_sensor_behavior=anomalies,
        skipped_sensors=skipped,
        limitations=limitations,
    )
