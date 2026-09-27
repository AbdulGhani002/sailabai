"""Scoring: metrics, the team's scoring protocol and the shared results table."""

from sailab.evaluation.metrics import (
    Confusion,
    ProbAccumulator,
    ReliabilityCurve,
    brier,
    expected_calibration_error,
    reliability_curve,
    skill_score,
)
from sailab.evaluation.protocol import EventScorer, compare_to_baseline, lead_bucket, scoring_mask

__all__ = [
    "Confusion",
    "EventScorer",
    "ProbAccumulator",
    "ReliabilityCurve",
    "brier",
    "compare_to_baseline",
    "expected_calibration_error",
    "lead_bucket",
    "reliability_curve",
    "scoring_mask",
    "skill_score",
]
