"""Evaluation — ref-based metrics, CLIP scoring, and orchestration."""

from core.evaluation.runner import compute_metrics, load_predictions

__all__ = [
    "compute_metrics",
    "load_predictions",
]
