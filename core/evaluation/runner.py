"""Evaluation orchestration — load predictions, run metrics, save results."""

import json
import os
import sys

from config import CLIP_MODEL_NAME
from core.evaluation.metrics import CLIPScorer, compute_ref_based_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(filepath: str) -> dict:
    """Load predictions from JSONL file → {image_id: caption}."""
    preds = {}
    if not os.path.exists(filepath):
        print(f"[WARN] Predictions file not found: {filepath}")
        return preds
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            preds[record["image_id"]] = record["caption"]
    return preds


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: dict,
    references: dict,
    image_paths: dict,
    metrics_file: str,
    device: str = "cuda:0",
    skip_ref_based: bool = False,
) -> dict:
    """Compute ref-based and ref-free metrics, save to disk.

    Args:
        predictions: {image_id: caption} dict.
        references: {image_id: [ref_captions]} dict.
        image_paths: {image_id: filesystem_path} dict for CLIP scoring.
        metrics_file: Path to save the metrics JSON.
        device: CUDA device for CLIP scorer.
        skip_ref_based: If True, only compute ref-free metrics.

    Returns:
        Metrics dict (e.g. {"BLEU-1": 45.2, "CLIPScore": 0.72, ...}).
    """
    metrics = {}

    # Reference-based
    if not skip_ref_based:
        print("  Computing ref-based metrics...")
        ref_metrics = compute_ref_based_metrics(predictions, references)
        metrics.update(ref_metrics)
        for k, v in ref_metrics.items():
            print(f"    {k}: {v:.2f}")

        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved (ref-based) → {metrics_file}")

    # Reference-free
    print("  Computing ref-free metrics...")
    try:
        scorer = CLIPScorer(model_name=CLIP_MODEL_NAME, device=device)
        clip_metrics = scorer.compute_refclipscore(
            image_paths, predictions, references
        )
        metrics.update(clip_metrics)
        for k, v in clip_metrics.items():
            print(f"    {k}: {v:.4f}")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved (with ref-free) → {metrics_file}")
    except Exception as e:
        print(f"    [WARN] CLIP failed (ref-free metrics unavailable): {e}")

    return metrics
