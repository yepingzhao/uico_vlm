#!/usr/bin/env python3
"""Compute reference-based and reference-free metrics for few-shot VLM predictions.

Follows the same pattern as scripts/run_eval.py for zero-shot.

Usage:
    python scripts/eval_fewshot.py --model llava --k 1
    python scripts/eval_fewshot.py --model qwen2vl --k 1 3 5
    python scripts/eval_fewshot.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, CLIP_MODEL_NAME
from data.dataset import load_test_dataset
from eval.ref_based import compute_ref_based_metrics
from eval.ref_free import CLIPScorer

FEWSHOT_K_VALUES = [1, 3, 5]


def _get_fewshot_models():
    """Discover which registered models support few-shot inference.

    Returns:
        List of model short names that implement generate_fewshot.
    """
    from models import get_wrapper

    candidates = ["llava", "llava-next", "qwen2vl", "qwen3vl"]
    available = []
    for name in candidates:
        try:
            wrapper = get_wrapper(name)
            if wrapper.supports_fewshot:
                available.append(name)
        except (ValueError, ImportError):
            pass
    return available


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


def compute_all_metrics(
    model_name: str,
    k: int,
    device: str = "cuda:0",
    skip_ref_based: bool = False,
):
    """Compute and save metrics for one (model, k) pair."""
    prompt_key = f"fewshot_k{k}"
    pred_file = os.path.join(
        OUTPUT_DIR, model_name, f"predictions_{prompt_key}.jsonl"
    )
    metrics_file = os.path.join(
        OUTPUT_DIR, model_name, f"metrics_{prompt_key}.json"
    )

    predictions = load_predictions(pred_file)
    if not predictions:
        print(f"[ERROR] No predictions for {model_name}/k={k}")
        return None

    print(f"[Eval] {model_name}/k={k}: {len(predictions)} captions")

    # Load test dataset for references
    ds = load_test_dataset(subsample=0)

    # Filter references to images that have predictions
    pred_img_ids = set(predictions.keys())
    ds.image_ids = sorted(set(ds.image_ids) & pred_img_ids)
    references = ds.all_references_dict()

    metrics = {}

    # Reference-based
    if not skip_ref_based:
        print("  Computing ref-based metrics...")
        ref_metrics = compute_ref_based_metrics(predictions, references)
        metrics.update(ref_metrics)
        for kk, v in ref_metrics.items():
            print(f"    {kk}: {v:.2f}")

        # Save immediately after ref-based (before potentially-fragile ref-free)
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved (ref-based) → {metrics_file}")

    # Reference-free
    print("  Computing ref-free metrics...")
    try:
        scorer = CLIPScorer(model_name=CLIP_MODEL_NAME, device=device)
        image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}
        clip_metrics = scorer.compute_refclipscore(image_paths, predictions, references)
        metrics.update(clip_metrics)
        for kk, v in clip_metrics.items():
            print(f"    {kk}: {v:.4f}")
        # Re-save with ref-free metrics included
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved (with ref-free) → {metrics_file}")
    except Exception as e:
        print(f"    [WARN] CLIP failed (ref-free metrics unavailable): {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot VLM Evaluation")
    parser.add_argument("--model", type=str, help="Single model name.")
    parser.add_argument("--k", nargs="+", type=int, default=[1],
                        help="Few-shot k values (e.g. 1 3 5).")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all few-shot model/k combos.")
    parser.add_argument("--ref_free_only", action="store_true",
                        help="Skip ref-based metrics.")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.all:
        fewshot_models = _get_fewshot_models()
        print(f"[Discover] Few-shot models: {fewshot_models}")
        combos = [(m, k) for m in fewshot_models for k in FEWSHOT_K_VALUES]
    else:
        combos = [(args.model, k) for k in args.k]

    all_metrics = {}
    for model_name, k in combos:
        m = compute_all_metrics(
            model_name, k, args.device, args.ref_free_only
        )
        if m:
            all_metrics[f"{model_name}/k={k}"] = m

    # Save global summary (same pattern as all_metrics.json)
    summary_file = os.path.join(OUTPUT_DIR, "fewshot_all_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Summary] → {summary_file}")
