#!/usr/bin/env python3
"""Compute reference-based and reference-free metrics for VLM predictions.

Reads predictions from .jsonl files, computes metrics, saves to metrics.json.

Usage:
    python scripts/run_eval.py --model blip2 --prompt A
    python scripts/run_eval.py --model blip2 --prompt A --ref_free_only   # skip ref-based
    python scripts/run_eval.py --all                                       # all models, all prompts
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
    prompt_key: str,
    device: str = "cuda:0",
    skip_ref_based: bool = False,
):
    """Compute and save metrics for one (model, prompt) pair."""
    pred_file = os.path.join(
        OUTPUT_DIR, model_name, f"predictions_prompt_{prompt_key.lower()}.jsonl"
    )
    metrics_file = os.path.join(
        OUTPUT_DIR, model_name, f"metrics_prompt_{prompt_key.lower()}.json"
    )

    predictions = load_predictions(pred_file)
    if not predictions:
        print(f"[ERROR] No predictions for {model_name}/{prompt_key}")
        return None

    print(f"[Eval] {model_name}/{prompt_key}: {len(predictions)} captions")

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
        for k, v in ref_metrics.items():
            print(f"    {k}: {v:.2f}")

    # Reference-free
    print("  Computing ref-free metrics...")
    scorer = CLIPScorer(model_name=CLIP_MODEL_NAME, device=device)
    image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}

    clip_metrics = scorer.compute_refclipscore(image_paths, predictions, references)
    metrics.update(clip_metrics)
    for k, v in clip_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {metrics_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluation")
    parser.add_argument("--model", type=str, help="Single model name.")
    parser.add_argument("--prompt", type=str, default="A", help="Prompt key.")
    parser.add_argument("--all", action="store_true", help="Evaluate all model/prompt combos.")
    parser.add_argument("--ref_free_only", action="store_true", help="Skip ref-based metrics.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.all:
        from config import MODEL_REGISTRY
        combos = []
        for name, _, _ in MODEL_REGISTRY:
            combos.append((name, "A", False))  # Primary: Prompt A, ref-based+ref-free
        # Sensitivity: B/C for LLaVA + Qwen2.5-VL (ref-free only — see
        # docs/research-notes/2026-06-04-prompt-gt-alignment-analysis.md §5a)
        for name in ["llava", "qwen2vl"]:
            for pk in ["B", "C"]:
                combos.append((name, pk, True))
    else:
        combos = [(args.model, args.prompt, args.ref_free_only)]

    all_metrics = {}
    for model_name, prompt_key, skip_ref_based in combos:
        m = compute_all_metrics(
            model_name, prompt_key, args.device, skip_ref_based
        )
        if m:
            all_metrics[f"{model_name}/{prompt_key}"] = m

    # Save global summary
    summary_file = os.path.join(OUTPUT_DIR, "all_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Summary] → {summary_file}")
