#!/usr/bin/env python3
"""Compute reference-based and reference-free metrics for VLM predictions.

Reads predictions from .jsonl files, computes metrics, saves to metrics.json.
Handles both zero-shot (--prompt) and few-shot (--k) evaluation.

Usage:
    # Zero-shot
    python scripts/run_eval.py --model blip2 --prompt A
    python scripts/run_eval.py --model blip2 --prompt A --ref_free_only
    python scripts/run_eval.py --all

    # Few-shot
    python scripts/run_eval.py --model llava --k 1
    python scripts/run_eval.py --model llava --k 1 3 5
    python scripts/run_eval.py --all --k 1 3 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR
from common.evaluator import load_predictions, compute_metrics
from data.dataset import load_test_dataset

FEWSHOT_K_VALUES = [1, 3, 5]


def _get_fewshot_models():
    """Discover which registered models support few-shot inference."""
    from models import get_wrapper
    candidates = ["llava", "llava-next", "qwen2vl", "qwen3vl", "internvl35"]
    available = []
    for name in candidates:
        try:
            wrapper = get_wrapper(name)
            if wrapper.supports_fewshot:
                available.append(name)
        except (ValueError, ImportError):
            pass
    return available


def _pred_filename_from_prompt(prompt_key: str) -> str:
    return f"predictions_prompt_{prompt_key.lower()}.jsonl"


def _metrics_filename_from_prompt(prompt_key: str) -> str:
    return f"metrics_prompt_{prompt_key.lower()}.json"


def _pred_filename_from_k(k: int) -> str:
    return f"predictions_fewshot_k{k}.jsonl"


def _metrics_filename_from_k(k: int) -> str:
    return f"metrics_fewshot_k{k}.json"


def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation")
    parser.add_argument("--model", type=str, help="Single model name.")
    parser.add_argument("--prompt", type=str, default="A",
                        help="Prompt key (zero-shot mode).")
    parser.add_argument("--k", nargs="+", type=int, default=[],
                        help="Few-shot k values (few-shot mode).")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all model/prompt combos.")
    parser.add_argument("--ref_free_only", action="store_true",
                        help="Skip ref-based metrics.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Determine combos
    if args.all:
        combos = []
        if args.k:
            # Few-shot mode: all few-shot models × all k
            fewshot_models = _get_fewshot_models()
            print(f"[Discover] Few-shot models: {fewshot_models}")
            for name in fewshot_models:
                for k in args.k:
                    combos.append((name, None, k, args.ref_free_only))
        else:
            # Zero-shot mode: all registered models × prompt A, + sensitivity B/C
            from config import MODEL_REGISTRY
            for name, _, _ in MODEL_REGISTRY:
                combos.append((name, "A", None, False))
            for name in ["llava", "qwen2vl"]:
                for pk in ["B", "C"]:
                    combos.append((name, pk, None, True))
    else:
        if args.k:
            combos = [(args.model, None, k, args.ref_free_only)
                      for k in args.k]
        else:
            combos = [(args.model, args.prompt, None, args.ref_free_only)]

    # Load test dataset once for references
    ds = load_test_dataset(subsample=0)
    image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}

    all_metrics = {}
    for model_name, prompt_key, k, skip_ref_based in combos:
        if k is not None:
            # Few-shot
            pred_file = os.path.join(
                OUTPUT_DIR, model_name, _pred_filename_from_k(k))
            metrics_file = os.path.join(
                OUTPUT_DIR, model_name, _metrics_filename_from_k(k))
            label = f"k={k}"
        else:
            # Zero-shot
            pred_file = os.path.join(
                OUTPUT_DIR, model_name, _pred_filename_from_prompt(prompt_key))
            metrics_file = os.path.join(
                OUTPUT_DIR, model_name,
                _metrics_filename_from_prompt(prompt_key))
            label = prompt_key

        predictions = load_predictions(pred_file)
        if not predictions:
            print(f"[ERROR] No predictions for {model_name}/{label}")
            continue

        print(f"\n[Eval] {model_name}/{label}: {len(predictions)} captions")

        # Filter references to images that have predictions
        pred_img_ids = set(predictions.keys())
        ds.image_ids = sorted(set(ds.image_ids) & pred_img_ids)
        references = ds.all_references_dict()

        metrics = compute_metrics(
            predictions=predictions,
            references=references,
            image_paths=image_paths,
            metrics_file=metrics_file,
            device=args.device,
            skip_ref_based=skip_ref_based,
        )
        if metrics:
            key = f"{model_name}/{label}"
            all_metrics[key] = metrics

    # Save global summary
    if all_metrics:
        if args.k:
            summary_file = os.path.join(OUTPUT_DIR, "fewshot_all_metrics.json")
        else:
            summary_file = os.path.join(OUTPUT_DIR, "zeroshot_all_metrics.json")
        # Merge with existing for incremental runs
        existing = {}
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                existing = json.load(f)
        existing.update(all_metrics)
        with open(summary_file, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n[Summary] → {summary_file}")


if __name__ == "__main__":
    main()
