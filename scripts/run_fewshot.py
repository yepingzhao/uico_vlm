#!/usr/bin/env python3
"""Run few-shot VLM inference on UICO test set.

Usage:
    python scripts/run_fewshot.py --models llava --k 1 --subsample 3
    python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5 --subsample 500
    python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, RANDOM_SEED, MAX_NEW_TOKENS
from data.dataset import load_test_dataset
from common.dataset_bundle import DatasetBundle
from common.strategies import FewShotStrategy
from common.pipeline import InferenceRunner
from models.fewshot import sample_examples
from config.prompts import PROMPT_FEWSHOT


def main():
    parser = argparse.ArgumentParser(description="Few-Shot VLM Inference")
    parser.add_argument("--models", nargs="+", default=["llava", "qwen2vl"],
                        help="Model short names.")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5],
                        help="Number of few-shot examples.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Number of test images (0 = full set).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing predictions before starting.")
    args = parser.parse_args()

    print(f"[Config] models={args.models}, k={args.k}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")

    # Load dataset once
    ds = load_test_dataset(subsample=args.subsample, seed=RANDOM_SEED)
    bundle = DatasetBundle.from_dataset(ds)
    print(f"[Data] Loaded {len(bundle)} images")

    # Pre-sample examples for each k (same examples for all test images)
    fewshot_cache = os.path.join(OUTPUT_DIR, "fewshot_cache")
    examples_cache = {}
    for k in args.k:
        examples_cache[k] = sample_examples(
            k, seed=RANDOM_SEED, cache_dir=fewshot_cache)
        print(f"[FewShot] k={k}: sampled {len(examples_cache[k])} examples")

    for model_name in args.models:
        for k in args.k:
            print(f"\n{'='*60}")
            print(f"[FewShot] model={model_name}, k={k}")
            print(f"{'='*60}")

            model_out_dir = os.path.join(OUTPUT_DIR, model_name)
            filename = f"predictions_fewshot_k{k}.jsonl"

            example_images, example_captions = zip(*examples_cache[k])

            strategy = FewShotStrategy(
                model_name=model_name,
                prompt_template=PROMPT_FEWSHOT,
                k=k,
                example_images=list(example_images),
                example_captions=list(example_captions),
                max_new_tokens=MAX_NEW_TOKENS,
            )
            runner = InferenceRunner(
                strategy=strategy,
                output_dir=model_out_dir,
                filename=filename,
                bundle=bundle,
                prompt_label=f"fewshot_k{k}",
            )
            runner.run(overwrite=args.overwrite, device=args.device)


if __name__ == "__main__":
    main()
