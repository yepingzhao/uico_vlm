#!/usr/bin/env python3
"""Run VLM inference on UICO test set with checkpoint/resume support.

Usage:
    # Zero-shot
    python scripts/run_inference.py --mode zeroshot --models llava --prompt A
    python scripts/run_inference.py --mode zeroshot --models llava qwen2vl --prompt B
    python scripts/run_inference.py --mode zeroshot --models llava --prompt A --overwrite

    # Few-shot
    python scripts/run_inference.py --mode fewshot --models llava --k 1 --subsample 3
    python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5
    python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5 --subsample 500
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, RANDOM_SEED, MAX_NEW_TOKENS
from data.dataset import DatasetBundle, load_test_dataset
from core.inference.strategies import ZeroShotStrategy, FewShotStrategy
from core.inference.runner import InferenceRunner
from config.prompts import PROMPT_MAP, PROMPT_FEWSHOT


def run_zeroshot(args):
    """Zero-shot inference across one or more models."""
    prompt_text = PROMPT_MAP[args.prompt]
    print(f"[Config] mode=zeroshot, models={args.models}, prompt={args.prompt}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")
    print(f"[Prompt] {args.prompt}: {prompt_text[:100]}...")

    ds = load_test_dataset(subsample=args.subsample, seed=RANDOM_SEED)
    bundle = DatasetBundle.from_dataset(ds)
    print(f"[Data] Loaded {len(bundle)} images")

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"[Model] {model_name}")
        print(f"{'='*60}")

        model_out_dir = os.path.join(OUTPUT_DIR, model_name)
        filename = f"predictions_prompt_{args.prompt.lower()}.jsonl"

        strategy = ZeroShotStrategy(
            model_name=model_name,
            prompt=prompt_text,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        runner = InferenceRunner(
            strategy=strategy,
            output_dir=model_out_dir,
            filename=filename,
            bundle=bundle,
            prompt_label=args.prompt,
        )
        runner.run(overwrite=args.overwrite, device=args.device)


def run_fewshot(args):
    """Few-shot inference across one or more models and k values."""
    print(f"[Config] mode=fewshot, models={args.models}, k={args.k}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")

    from models.fewshot import sample_examples

    ds = load_test_dataset(subsample=args.subsample, seed=RANDOM_SEED)
    bundle = DatasetBundle.from_dataset(ds)
    print(f"[Data] Loaded {len(bundle)} images")

    # Pre-sample examples for each k (shared across all models)
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


def main():
    parser = argparse.ArgumentParser(
        description="VLM Inference on UICO Test Set")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["zeroshot", "fewshot"],
                        help="Inference mode.")
    parser.add_argument("--models", nargs="+", default=["llava"],
                        help="Model short names.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Number of images (0 = full test set).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing predictions before starting.")
    # Zeroshot-only
    parser.add_argument("--prompt", type=str, default="A",
                        choices=["A", "B", "C"],
                        help="Prompt template key (zeroshot only).")
    # Fewshot-only
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5],
                        help="Number of few-shot examples per model.")
    args = parser.parse_args()

    if args.mode == "zeroshot":
        run_zeroshot(args)
    else:
        run_fewshot(args)


if __name__ == "__main__":
    main()
