#!/usr/bin/env python3
"""Run zero-shot VLM inference on UICO test set with checkpoint/resume support.

Usage:
    python scripts/run_zeroshot.py --models llava --subsample 1000 --prompt A
    python scripts/run_zeroshot.py --models llava qwen2vl --prompt B
    python scripts/run_zeroshot.py --models llava --prompt A --overwrite
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
from common.strategies import ZeroShotStrategy
from common.pipeline import InferenceRunner
from config.prompts import PROMPT_MAP


def main():
    parser = argparse.ArgumentParser(description="VLM Zero-Shot Inference")
    parser.add_argument("--models", nargs="+", default=["llava"],
                        help="Model short names.")
    parser.add_argument("--prompt", type=str, default="A",
                        choices=["A", "B", "C"],
                        help="Prompt template key.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Number of images (0 = full test set).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing predictions before starting.")
    args = parser.parse_args()

    prompt_text = PROMPT_MAP[args.prompt]
    print(f"[Config] models={args.models}, prompt={args.prompt}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")
    print(f"[Prompt] {args.prompt}: {prompt_text[:100]}...")

    # Load dataset once (shared across models)
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


if __name__ == "__main__":
    main()
