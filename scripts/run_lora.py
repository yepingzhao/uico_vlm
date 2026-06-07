#!/usr/bin/env python3
"""Run inference with a QLoRA fine-tuned VLM on the UICO test set.

Usage:
    python scripts/run_lora.py --model llava
    python scripts/run_lora.py --model llava --subsample 100
    python scripts/run_lora.py --model internvl2
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
from common.strategies import LoRAStrategy
from common.pipeline import InferenceRunner
from prompts.templates import PROMPT_A


def main():
    parser = argparse.ArgumentParser(description="LoRA VLM Inference")
    parser.add_argument("--model", type=str, default="llava",
                        help="Model short name.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Number of test images (0 = full set).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing predictions before starting.")
    args = parser.parse_args()

    lora_dir = os.path.join(OUTPUT_DIR, f"{args.model}-lora")
    if not os.path.isdir(lora_dir):
        print(f"[ERROR] LoRA checkpoint not found: {lora_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[Config] model={args.model}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")
    print(f"[LoRA] {lora_dir}")

    ds = load_test_dataset(subsample=args.subsample, seed=RANDOM_SEED)
    bundle = DatasetBundle.from_dataset(ds)
    print(f"[Data] Loaded {len(bundle)} images")

    strategy = LoRAStrategy(
        model_name=args.model,
        lora_dir=lora_dir,
        prompt=PROMPT_A,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    runner = InferenceRunner(
        strategy=strategy,
        output_dir=lora_dir,
        filename="predictions_prompt_a.jsonl",
        bundle=bundle,
        prompt_label="a",
    )
    runner.run(overwrite=args.overwrite, device=args.device)


if __name__ == "__main__":
    main()
