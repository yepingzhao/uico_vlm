#!/usr/bin/env python3
"""Run few-shot VLM inference on UICO test set.

Usage:
    # Quick test with 3 images, k=1
    python run_fewshot.py --models llava --k 1 --subsample 3

    # Dev run: 500 images, k=1,3,5
    python run_fewshot.py --models llava qwen2vl qwen3vl --k 1 3 5 --subsample 500

    # Full run: all 3500 images, k=1,3,5
    python run_fewshot.py --models llava qwen2vl qwen3vl --k 1 3 5

    # Qwen3-VL few-shot (replaces standalone run_fewshot_qwen3vl.py)
    python run_fewshot.py --models qwen3vl --k 1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OUTPUT_DIR,
    RANDOM_SEED,
    MAX_NEW_TOKENS,
)
from data.dataset import load_test_dataset
from fewshot.sampler import sample_examples
from models.utils import load_checkpoint


# Few-shot prompt template: k examples + instruction
FEWSHOT_PROMPT = (
    "Now describe any urban incivility or civic norm violations "
    "visible in the image above in one or two sentences."
)


from models import get_wrapper as _get_wrapper


def run_fewshot(
    model_names: list,
    k_values: list,
    subsample: int = 0,
    device: str = "cuda:0",
):
    """Run few-shot inference for each (model, k) combination.

    Args:
        model_names: ["llava", "qwen2vl"].
        k_values: [1, 3, 5].
        subsample: Number of test images (0 = full set).
        device: CUDA device.
    """
    # Load test dataset
    ds = load_test_dataset(subsample=subsample, seed=RANDOM_SEED)
    print(f"[Data] Loaded {len(ds)} test images")
    image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}

    # Pre-sample examples for each k (same examples for all test images)
    fewshot_cache = os.path.join(OUTPUT_DIR, "fewshot_cache")
    examples_cache = {}
    for k in k_values:
        examples_cache[k] = sample_examples(k, seed=RANDOM_SEED, cache_dir=fewshot_cache)
        print(f"[FewShot] k={k}: sampled {len(examples_cache[k])} examples")

    for model_name in model_names:
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"[FewShot] model={model_name}, k={k}")
            print(f"{'='*60}")

            # Output path
            model_out_dir = os.path.join(OUTPUT_DIR, model_name)
            os.makedirs(model_out_dir, exist_ok=True)
            pred_file = os.path.join(
                model_out_dir, f"predictions_fewshot_k{k}.jsonl"
            )

            # Resume
            processed = load_checkpoint(pred_file)
            remaining = [i for i in ds.image_ids if i not in processed]
            print(f"[Resume] {len(processed)} done, {len(remaining)} remaining")

            if not remaining:
                print("[Skip] All images already processed.")
                continue

            # Load model
            wrapper = _get_wrapper(model_name)
            print(f"[Load] Loading {model_name} on {device} ...")
            t0 = time.time()
            wrapper.load(device=device)
            print(f"[Load] Done in {time.time() - t0:.1f}s")

            # Get examples for this k
            example_images, example_captions = zip(*examples_cache[k])
            example_images = list(example_images)
            example_captions = list(example_captions)

            # Inference loop
            with open(pred_file, "a") as f_out:
                for i, img_id in enumerate(remaining):
                    img_path = image_paths[img_id]
                    try:
                        caption = wrapper.generate_fewshot(
                            test_image_path=img_path,
                            prompt_template=FEWSHOT_PROMPT,
                            example_images=example_images,
                            example_captions=example_captions,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                    except Exception as e:
                        print(f"  [ERROR] image_id={img_id}: {e}", file=sys.stderr)
                        caption = ""

                    record = {
                        "image_id": img_id,
                        "file_name": os.path.basename(img_path),
                        "caption": caption,
                        "prompt": f"fewshot_k{k}",
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if (i + 1) % 10 == 0:
                        f_out.flush()
                        print(
                            f"  [{i+1}/{len(remaining)}] {caption[:80]}...",
                            flush=True,
                        )

            # Clean up
            del wrapper
            if device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()

            print(f"[Done] model={model_name}, k={k} → {pred_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot VLM Inference")
    parser.add_argument(
        "--models", nargs="+", default=["llava", "qwen2vl", "qwen3vl"],
        help="Model short names."
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 3, 5],
        help="Number of few-shot examples."
    )
    parser.add_argument(
        "--subsample", type=int, default=0,
        help="Number of test images (0 = full set)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device."
    )
    args = parser.parse_args()

    print(f"[Config] models={args.models}, k={args.k}, "
          f"subsample={args.subsample or 'full'}, device={args.device}")

    run_fewshot(
        model_names=args.models,
        k_values=args.k,
        subsample=args.subsample,
        device=args.device,
    )
