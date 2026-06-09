#!/usr/bin/env python3
"""QLoRA fine-tuning + inference on the UICO dataset.

Supports three modes:
  - Default (no mode flag): train then run inference
  - --train: train only
  - --inference_only: inference only (requires existing checkpoint)

Usage:
    python scripts/run_lora.py --model llava                                    # train + infer
    python scripts/run_lora.py --model llava --train                            # train only
    python scripts/run_lora.py --model llava --inference_only --subsample 100   # infer only
    python scripts/run_lora.py --model llava --epochs 3 --lora_r 8 --train     # custom hparams
    python scripts/run_lora.py --model qwen2vl --max_samples 500 --no_swanlab  # quick test

Output:
    outputs/<model>-lora/  -> adapter weights, training.log
    outputs/<model>-lora/predictions_prompt_a.jsonl  -> inference results
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, RANDOM_SEED, MAX_NEW_TOKENS
from data.dataset import DatasetBundle, load_test_dataset
from core.inference.strategies import LoRAStrategy
from core.inference.runner import InferenceRunner
from core.training.adapters import get_training_adapter
from core.training.runner import TrainingRunner
from config.prompts import PROMPT_A


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA VLM Fine-Tuning + Inference")
    # ---- Model ---------------------------------------------------------
    parser.add_argument("--model", type=str, default="llava",
                        help="Model short name (see config/training.py).")

    # ---- Mode ----------------------------------------------------------
    parser.add_argument("--train", action="store_true", default=False,
                        help="Run training (default: train+inference).")
    parser.add_argument("--inference_only", action="store_true",
                        default=False,
                        help="Skip training, run inference only.")

    # ---- Training hyperparams ------------------------------------------
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit training samples (0 = all).")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=50,
                        help="Run validation every N global steps "
                             "(0 = epoch-only).")
    parser.add_argument("--val_samples", type=int, default=10,
                        help="Number of validation images for collapse "
                             "detection.")
    parser.add_argument("--val_max_samples", type=int, default=0,
                        help="Max val images for best-checkpoint selection "
                             "(0 = all).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_swanlab", action="store_true",
                        help="Disable SwanLab logging.")

    # ---- Inference opts ------------------------------------------------
    parser.add_argument("--subsample", type=int, default=0,
                        help="Number of test images for inference "
                             "(0 = full set).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing predictions before "
                             "inference.")
    parser.add_argument("--partition", type=str, default="",
                        help="Strided partition: 'k/n' (every n-th image).")
    parser.add_argument("--chunk", type=str, default="",
                        help="Contiguous chunk: 'k/n' (block k of n).")

    args = parser.parse_args()

    # Resolve mode
    do_train = args.train or not args.inference_only
    do_inference = args.inference_only or not args.train
    # If neither --train nor --inference_only: both (default)

    if not do_train and not do_inference:
        print("[ERROR] Conflicting mode flags.", file=sys.stderr)
        sys.exit(1)

    lora_dir = os.path.join(OUTPUT_DIR, f"{args.model}-lora")

    # ── Training ───────────────────────────────────────────────────
    if do_train:
        adapter = get_training_adapter(args.model)
        runner = TrainingRunner(
            model_name=args.model,
            adapter=adapter,
            args=args,
        )
        runner.train()
        # lora_dir now contains trained weights
    elif not os.path.isdir(lora_dir):
        print(f"[ERROR] LoRA checkpoint not found: {lora_dir}",
              file=sys.stderr)
        sys.exit(1)

    # ── Resolve best checkpoint for inference ────────────────────────
    best_ckpt_dir = os.path.join(lora_dir, "best_checkpoint")
    best_adapter = os.path.join(best_ckpt_dir, "adapter_model.safetensors")
    if do_inference and os.path.isfile(best_adapter):
        print(f"[Best] Using best checkpoint: {best_ckpt_dir}")
        lora_dir = best_ckpt_dir
    elif do_inference and os.path.isdir(best_ckpt_dir):
        print(f"[Best] best_checkpoint/ exists but no adapter found, "
              f"using final: {lora_dir}")

    # ── Inference ──────────────────────────────────────────────────
    if do_inference:
        if not os.path.isdir(lora_dir):
            print(f"[ERROR] LoRA checkpoint not found: {lora_dir}",
                  file=sys.stderr)
            sys.exit(1)

        print(f"\n[Config] inference: model={args.model}, "
              f"subsample={args.subsample or 'full'}, "
              f"device={args.device}"
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
        inf_runner = InferenceRunner(
            strategy=strategy,
            output_dir=lora_dir,
            filename="predictions_prompt_a.jsonl",
            bundle=bundle,
            prompt_label="a",
        )
        partition = chunk = None
        if args.partition:
            k, n = args.partition.split("/")
            partition = (int(k), int(n))
        if args.chunk:
            k, n = args.chunk.split("/")
            chunk = (int(k), int(n))
        inf_runner.run(overwrite=args.overwrite, device=args.device,
                       partition=partition, chunk=chunk)


if __name__ == "__main__":
    main()
