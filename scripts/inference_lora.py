#!/usr/bin/env python3
"""Run inference with a QLoRA fine-tuned VLM on the UICO test set.

Usage:
    python scripts/inference_lora.py --model llava
    python scripts/inference_lora.py --model llava-next
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, RANDOM_SEED
from config.training import get_lora_config
from data.dataset import load_test_dataset
from models.lora import load_qlora_for_inference
from models.utils import load_checkpoint
from prompts.templates import PROMPT_A


def _import_class(class_name: str):
    """Import a HuggingFace class by name."""
    import transformers
    if hasattr(transformers, class_name):
        return getattr(transformers, class_name)
    raise ValueError(f"Unknown class: {class_name}")


def main():
    parser = argparse.ArgumentParser(description="LoRA VLM Inference")
    parser.add_argument("--model", type=str, default="llava",
                        help="Model short name.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model_cfg = get_lora_config(args.model)
    model_id = model_cfg["model_id"]
    lora_dir = os.path.join(OUTPUT_DIR, f"{args.model}-lora")
    pred_file = os.path.join(lora_dir, "predictions_prompt_a.jsonl")

    print(f"[Model] {args.model} -> {model_id}")
    print(f"[LoRA] {lora_dir}")

    # ── Load model ──
    print("[Load] Loading QLoRA model...")
    t0 = time.time()
    model_class = _import_class(model_cfg["model_class_name"])
    model, processor = load_qlora_for_inference(
        model_class, model_id, lora_dir, args.device,
    )
    print(f"[Load] Done in {time.time() - t0:.1f}s")

    # ── Load test data ──
    ds = load_test_dataset(subsample=0, seed=RANDOM_SEED)
    print(f"[Data] {len(ds)} test images")

    # ── Resume ──
    processed = load_checkpoint(pred_file)
    remaining = [i for i in ds.image_ids if i not in processed]
    print(f"[Resume] {len(processed)} done, {len(remaining)} remaining")

    if not remaining:
        print("[Skip] All done.")
        return

    os.makedirs(lora_dir, exist_ok=True)
    user_prompt = PROMPT_A

    with open(pred_file, "a") as f_out:
        for i, img_id in enumerate(remaining):
            img_path = ds.get_image_path(img_id)
            image = Image.open(img_path).convert("RGB")

            conv = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }]
            prompt = processor.apply_chat_template(
                conv, add_generation_prompt=True)
            inputs = processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(args.device, torch.bfloat16)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            caption = processor.decode(
                generated[0], skip_special_tokens=True).strip()

            record = {
                "image_id": img_id,
                "file_name": os.path.basename(img_path),
                "caption": caption,
                "prompt": "a",
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                f_out.flush()
                print(f"  [{i+1}/{len(remaining)}] {caption[:80]}...", flush=True)

    print(f"[Done] -> {pred_file}")


if __name__ == "__main__":
    main()
