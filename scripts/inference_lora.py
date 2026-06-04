#!/usr/bin/env python3
"""Run inference with the QLoRA fine-tuned LLaVA model on the UICO test set."""

import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, DATA_BASE, RANDOM_SEED
from data.dataset import load_test_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LORA_DIR = os.path.join(OUTPUT_DIR, "llava-lora")
PRED_FILE = os.path.join(OUTPUT_DIR, "llava-lora", "predictions_prompt_a.jsonl")
DEVICE = "cuda:0"


def main():
    os.makedirs(LORA_DIR, exist_ok=True)

    # Load base model with 4-bit (same as training)
    print("[Load] Loading QLoRA model...")
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()
    processor = AutoProcessor.from_pretrained(LORA_DIR)
    print(f"[Load] Done in {time.time() - t0:.1f}s")

    # Load test data
    ds = load_test_dataset(subsample=0, seed=RANDOM_SEED)
    print(f"[Data] {len(ds)} test images")

    # Resume
    processed = set()
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE) as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["image_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    remaining = [i for i in ds.image_ids if i not in processed]
    print(f"[Resume] {len(processed)} done, {len(remaining)} remaining")

    if not remaining:
        print("[Skip] All done.")
        return

    # Use the training-format prompt for inference
    user_prompt = "Describe this urban scene in one sentence."

    with open(PRED_FILE, "a") as f_out:
        for i, img_id in enumerate(remaining):
            img_path = ds.get_image_path(img_id)
            image = Image.open(img_path).convert("RGB")

            # Same format as training: USER: <image>\nprompt ASSISTANT:
            conv = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }]
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(DEVICE, torch.float16)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                )
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            caption = processor.decode(
                generated[0], skip_special_tokens=True
            ).strip()

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

    print(f"[Done] → {PRED_FILE}")


if __name__ == "__main__":
    main()
