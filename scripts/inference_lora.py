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

from config import MAX_NEW_TOKENS, OUTPUT_DIR, RANDOM_SEED
from config.training import get_lora_config
from data.dataset import load_test_dataset
from models.lora import load_qlora_for_inference
from models.utils import load_checkpoint
from prompts.templates import PROMPT_A

# InternVL image token constants (matching data/training_dataset.py)
_INTERNVL_IMG_START = "<img>"
_INTERNVL_IMG_END = "</img>"
_INTERNVL_IMG_CONTEXT = "<IMG_CONTEXT>"
_INTERNVL_NUM_IMAGE_TOKENS = 256


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
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        model_kwargs=model_cfg.get("model_kwargs"),
    )
    print(f"[Load] Done in {time.time() - t0:.1f}s")

    is_internvl = args.model in ("internvl2", "internvl3", "internvl35")

    # InternVL: processor from load_qlora_for_inference may be an
    # AutoProcessor that tries to use InternVLProcessor (broken for
    # InternVL3.5's TokenizersBackend). Replace with tokenizer+image_processor.
    image_processor = None
    if is_internvl:
        from transformers import AutoImageProcessor
        if args.model == "internvl35":
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=model_cfg.get("trust_remote_code", False),
                local_files_only=True,
            )
        image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            local_files_only=True,
        )
        # single-patch mode for stability
        if args.model == "internvl35":
            image_processor.max_patches = 1
        # Set img_context_token_id on the inner model for generate()
        from transformers import AutoModel
        _img_ctx = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        _inner = model
        for _step in ("base_model", "model"):
            _inner = getattr(_inner, _step, _inner)
        if hasattr(_inner, "img_context_token_id"):
            _inner.img_context_token_id = _img_ctx
            print(f"[InternVL] img_context_token_id={_img_ctx}")

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

            if is_internvl:
                # InternVL: plain-string chat template with IMG_CONTEXT expansion.
                # The tokenizer (processor) handles text; image_processor handles
                # pixel_values. Generate takes them as separate kwargs.
                img_tokens = (
                    f"{_INTERNVL_IMG_START}"
                    f"{_INTERNVL_IMG_CONTEXT * _INTERNVL_NUM_IMAGE_TOKENS}"
                    f"{_INTERNVL_IMG_END}"
                )
                conv = [{
                    "role": "user",
                    "content": f"{img_tokens}\n{user_prompt}",
                }]
                result = processor.apply_chat_template(
                    conv, add_generation_prompt=True)
                # Normalize: BatchEncoding → list
                if hasattr(result, "data") and "input_ids" in result.data:
                    ids_list = result.data["input_ids"]
                elif isinstance(result, dict) and "input_ids" in result:
                    ids_list = result["input_ids"]
                elif hasattr(result, "ids"):
                    ids_list = result.ids
                else:
                    ids_list = result
                if isinstance(ids_list, list) and len(ids_list) > 0 and \
                        isinstance(ids_list[0], list):
                    ids_list = ids_list[0]
                input_ids = torch.tensor(
                    [ids_list], dtype=torch.long).to(args.device)
                attention_mask = torch.ones_like(input_ids)
                img_outputs = image_processor(
                    images=image, return_tensors="pt")
                pixel_values = img_outputs["pixel_values"].to(
                    args.device, torch.bfloat16)
                input_len = input_ids.shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                # InternVL.generate() passes inputs_embeds to the language
                # model, so output_ids contains ONLY generated tokens (no
                # input prepended). Standard models prepend input tokens.
                generated = output_ids
                caption = processor.decode(
                    generated[0], skip_special_tokens=True).strip()
            else:
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
                        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
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
