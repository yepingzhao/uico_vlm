#!/usr/bin/env python3
"""LoRA fine-tune LLaVA-1.5-7B on the UICO training set.

Key design:
  - LoRA rank=8 on all language-model attention projections (q,k,v,o)
  - Multi-modal projector trained fully (small, ~20M params)
  - Vision encoder frozen
  - 1 epoch, ~30K images, batch_size=4 + gradient accumulation
  - Causal LM loss on caption tokens (ignores image placeholder)

Usage:
    python -m vlm_eval.train.train_llava_lora

Output:
    vlm_eval/outputs/llava-lora/  → adapter weights, config
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_BASE, OUTPUT_DIR
from data.dataset import _resolve_image_path

IMAGES_BASE_DIR = os.path.join(DATA_BASE, "images")


# ── Config ──────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    output_dir: str = os.path.join(OUTPUT_DIR, "llava-lora")

    # Data
    train_ann_file: str = os.path.join(DATA_BASE, "annotations", "captions_train.json")
    max_samples: int = 0        # 0 = all training images

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training
    batch_size: int = 1         # per-device (bs>1 OOM with 576 img tokens)
    gradient_accumulation_steps: int = 8  # effective batch = 1 * 8 = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_epochs: int = 1
    max_grad_norm: float = 1.0

    # Image preprocessing
    max_length: int = 128       # max generated tokens (also truncates target)

    # Checkpoint
    save_steps: int = 2000
    logging_steps: int = 50

    # Device
    device: str = "cuda:0"
    seed: int = 42


# ── Dataset ──────────────────────────────────────────────────────────
class UICOInstructionDataset(Dataset):
    """UICO training set → instruction-following format for LLaVA.

    Each image has 5 reference captions. We randomly select one per image
    at dataset init time (fixed by seed) to keep training stable.
    """

    def __init__(self, ann_file: str, processor, config: TrainConfig):
        self.processor = processor
        self.config = config
        self.samples = []  # list of (image_path, caption)

        with open(ann_file) as f:
            data = json.load(f)

        # Image id → file_name
        id_to_file: Dict[int, str] = {}
        for img in data["images"]:
            id_to_file[img["id"]] = img["file_name"]

        # Group captions by image_id
        by_image: Dict[int, list] = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann["caption"].strip())

        # Randomly select 1 caption per image
        rng = torch.Generator().manual_seed(config.seed)
        image_ids = sorted(by_image.keys())
        if config.max_samples > 0:
            image_ids = image_ids[:config.max_samples]

        for img_id in image_ids:
            fname = id_to_file[img_id]
            path = _resolve_image_path(IMAGES_BASE_DIR, fname)
            if not os.path.exists(path):
                continue
            captions = by_image[img_id]
            idx = torch.randint(0, len(captions), (1,), generator=rng).item()
            self.samples.append((path, captions[idx]))

        print(f"[Dataset] {len(self.samples)} training examples "
              f"({len(image_ids)} images)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        # Training format: "USER: <image>\nPROMPT ASSISTANT: {caption}"
        # The model should predict only the "{caption}" part.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this urban scene in one sentence."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        # Process image+text TOGETHER — expands <image> into 576 patch tokens
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)

        # Labels: only predict the assistant response (caption part)
        # Strategy: tokenize the prompt without the assistant response,
        # then mask everything up to that point
        user_prompt = self.processor.apply_chat_template(
            [conversation[0]], add_generation_prompt=True
        )
        user_ids = self.processor.tokenizer(
            user_prompt, return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        # Mask user prompt tokens + image tokens
        labels[:len(user_ids)] = -100
        # Also mask any remaining image tokens
        labels[input_ids == 32000] = -100

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "labels": labels,
        }


# ── Training ──────────────────────────────────────────────────────────
def train():
    config = TrainConfig()
    torch.manual_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[Config] {config}")
    print(f"[Device] {config.device}")

    # ── Load model (QLoRA: 4-bit quantized base + LoRA adapters) ──
    print("[Load] Loading LLaVA-1.5-7B (4-bit QLoRA) ...")
    t0 = time.time()

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map=config.device,
        torch_dtype=torch.float16,
    )

    # Freeze vision encoder (already quantized, no gradients needed)
    # Apply LoRA to language model attention layers
    # Note: multi_modal_projector is quantized 4-bit and stays frozen;
    # all trainable capacity comes from LoRA adapters.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    model.print_trainable_parameters()
    print(f"[Load] Done in {time.time() - t0:.1f}s")

    # ── Load data ──
    processor = AutoProcessor.from_pretrained(config.model_id)

    def collate_fn(batch):
        """Pad variable-length sequences and stack pixel values."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # Pad input_ids and labels to max length in batch
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_token_id = processor.tokenizer.pad_token_id

        input_ids_list, labels_list = [], []
        for item in batch:
            ids = item["input_ids"]
            labs = item["labels"]
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_token_id)])
                labs = torch.cat([labs, torch.full((pad_len,), -100)])
            input_ids_list.append(ids)
            labels_list.append(labs)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(input_ids_list).ne(pad_token_id).long(),
            "labels": torch.stack(labels_list),
        }

    train_ds = UICOInstructionDataset(config.train_ann_file, processor, config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,   # 0 avoids multiprocessing issues with PIL images
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Optimizer & scheduler ──
    total_steps = (
        len(train_loader) // config.gradient_accumulation_steps * config.num_epochs
    )
    warmup_steps = int(total_steps * config.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ──
    model.train()
    global_step = 0
    total_loss = 0.0

    print(f"\n[Train] {total_steps} steps, {warmup_steps} warmup")
    print(f"[Train] Effective batch size: "
          f"{config.batch_size * config.gradient_accumulation_steps}")

    # ── SwanLab logging ──
    import swanlab
    swanlab.init(
        project="uico_vlm-llava-lora",
        config={
            "model": "llava-1.5-7b-hf",
            "method": "QLoRA",
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "batch_size": config.batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "effective_batch": config.batch_size * config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "total_steps": total_steps,
            "train_images": len(train_ds),
        },
    )

    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}\n[Epoch] {epoch + 1}/{config.num_epochs}\n{'='*50}")
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(
                config.device, dtype=torch.float16
            )
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()
            epoch_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg = total_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step={global_step}/{total_steps} loss={avg:.4f} "
                          f"lr={lr:.2e}", flush=True)
                    swanlab.log({"loss": avg, "lr": lr}, step=global_step)
                    total_loss = 0.0

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    ckpt_dir = os.path.join(config.output_dir,
                                            f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    print(f"  [Save] {ckpt_dir}")

        avg_epoch_loss = epoch_loss / len(train_loader) * config.gradient_accumulation_steps
        print(f"[Epoch {epoch+1}] avg_loss={avg_epoch_loss:.4f}")

    # ── Final save ──
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    swanlab.finish()
    print(f"\n[Done] Model saved → {config.output_dir}")
    print(f"  adapter_config.json + adapter_model.safetensors")


if __name__ == "__main__":
    train()
