#!/usr/bin/env python3
"""QLoRA fine-tune VLMs on the UICO training set.

Supports any model registered in config/training.py:MODEL_LORA_CONFIGS.

Usage:
    python scripts/train_lora.py --model llava
    python scripts/train_lora.py --model llava-next --lora_r 16 --epochs 3
    python scripts/train_lora.py --model llava --max_samples 500   # quick test

Output:
    outputs/<model>-lora/  -> adapter weights, config
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR
from config.training import TrainingConfig, get_lora_config
from data.training_dataset import UICOInstructionDataset, collate_fn
from models.lora import make_lora_config, load_qlora_model


def _import_class(class_name: str):
    """Import a HuggingFace class by name."""
    import transformers
    if hasattr(transformers, class_name):
        return getattr(transformers, class_name)
    raise ValueError(f"Unknown class: {class_name}")


def train():
    parser = argparse.ArgumentParser(description="QLoRA VLM Fine-Tuning")
    parser.add_argument("--model", type=str, default="llava",
                        help="Model short name (see config/training.py).")
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_swanlab", action="store_true",
                        help="Disable SwanLab logging.")
    args = parser.parse_args()

    # ── Resolve model config ──
    model_cfg = get_lora_config(args.model)

    config = TrainingConfig()
    config.model_id = model_cfg["model_id"]
    config.model_class_name = model_cfg["model_class_name"]
    config.processor_class_name = model_cfg["processor_class_name"]
    config.target_modules = model_cfg["target_modules"]
    config.output_dir = os.path.join(OUTPUT_DIR, f"{args.model}-lora")

    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.max_samples = args.max_samples
    config.save_steps = args.save_steps
    config.logging_steps = args.logging_steps
    config.device = args.device
    config.seed = args.seed

    torch.manual_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[Config] model={args.model} model_id={config.model_id}")
    print(f"[Config] LoRA r={config.lora_r} alpha={config.lora_alpha}")
    print(f"[Config] batch={config.batch_size}x{config.gradient_accumulation_steps}"
          f" lr={config.learning_rate} epochs={config.num_epochs}")

    # ── Load model ──
    print(f"[Load] {config.model_id} (4-bit QLoRA) ...")
    t0 = time.time()
    model_class = _import_class(config.model_class_name)
    lora_config = make_lora_config(
        r=config.lora_r, alpha=config.lora_alpha,
        dropout=config.lora_dropout, target_modules=list(config.target_modules),
    )
    model = load_qlora_model(model_class, config.model_id, lora_config, config.device)
    model.print_trainable_parameters()
    print(f"[Load] Done in {time.time() - t0:.1f}s")

    # ── Load data ──
    processor_class = _import_class(config.processor_class_name)
    processor_kwargs = {}
    if args.model == "qwen2vl":
        # Lower resolution for QLoRA training to fit 24GB VRAM
        # (Qwen2.5-VL dynamic resolution can produce very large feature maps)
        processor_kwargs = dict(
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
    processor = processor_class.from_pretrained(
        config.model_id, **processor_kwargs,
    )

    train_ds = UICOInstructionDataset(
        ann_file=config.train_ann_file,
        processor=processor,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    def _collate(batch):
        return collate_fn(processor, batch)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=_collate,
    )

    # ── Optimizer ──
    total_steps = (
        len(train_loader) // config.gradient_accumulation_steps * config.num_epochs
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── SwanLab ──
    if not args.no_swanlab:
        import swanlab
        swanlab.init(
            project=f"uico_vlm-{args.model}-lora",
            config={
                "model": config.model_id, "method": "QLoRA",
                "lora_r": config.lora_r, "lora_alpha": config.lora_alpha,
                "batch_size": config.batch_size,
                "grad_accum": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "total_steps": total_steps, "train_images": len(train_ds),
            },
        )

    # ── Training loop ──
    model.train()
    global_step = 0
    total_loss = 0.0
    print(f"\n[Train] {total_steps} steps, {warmup_steps} warmup")

    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}\n[Epoch] {epoch+1}/{config.num_epochs}\n{'='*50}")
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(
                config.device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(config.device)

            model_kwargs = dict(
                pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels,
            )
            if image_grid_thw is not None:
                model_kwargs["image_grid_thw"] = image_grid_thw

            outputs = model(**model_kwargs)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
            epoch_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg = total_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step={global_step}/{total_steps} loss={avg:.4f} "
                          f"lr={lr:.2e}", flush=True)
                    if not args.no_swanlab:
                        import swanlab
                        swanlab.log({"loss": avg, "lr": lr}, step=global_step)
                    total_loss = 0.0

                if global_step % config.save_steps == 0:
                    ckpt_dir = os.path.join(
                        config.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    print(f"  [Save] {ckpt_dir}")

        avg_ep = epoch_loss / len(train_loader) * config.gradient_accumulation_steps
        print(f"[Epoch {epoch+1}] avg_loss={avg_ep:.4f}")

    # ── Final save ──
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    if not args.no_swanlab:
        import swanlab
        swanlab.finish()
    print(f"\n[Done] -> {config.output_dir}")


if __name__ == "__main__":
    train()
