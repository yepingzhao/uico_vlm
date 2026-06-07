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
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from config import OUTPUT_DIR, VAL_IMAGES_DIR
from config.training import TrainingConfig, get_lora_config
from data.training_dataset import UICOInstructionDataset, collate_fn
from models.lora import make_lora_config, load_qlora_model
from prompts.templates import PROMPT_A


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
    parser.add_argument("--val_steps", type=int, default=500,
                        help="Run validation every N global steps (0 = epoch-only).")
    parser.add_argument("--val_samples", type=int, default=10,
                        help="Number of validation images to use for mode collapse detection.")
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
    config.val_steps = args.val_steps
    config.val_samples = args.val_samples

    torch.manual_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Training log (structured JSON for agent cross-session sync) ──
    log_path = os.path.join(config.output_dir, "training.log")

    def _log(entry: dict):
        """Append a JSON line to the training log (unbuffered)."""
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        # Also print to stdout for interactive monitoring
        if "loss" in entry:
            print(f"  step={entry['step']}/{entry['total']} "
                  f"loss={entry['loss']:.4f} lr={entry['lr']:.2e}", flush=True)

    _log({"event": "start", "model": args.model, "model_id": config.model_id,
          "lora_r": config.lora_r, "lora_alpha": config.lora_alpha,
          "lr": config.learning_rate, "epochs": config.num_epochs,
          "target_modules": list(config.target_modules),
          "warmup_ratio": config.warmup_ratio})

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
    model = load_qlora_model(model_class, config.model_id, lora_config,
                              config.device,
                              trust_remote_code=model_cfg.get("trust_remote_code", False),
                              model_kwargs=model_cfg.get("model_kwargs"))
    model.print_trainable_parameters()
    print(f"[Load] Done in {time.time() - t0:.1f}s")

    # ── Load data ──
    processor_class = _import_class(config.processor_class_name)
    processor_kwargs = {}
    is_internvl2 = args.model in ("internvl2", "internvl3", "internvl35")
    is_phi35 = args.model == "phi35-vision"
    if is_internvl2:
        # InternVL2/3/3.5: AutoProcessor returns only the tokenizer (no image
        # processor), except InternVL3.5 where processor_config.json causes
        # InternVLProcessor to be used — which crashes on TokenizersBackend
        # (missing start_image_token etc.). For InternVL3.5, force-load the
        # tokenizer directly and bypass InternVLProcessor entirely.
        # We load the image processor separately and attach it for use in the
        # training dataset.
        # Forward local_files_only for offline env
        _internvl_extra = {}
        if model_cfg.get("model_kwargs", {}).get("local_files_only"):
            _internvl_extra["local_files_only"] = True
        if args.model == "internvl35":
            from transformers import AutoTokenizer as _Tok
            processor = _Tok.from_pretrained(
                config.model_id,
                trust_remote_code=model_cfg.get("trust_remote_code", False),
                **_internvl_extra,
            )
        else:
            processor = processor_class.from_pretrained(
                config.model_id,
                trust_remote_code=model_cfg.get("trust_remote_code", False),
                **_internvl_extra,
            )
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            **_internvl_extra,
        )
        # InternVL3.5: force single-patch mode for training stability.
        # Dynamic patches produce variable-length IMG_CONTEXT (256-3072 tokens)
        # which causes extreme sequence length variation → QLoRA instability.
        if args.model == "internvl35":
            image_processor.max_patches = 1
    elif args.model == "qwen2vl":
        # Lower resolution for QLoRA training to fit 24GB VRAM
        # (Qwen2.5-VL dynamic resolution can produce very large feature maps)
        processor_kwargs = dict(
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
    elif args.model == "qwen3vl":
        # Same low-res strategy as Qwen2VL: Qwen3-VL dynamic resolution
        # can produce large feature maps → OOM risk on 24GB with QLoRA 4-bit
        processor_kwargs = dict(
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
    elif args.model == "llava-next":
        # Lower resolution for LLaVA-NeXT dynamic high resolution to fit 24GB
        # Default size={336,672} → 5 patches → OOM on 24GB with QLoRA 4-bit
        processor_kwargs = dict(
            size={"shortest_edge": 168, "longest_edge": 336},
        )
    if not is_internvl2:
        # Forward local_files_only from model_kwargs to processor (offline env)
        proc_extra = {}
        if model_cfg.get("model_kwargs", {}).get("local_files_only"):
            proc_extra["local_files_only"] = True
        processor = processor_class.from_pretrained(
            config.model_id,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            **processor_kwargs,
            **proc_extra,
        )
    if is_phi35:
        # Phi3VProcessor has no chat_template attribute (it lives on the
        # tokenizer). ProcessorMixin.apply_chat_template() accesses
        # self.chat_template first, so we copy it from the tokenizer.
        processor.chat_template = processor.tokenizer.chat_template

    train_ds = UICOInstructionDataset(
        ann_file=config.train_ann_file,
        processor=processor,
        max_samples=config.max_samples,
        seed=config.seed,
        is_internvl2=is_internvl2,
        is_phi35=is_phi35,
    )
    if is_internvl2:
        train_ds.image_processor = image_processor
        # InternVL2's forward()/generate() assert img_context_token_id is set.
        # PEFT wraps: PeftModel.base_model (LoraModel).model (InternVLChatModel).
        # When PeftModel.generate() calls base_model.generate(), LoraModel's
        # __getattr__ delegates to self.model (= InternVLChatModel).generate().
        # So we must set the attribute on InternVLChatModel directly, because
        # Python __setattr__ is NOT delegated by __getattr__.
        _img_ctx = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        # Navigate to the innermost model: PeftModel → LoraModel → InternVLChatModel
        _inner = model
        for _step in ("base_model", "model"):
            _inner = getattr(_inner, _step, _inner)
        # _inner should now be InternVLChatModel (or InternVL2-8B equivalent)
        if hasattr(_inner, "img_context_token_id"):
            _inner.img_context_token_id = _img_ctx
            print(f"[InternVL2] img_context_token_id={_img_ctx} "
                  f"set on {type(_inner).__name__}")

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

    # ── Validation images (fixed subset, zero overlap with test) ──
    def _load_val_images(val_dir: str, n_samples: int) -> list:
        """Pick n_samples deterministically from the val image directory."""
        import os as _os
        all_files = sorted(
            f for f in _os.listdir(val_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(all_files) < n_samples:
            n_samples = len(all_files)
        step = max(1, len(all_files) // n_samples)
        return [_os.path.join(val_dir, all_files[i * step])
                for i in range(n_samples)]

    def _compute_collapse_metrics(captions: list) -> dict:
        """Detect mode collapse signals from a list of captions."""
        words_per_caption = [c.lower().split() for c in captions]

        # 1. Average length
        lengths = [len(w) for w in words_per_caption]
        avg_len = sum(lengths) / max(1, len(lengths))

        # 2. Repetition ratio: unique/total per caption, then average
        rep_ratios = []
        for toks in words_per_caption:
            if len(toks) == 0:
                rep_ratios.append(1.0)
            else:
                rep_ratios.append(len(set(toks)) / len(toks))
        avg_rep = sum(rep_ratios) / max(1, len(rep_ratios))

        # 3. Duplicate rate
        n = len(captions)
        if n > 1:
            from collections import Counter
            counts = Counter(captions)
            dup_rate = sum(1 for c, cnt in counts.items() if cnt > 1) / n
        else:
            dup_rate = 0.0

        # 4. Self-BLEU: mean pairwise BLEU-1 between all caption pairs
        if n > 1:
            bleu_scores = []
            for i in range(n):
                for j in range(i + 1, n):
                    ref_words = set(words_per_caption[j])
                    hyp_words = words_per_caption[i]
                    if len(hyp_words) == 0:
                        bleu_scores.append(0.0)
                    else:
                        matches = sum(1 for w in hyp_words if w in ref_words)
                        bleu_scores.append(matches / len(hyp_words))
            self_bleu = sum(bleu_scores) / len(bleu_scores)
        else:
            self_bleu = 0.0

        return dict(
            avg_len=round(avg_len, 1),
            rep_ratio=round(avg_rep, 4),
            dup_rate=round(dup_rate, 4),
            self_bleu=round(self_bleu, 4),
        )

    def run_validation(step, epoch):
        """Run inference on val images, compute collapse metrics, log results."""
        print(f"\n[Val] Running validation (step={step}, epoch={epoch})...",
              flush=True)
        model.eval()
        captions = []
        t0 = time.time()

        for img_path in val_images:
            image = Image.open(img_path).convert("RGB")
            if is_internvl2:
                # InternVL2: expand <image> → <img><IMG_CONTEXT>×256</img>
                # BEFORE apply_chat_template (which returns token IDs, not text).
                img_tokens = f"<img>{'<IMG_CONTEXT>' * 256}</img>"
                conv = [{
                    "role": "user",
                    "content": f"{img_tokens}\n{PROMPT_A}",
                }]
                result = processor.apply_chat_template(
                    conv, add_generation_prompt=True)
                # InternVL's apply_chat_template returns BatchEncoding.
                # result["input_ids"] may return a BatchEncoding wrapper
                # (not raw data) in some AutoProcessor versions — use
                # .data[] for the raw list[int].
                if hasattr(result, "data") and "input_ids" in result.data:
                    ids_list = result.data["input_ids"]
                elif isinstance(result, dict) and "input_ids" in result:
                    ids_list = result["input_ids"]
                elif hasattr(result, "ids"):
                    ids_list = result.ids
                else:
                    ids_list = result
                # Normalize: may be nested [[ids]] or flat [ids]
                if isinstance(ids_list, list) and len(ids_list) > 0 and isinstance(ids_list[0], list):
                    ids_list = ids_list[0]
                input_ids = torch.tensor([ids_list], dtype=torch.long).to(config.device)
                attention_mask = torch.ones_like(input_ids)
                img_outputs = image_processor(
                    images=image, return_tensors="pt")
                pixel_values = img_outputs["pixel_values"].to(
                    config.device, torch.bfloat16)
                input_len = input_ids.shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        do_sample=False,
                    )
            elif is_phi35:
                # Phi-3.5: string content with <|image_1|> tag, processor
                # handles image+text via __call__. Chat template accessible
                # because we copied it from tokenizer to processor.
                #
                # The vendored modeling_phi3_v.py calls
                # past_key_values.get_max_length() which does not exist on
                # DynamicCache in transformers >= 4.49. Monkey-patch it.
                from transformers.cache_utils import DynamicCache
                if not hasattr(DynamicCache, "get_max_length"):
                    DynamicCache.get_max_length = lambda self: None
                conv = [{
                    "role": "user",
                    "content": f"<|image_1|>\n{PROMPT_A}",
                }]
                text_prompt = processor.apply_chat_template(
                    conv, add_generation_prompt=True)
                inputs = processor(
                    images=image, text=text_prompt, return_tensors="pt"
                ).to(config.device, torch.bfloat16)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=128, do_sample=False)
            else:
                conv = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT_A},
                    ],
                }]
                text_prompt = processor.apply_chat_template(
                    conv, add_generation_prompt=True)
                inputs = processor(
                    images=image, text=text_prompt, return_tensors="pt"
                ).to(config.device, torch.bfloat16)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=128, do_sample=False)

            # InternVL2.generate() passes inputs_embeds to the language model,
            # so the returned output_ids contain ONLY generated tokens (no input
            # prepended). Standard models return input_ids + generated tokens.
            if is_internvl2:
                generated = output_ids
            else:
                generated = output_ids[:, input_len:]
            # Decode: for InternVL2 processor IS the tokenizer
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            caption = tok.decode(
                generated[0], skip_special_tokens=True).strip()
            captions.append(caption)

        model.train()
        elapsed = time.time() - t0

        metrics = _compute_collapse_metrics(captions)
        _log({"event": "val", "step": step, "epoch": epoch, **metrics,
              "captions": captions, "elapsed_s": round(elapsed, 1)})

        if not args.no_swanlab:
            import swanlab
            swanlab.log(
                {f"val/{k}": v for k, v in metrics.items()}, step=step)

        # Warnings for red-flag metrics
        if metrics["avg_len"] > 50:
            print(f"  ⚠ WARNING: avg_len={metrics['avg_len']} (>50), "
                  f"possible mode collapse!", flush=True)
        if metrics["rep_ratio"] < 0.5:
            print(f"  ⚠ WARNING: rep_ratio={metrics['rep_ratio']} (<0.5), "
                  f"high repetition!", flush=True)
        if metrics["dup_rate"] > 0.3:
            print(f"  ⚠ WARNING: dup_rate={metrics['dup_rate']} (>0.3), "
                  f"low diversity!", flush=True)
        if metrics["self_bleu"] > 0.6:
            print(f"  ⚠ WARNING: self_bleu={metrics['self_bleu']} (>0.6), "
                  f"captions too similar!", flush=True)

        print(f"  avg_len={metrics['avg_len']} rep_ratio={metrics['rep_ratio']} "
              f"dup_rate={metrics['dup_rate']} self_bleu={metrics['self_bleu']} "
              f"({elapsed:.1f}s)", flush=True)

    val_images = _load_val_images(VAL_IMAGES_DIR, config.val_samples)
    print(f"[Val] {len(val_images)} images from {VAL_IMAGES_DIR}")

    # ── Training loop ──
    model.train()
    global_step = 0
    total_loss = 0.0
    print(f"\n[Train] {total_steps} steps, {warmup_steps} warmup")

    if config.val_steps > 0:
        run_validation(step=0, epoch=0)

    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}\n[Epoch] {epoch+1}/{config.num_epochs}\n{'='*50}")
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(
                config.device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(config.device)
            mm_token_type_ids = batch.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(config.device)
            image_sizes = batch.get("image_sizes")
            if image_sizes is not None:
                image_sizes = image_sizes.to(config.device)
            image_flags = batch.get("image_flags")
            if image_flags is not None:
                image_flags = image_flags.to(config.device)

            model_kwargs = dict(
                pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels,
            )
            if image_grid_thw is not None:
                model_kwargs["image_grid_thw"] = image_grid_thw
            if mm_token_type_ids is not None:
                model_kwargs["mm_token_type_ids"] = mm_token_type_ids
            if image_sizes is not None:
                model_kwargs["image_sizes"] = image_sizes
            if image_flags is not None:
                model_kwargs["image_flags"] = image_flags

            # InternVL2: PeftModelForCausalLM.forward() injects inputs_embeds
            # which InternVLChatModel.forward() does not accept. Route through
            # LoraModel directly (LoRA adapters are still active).
            if is_internvl2:
                outputs = model.base_model(**model_kwargs)
            else:
                outputs = model(**model_kwargs)
            loss = outputs.loss / config.gradient_accumulation_steps

            # NaN/Inf detection (safety net)
            if torch.isnan(loss) or torch.isinf(loss):
                msg = (f"Loss is NaN/Inf at global_step={global_step}, "
                       f"batch={step}. Aborting training.")
                print(f"\n[FATAL] {msg}", flush=True)
                raise RuntimeError(msg)

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
                    _log({"step": global_step, "total": total_steps,
                          "loss": round(avg, 6), "lr": lr,
                          "epoch": epoch + 1})
                    if not args.no_swanlab:
                        import swanlab
                        swanlab.log({"loss": avg, "lr": lr}, step=global_step)
                    total_loss = 0.0

                if global_step % config.save_steps == 0:
                    ckpt_dir = os.path.join(
                        config.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    _log({"event": "checkpoint", "step": global_step,
                          "epoch": epoch + 1})
                    print(f"  [Save] {ckpt_dir}")

                if config.val_steps > 0 and global_step % config.val_steps == 0:
                    run_validation(step=global_step, epoch=epoch + 1)

        avg_ep = epoch_loss / len(train_loader) * config.gradient_accumulation_steps
        _log({"event": "epoch_end", "epoch": epoch + 1,
              "avg_loss": round(avg_ep, 6), "step": global_step})
        print(f"[Epoch {epoch+1}] avg_loss={avg_ep:.4f}")

    # ── Final save ──
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    if not args.no_swanlab:
        import swanlab
        swanlab.finish()
    _log({"event": "done", "step": global_step, "output_dir": config.output_dir})
    print(f"\n[Done] -> {config.output_dir}")


if __name__ == "__main__":
    train()
