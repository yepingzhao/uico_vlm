"""TrainingRunner — orchestrates the QLoRA fine-tuning loop.

Model-agnostic training loop that delegates all model-specific decisions
to a TrainingModelAdapter.  The runner owns: SwanLab init, training log,
validation scheduling, gradient accumulation, checkpointing, and the
epoch/step iteration — everything that does NOT depend on the specific
VLM architecture.

Parallel to core/inference/runner.py:InferenceRunner on the inference side.
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

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


class TrainingRunner:
    """Orchestrates the QLoRA fine-tuning loop.

    Responsibilities:
      - Model loading (4-bit QLoRA base + LoRA adapters)
      - Training loop: epoch/step, forward/backward, gradient accumulation
      - NaN detection, logging, checkpoint save, validation scheduling
      - SwanLab integration
      - Final model + processor save

    All model-specific decisions (processor, forward routing, validation
    inference) are delegated to ``self.adapter``.
    """

    def __init__(
        self,
        model_name: str,
        adapter,  # TrainingModelAdapter
        args,      # argparse.Namespace
    ):
        self._model_name = model_name
        self._adapter = adapter
        self._args = args

        # Resolve model config from registry
        self._model_cfg = get_lora_config(model_name)

        # Build TrainingConfig
        cfg = TrainingConfig()
        cfg.model_id = self._model_cfg["model_id"]
        cfg.model_class_name = self._model_cfg["model_class_name"]
        cfg.processor_class_name = self._model_cfg["processor_class_name"]
        cfg.target_modules = self._model_cfg["target_modules"]
        cfg.output_dir = os.path.join(OUTPUT_DIR, f"{model_name}-lora")

        cfg.lora_r = args.lora_r
        cfg.lora_alpha = args.lora_alpha
        cfg.batch_size = args.batch_size
        cfg.gradient_accumulation_steps = args.grad_accum
        cfg.learning_rate = args.lr
        cfg.num_epochs = args.epochs
        cfg.max_samples = args.max_samples
        cfg.save_steps = args.save_steps
        cfg.logging_steps = args.logging_steps
        cfg.device = args.device
        cfg.seed = args.seed
        cfg.val_steps = args.val_steps
        cfg.val_max_samples = args.val_max_samples

        self._config = cfg
        self._processor = None
        self._image_processor = None

    # -- Public API ------------------------------------------------------

    def train(self) -> str:
        """Run full training loop.  Returns output_dir."""
        cfg = self._config
        args = self._args
        adapter = self._adapter
        model_cfg = self._model_cfg

        torch.manual_seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

        # ---- training log -----------------------------------------------
        log_path = os.path.join(cfg.output_dir, "training.log")

        def _log(entry: dict):
            entry.setdefault(
                "timestamp", datetime.now(timezone.utc).isoformat())
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
            if "loss" in entry:
                print(f"  step={entry['step']}/{entry['total']} "
                      f"loss={entry['loss']:.4f} lr={entry['lr']:.2e}",
                      flush=True)

        _log({"event": "start", "model": self._model_name,
              "model_id": cfg.model_id,
              "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
              "lr": cfg.learning_rate, "epochs": cfg.num_epochs,
              "target_modules": list(cfg.target_modules),
              "warmup_ratio": cfg.warmup_ratio})

        print(f"[Config] model={self._model_name} model_id={cfg.model_id}")
        print(f"[Config] LoRA r={cfg.lora_r} alpha={cfg.lora_alpha}")
        print(f"[Config] batch={cfg.batch_size}x{cfg.gradient_accumulation_steps}"
              f" lr={cfg.learning_rate} epochs={cfg.num_epochs}")

        # ---- load model ------------------------------------------------
        print(f"[Load] {cfg.model_id} (4-bit QLoRA) ...")
        t0 = time.time()
        model_class = _import_class(cfg.model_class_name)
        lora_config = make_lora_config(
            r=cfg.lora_r, alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
        )
        model = load_qlora_model(
            model_class, cfg.model_id, lora_config, cfg.device,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            model_kwargs=model_cfg.get("model_kwargs"))
        model.print_trainable_parameters()
        print(f"[Load] Done in {time.time() - t0:.1f}s")

        # ---- load processor (adapter) ----------------------------------
        proc = adapter.load_processor(cfg.model_id, model_cfg)
        processor = proc["processor"]
        image_processor = proc["image_processor"]
        self._processor = processor
        self._image_processor = image_processor

        # Post-load setup (InternVL img_context_token_id, etc.)
        adapter.setup_post_load(model, processor, image_processor,
                                cfg.device)

        # ---- dataset ---------------------------------------------------
        ds_kwargs = adapter.get_dataset_kwargs()
        train_ds = UICOInstructionDataset(
            ann_file=cfg.train_ann_file,
            processor=processor,
            max_samples=cfg.max_samples,
            seed=cfg.seed,
            **ds_kwargs,
        )
        if image_processor is not None:
            # InternVL: attach image_processor for dataset items
            train_ds.image_processor = image_processor

        def _collate(batch):
            return collate_fn(processor, batch)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=_collate,
        )

        # ---- optimizer + scheduler ------------------------------------
        total_steps = (
            len(train_loader) // cfg.gradient_accumulation_steps
            * cfg.num_epochs
        )
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ---- SwanLab ---------------------------------------------------
        if not args.no_swanlab:
            import swanlab
            swanlab.init(
                project=f"uico_vlm-{self._model_name}-lora",
                config={
                    "model": cfg.model_id, "method": "QLoRA",
                    "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
                    "batch_size": cfg.batch_size,
                    "grad_accum": cfg.gradient_accumulation_steps,
                    "learning_rate": cfg.learning_rate,
                    "num_epochs": cfg.num_epochs,
                    "total_steps": total_steps,
                    "train_images": len(train_ds),
                },
            )

        # ---- validation set ----------------------------------------------
        val_ds = UICOInstructionDataset(
            ann_file=cfg.val_ann_file,
            processor=processor,
            max_samples=cfg.val_max_samples,
            seed=cfg.seed,
            **ds_kwargs,
        )
        if image_processor is not None:
            val_ds.image_processor = image_processor
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=_collate,
        )
        print(f"[Val] {len(val_ds)} validation examples")

        # ---- training loop ---------------------------------------------
        model.train()

        # ---- best checkpoint tracking ------------------------------------
        self._best_val_loss = float("inf")
        self._best_step = None

        global_step = 0
        total_loss = 0.0
        print(f"\n[Train] {total_steps} steps, {warmup_steps} warmup")

        if cfg.val_steps > 0:
            val_loss = self._compute_val_loss(
                model, val_loader, adapter, cfg.device)
            print(f"  [Val] step=0 loss={val_loss:.4f}", flush=True)
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_step = global_step
                best_dir = os.path.join(cfg.output_dir, "best_checkpoint")
                model.save_pretrained(best_dir)
                _log({
                    "event": "best_checkpoint",
                    "step": global_step,
                    "epoch": 0,
                    "val_loss": round(val_loss, 4),
                })
                print(
                    f"  [Best] step={global_step} "
                    f"val_loss={val_loss:.4f} → {best_dir}",
                    flush=True,
                )

        for epoch in range(cfg.num_epochs):
            print(f"\n{'='*50}\n[Epoch] {epoch+1}/{cfg.num_epochs}\n"
                  f"{'='*50}")
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                model_kwargs = self._build_model_kwargs(batch, cfg.device)
                labels = batch["labels"].to(cfg.device)

                # Forward
                if adapter.use_base_model_forward:
                    outputs = model.base_model(
                        **model_kwargs, labels=labels)
                else:
                    outputs = model(**model_kwargs, labels=labels)
                loss = outputs.loss / cfg.gradient_accumulation_steps

                # NaN / Inf guard
                if torch.isnan(loss) or torch.isinf(loss):
                    msg = (
                        f"Loss is NaN/Inf at global_step={global_step}, "
                        f"batch={step}. Aborting training."
                    )
                    print(f"\n[FATAL] {msg}", flush=True)
                    raise RuntimeError(msg)

                loss.backward()
                total_loss += loss.item()
                epoch_loss += loss.item()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        avg = total_loss / cfg.logging_steps
                        lr = scheduler.get_last_lr()[0]
                        _log({
                            "step": global_step, "total": total_steps,
                            "loss": round(avg, 6), "lr": lr,
                            "epoch": epoch + 1,
                        })
                        if not args.no_swanlab:
                            import swanlab as _sl
                            _sl.log({"loss": avg, "lr": lr},
                                    step=global_step)
                        total_loss = 0.0

                    if global_step % cfg.save_steps == 0:
                        ckpt_dir = os.path.join(
                            cfg.output_dir,
                            f"checkpoint-{global_step}")
                        model.save_pretrained(ckpt_dir)
                        _log({"event": "checkpoint",
                              "step": global_step, "epoch": epoch + 1})
                        print(f"  [Save] {ckpt_dir}")

                    if (cfg.val_steps > 0
                            and global_step % cfg.val_steps == 0):
                        val_loss = self._compute_val_loss(
                            model, val_loader, adapter, cfg.device)
                        print(
                            f"  [Val] step={global_step} "
                            f"loss={val_loss:.4f}",
                            flush=True,
                        )
                        if val_loss < self._best_val_loss:
                            self._best_val_loss = val_loss
                            self._best_step = global_step
                            best_dir = os.path.join(
                                cfg.output_dir, "best_checkpoint")
                            model.save_pretrained(best_dir)
                            _log({
                                "event": "best_checkpoint",
                                "step": global_step,
                                "epoch": epoch + 1,
                                "val_loss": round(val_loss, 4),
                            })
                            print(
                                f"  [Best] step={global_step} "
                                f"val_loss={val_loss:.4f} → {best_dir}",
                                flush=True,
                            )

            avg_ep = (
                epoch_loss / len(train_loader)
                * cfg.gradient_accumulation_steps)
            _log({"event": "epoch_end", "epoch": epoch + 1,
                  "avg_loss": round(avg_ep, 6), "step": global_step})
            print(f"[Epoch {epoch+1}] avg_loss={avg_ep:.4f}")

        # ---- final save ------------------------------------------------
        model.save_pretrained(cfg.output_dir)
        processor.save_pretrained(cfg.output_dir)
        if not args.no_swanlab:
            import swanlab as _sl
            _sl.finish()
        _log({"event": "done", "step": global_step,
              "output_dir": cfg.output_dir})
        print(f"\n[Done] -> {cfg.output_dir}")

        # ---- best checkpoint summary ----------------------------------
        if self._best_step is not None:
            _log({
                "event": "best_checkpoint_final",
                "best_step": self._best_step,
                "best_val_loss": round(self._best_val_loss, 4),
            })
            print(
                f"[Best] Final best: step={self._best_step} "
                f"val_loss={self._best_val_loss:.4f}",
                flush=True,
            )
        else:
            print("[Best] No validation runs — using final checkpoint.",
                  flush=True)

        return cfg.output_dir

    # -- Internal helpers ------------------------------------------------

    @staticmethod
    def _build_model_kwargs(batch: dict, device: str) -> dict:
        """Build forward kwargs from batch dict.

        Model-agnostic pass-through: every tensor key the dataset emitted
        is forwarded to the model.  ``pixel_values`` is cast to bfloat16
        (standard for QLoRA mixed-precision); other tensors stay at their
        native dtype.
        """
        kwargs = {}
        for key, value in batch.items():
            if key == "pixel_values":
                kwargs[key] = value.to(device, dtype=torch.bfloat16)
            elif key not in ("labels",):
                kwargs[key] = value.to(device)
        return kwargs

    def _compute_val_loss(self, model, val_loader, adapter,
                          device: str) -> float:
        """Compute average validation loss on the held-out val set.

        Returns the mean masked-LM loss across all val batches.
        Lower is better — used as the best-checkpoint criterion.
        """
        model.eval()
        total = 0.0
        count = 0
        for batch in val_loader:
            model_kwargs = self._build_model_kwargs(batch, device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                if adapter.use_base_model_forward:
                    outputs = model.base_model(
                        **model_kwargs, labels=labels)
                else:
                    outputs = model(**model_kwargs, labels=labels)
            total += outputs.loss.item()
            count += 1
            # Guard against NaN/Inf — QLoRA 4-bit has known numerical
            # instability risk (documented in project memory).
            if math.isnan(total) or math.isinf(total):
                print(
                    f"  ⚠ WARNING: val loss NaN/Inf at batch {count}, "
                    f"skipping remaining val batches",
                    flush=True,
                )
                model.train()
                return float("inf")
        model.train()
        return total / max(1, count)
