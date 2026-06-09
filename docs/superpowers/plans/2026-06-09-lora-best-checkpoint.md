# LoRA Best Checkpoint Auto-Save & Auto-Select Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During QLoRA training, automatically track and save the best checkpoint (by composite validation score); inference after training auto-selects the best checkpoint.

**Architecture:** The training runner already runs validation and computes collapse metrics. We add a composite `val_score` from those metrics, track the best-scoring step, and save a copy to `best_checkpoint/`. The `run_lora.py` inference phase then auto-detects and prefers `best_checkpoint/` over the final checkpoint. Two files touched, no new dependencies.

**Tech Stack:** Python 3.10, PyTorch, PEFT, existing codebase

---

## Files to Change

| File | Action | Why |
|---|---|---|
| `core/training/runner.py` | MODIFY | Add best-score tracking, best checkpoint saving |
| `scripts/run_lora.py` | MODIFY | Auto-detect best checkpoint for inference |

---

### Task 1: Add best-checkpoint tracking and saving to TrainingRunner

**Files:**
- Modify: `core/training/runner.py`

- [ ] **Step 1: Initialize best-score tracking in `train()`**

After line 213 (`model.train()`), before the training loop begins, add tracking variables:

```python
# ---- best checkpoint tracking ------------------------------------
self._best_val_score = float("-inf")
self._best_step = None
```

Insert at `core/training/runner.py:213` — right after `model.train()` and right before `global_step = 0`.

- [ ] **Step 2: Compute val_score after validation and save best checkpoint**

Modify the validation call site at lines 285-291. Currently:

```python
                    if (cfg.val_steps > 0
                            and global_step % cfg.val_steps == 0):
                        self._run_validation(
                            model, processor, image_processor,
                            val_images, global_step, epoch=epoch + 1,
                            _log=_log,
                        )
```

Change `_run_validation` to return the metrics dict so the caller can compute the score. First, modify `_run_validation` to `return metrics` at the end (after line 441). Then change the call site to:

```python
                    if (cfg.val_steps > 0
                            and global_step % cfg.val_steps == 0):
                        metrics = self._run_validation(
                            model, processor, image_processor,
                            val_images, global_step, epoch=epoch + 1,
                            _log=_log,
                        )
                        val_score = (
                            metrics["rep_ratio"]
                            - metrics["self_bleu"]
                            - metrics["dup_rate"]
                        )
                        if val_score > self._best_val_score:
                            self._best_val_score = val_score
                            self._best_step = global_step
                            best_dir = os.path.join(
                                cfg.output_dir, "best_checkpoint")
                            model.save_pretrained(best_dir)
                            _log({
                                "event": "best_checkpoint",
                                "step": global_step,
                                "epoch": epoch + 1,
                                "val_score": round(val_score, 4),
                                "rep_ratio": metrics["rep_ratio"],
                                "self_bleu": metrics["self_bleu"],
                                "dup_rate": metrics["dup_rate"],
                            })
                            print(
                                f"  [Best] step={global_step} "
                                f"score={val_score:.4f} → {best_dir}",
                                flush=True,
                            )
```

- [ ] **Step 3: Make `_run_validation` return the metrics dict**

At the end of `_run_validation` (after the `print()` at line 441), add:

```python
        return metrics
```

- [ ] **Step 4: Handle initial validation (step 0) — same pattern**

The initial validation at lines 218-222 also needs updating. Currently:

```python
        if cfg.val_steps > 0:
            self._run_validation(
                model, processor, image_processor, val_images,
                global_step, epoch=0, _log=_log,
            )
```

Change to:

```python
        if cfg.val_steps > 0:
            metrics = self._run_validation(
                model, processor, image_processor, val_images,
                global_step, epoch=0, _log=_log,
            )
            val_score = (
                metrics["rep_ratio"]
                - metrics["self_bleu"]
                - metrics["dup_rate"]
            )
            if val_score > self._best_val_score:
                self._best_val_score = val_score
                self._best_step = global_step
                best_dir = os.path.join(cfg.output_dir, "best_checkpoint")
                model.save_pretrained(best_dir)
                _log({
                    "event": "best_checkpoint",
                    "step": global_step,
                    "epoch": 0,
                    "val_score": round(val_score, 4),
                    "rep_ratio": metrics["rep_ratio"],
                    "self_bleu": metrics["self_bleu"],
                    "dup_rate": metrics["dup_rate"],
                })
                print(
                    f"  [Best] step={global_step} "
                    f"score={val_score:.4f} → {best_dir}",
                    flush=True,
                )
```

- [ ] **Step 5: Log final best checkpoint info at end of training**

After the final save block (after line 302, before the `return`), add a summary log entry:

```python
        # ---- best checkpoint summary ----------------------------------
        if self._best_step is not None:
            _log({
                "event": "best_checkpoint_final",
                "best_step": self._best_step,
                "best_val_score": round(self._best_val_score, 4),
            })
            print(
                f"[Best] Final best: step={self._best_step} "
                f"score={self._best_val_score:.4f}",
                flush=True,
            )
        else:
            print("[Best] No validation runs — using final checkpoint.",
                  flush=True)
```

- [ ] **Step 6: Verify training changes are syntactically correct**

```bash
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm && python -c "from core.training.runner import TrainingRunner; print('OK')"
```

Expected: prints `OK` with no errors.

---

### Task 2: Auto-select best checkpoint in run_lora.py inference phase

**Files:**
- Modify: `scripts/run_lora.py`

- [ ] **Step 1: Add best-checkpoint resolution logic**

In `run_lora.py`, after training completes (line 107 `runner.train()`) and before the inference section, add best-checkpoint resolution. Insert a block between line 108 and line 114 (the `if do_inference:` line):

```python
    # ── Resolve best checkpoint for inference ────────────────────────
    best_ckpt_dir = os.path.join(lora_dir, "best_checkpoint")
    best_adapter = os.path.join(best_ckpt_dir, "adapter_model.safetensors")
    if do_inference and os.path.isfile(best_adapter):
        print(f"[Best] Using best checkpoint: {best_ckpt_dir}")
        lora_dir = best_ckpt_dir
    elif do_inference and os.path.isdir(best_ckpt_dir):
        print(f"[Best] best_checkpoint/ exists but no adapter found, "
              f"using final: {lora_dir}")
```

This handles three cases:
1. `best_checkpoint/adapter_model.safetensors` exists → use best checkpoint
2. `best_checkpoint/` exists but has no adapter (corrupted) → fall back to final
3. No `best_checkpoint/` at all (val was never run, or old training run) → fall back to final

- [ ] **Step 2: Verify the full script parses correctly**

```bash
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm && python -c "import scripts.run_lora; print('OK')"
```

Expected: prints `OK` with no import errors.

---

### Task 3: End-to-end dry-run verification

- [ ] **Step 1: Verify that existing training runs (without best_checkpoint/) still work**

```bash
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm && python scripts/run_lora.py --model llava --inference_only --subsample 5
```

Expected: runs inference normally, uses the existing final checkpoint (no `best_checkpoint/` found).

- [ ] **Step 2: Review the flow with a manual check of the training log format**

```bash
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm && grep "best_checkpoint" core/training/runner.py
```

Expected: finds the log event strings added in Task 1.

- [ ] **Step 3: Commit**

```bash
git add core/training/runner.py scripts/run_lora.py
git commit -m "feat: auto-save best LoRA checkpoint and auto-select for inference

Training runner now tracks a composite val_score (rep_ratio - self_bleu - dup_rate)
during mid-training validation and saves the best-scoring checkpoint to
outputs/<model>-lora/best_checkpoint/.  run_lora.py inference phase auto-detects
and prefers best_checkpoint/ when present.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Validation

```bash
# Unit-level: both modules import cleanly
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm
python -c "from core.training.runner import TrainingRunner; print('OK')"
python -c "import scripts.run_lora; print('OK')"

# Integration: inference-only mode still works with existing checkpoints
python scripts/run_lora.py --model llava --inference_only --subsample 5

# Integration: full train+inference with best checkpoint
# (requires GPU, ~1-2 hours for a full run — verify best_checkpoint/ is created)
# python scripts/run_lora.py --model llava --subsample 50 --val_steps 20 --save_steps 100
# ls outputs/llava-lora/best_checkpoint/adapter_model.safetensors
```

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `best_checkpoint/` without processor files causes InternVL `load_lora` to fail | Low | InternVL `load_lora` already has fallback to `model_id` for processor; tested in existing code |
| Validation runs at step 0 produce noisy initial score | Low | Acceptable — if later checkpoints are worse, step 0 stays "best"; training should improve from step 0 |
| Composite score doesn't correlate with actual quality | Medium | Score is simple and interpretable (higher diversity = better); can be refined later by adjusting formula |

## Acceptance

- [ ] Training saves `best_checkpoint/` when validation detects improved score
- [ ] Inference auto-uses best checkpoint when available
- [ ] Falls back to final checkpoint when no best checkpoint exists
- [ ] Training log records best_checkpoint events
- [ ] Existing inference-only flow unchanged
