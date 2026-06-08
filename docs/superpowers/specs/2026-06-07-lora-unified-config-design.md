# LoRA Unified Config: Fair Comparison Design

**Date**: 2026-06-07
**Status**: Approved

## Context

Three VLMs (LLaVA-1.5-7B, Qwen3-VL-8B, InternVL3.5-8B) are fine-tuned via QLoRA on the UICO training set. The original experiments used inconsistent configurations:
- LLaVA: r=4, alpha=8, lr=5e-5, 7 target_modules, step=100 (early stop due to mode collapse at step 200-300)
- Qwen3VL: r=16, alpha=32, lr=2e-4, 4 target_modules, step=1000
- InternVL35: r=16, alpha=32, lr=2e-4, 4 target_modules, step=2000

The goal is to retrain all three with a unified configuration for fair comparison.

## Unified Configuration

| Parameter | Value |
|-----------|-------|
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning Rate | 2e-4 |
| Warmup Ratio | 0.1 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 1 |
| Batch Size × GradAccum | 1 × 8 |
| Max Grad Norm | 1.0 |
| Quantization | 4-bit NF4, bfloat16 compute |
| Checkpoint Selection | val loss minimum (community standard) |
| Validation | val_steps enabled; collapse metrics logged as auxiliary observation |

### Why 4 target modules (attention-only)

Qwen3VL and InternVL35 both use Qwen3 LLM backbone. Under QLoRA 4-bit quantization, 7 target modules (attention + MLP) cause NaN loss on Qwen backbones. Both models were trained stably with attention-only modules (q/k/v/o_proj). LLaVA's Vicuna-7B backbone is also LLaMA-derived; the same attention-only strategy is expected to prevent mode collapse observed in the original 7-module run.

### Why val loss minimum

Community-standard practice for QLoRA checkpoint selection. No composite score, no collapse-metric gating. If the lowest-val-loss checkpoint for LLaVA exhibits mode collapse, this is an intrinsic limitation of QLoRA on that backbone and a legitimate finding to discuss.

## Training Commands

```bash
# LLaVA
python scripts/train_lora.py --model llava --device cuda:X --val_steps 250 --no_swanlab

# Qwen3VL
python scripts/train_lora.py --model qwen3vl --device cuda:X --val_steps 250 --no_swanlab

# InternVL35
python scripts/train_lora.py --model internvl35 --device cuda:X --val_steps 250 --no_swanlab
```

## Code Changes

1. `config/training.py`: Update `llava` entry — change `target_modules` from 7 to 4
2. No other code changes needed; the training script already supports all three models via the adapter pattern

## Prerequisites

- transformers ≥ 5.9.0 in conda env (required for Qwen3VL support)
- bitsandbytes CUDA 13.x compatibility: set `LD_LIBRARY_PATH` to include nvidia/cu13/lib

## Experiment Scope

Only Prompt A inference after fine-tuning (no ZS/FS after LoRA — the "ZS after LoRA" concept doesn't apply since the model has been trained on task data, and FS+LoRA is a separate research question).

## Results Table Structure

```
             ZS    FS(k=1)  FS(k=3)  LoRA
LLaVA        x     x        x        x
Qwen3VL      x     x        x        x
InternVL35   x     x        x        x
```
