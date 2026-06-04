# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM evaluation benchmark for urban incivility captioning on the UICO dataset (CCMC images in COCO format). Evaluates 15+ vision-language models zero-shot, few-shot, and fine-tuned via QLoRA on the task of describing civic norm violations in urban scenes.

**Paper:** TCSVT 2026 submission comparing VLM baselines against specialized captioning models (Transformer, AoANet, UIFormer, etc.).

## Environment

```bash
conda env create -f environment.yml   # creates env named "uico_vlm"
conda activate uico_vlm
```

Key dependencies: `torch==2.5.1`, `transformers>=4.46`, `vllm`, `pycocotools`, `coco-caption` (from GitHub).

## Package Import Issue

The code imports itself as `vlm_eval` (e.g. `from vlm_eval.config import ...`) but the repo directory is named `uico_vlm`. The run scripts work around this by inserting the workspace root into `sys.path`. For imports to work, create a symlink in the workspace root:

```bash
ln -s /home/uesr/zhaoyeping/workspace-code/uico_vlm /home/uesr/zhaoyeping/workspace-code/vlm_eval
```

Without this, `from vlm_eval.xxx` imports will fail.

## Data

Data lives at `/home/uesr/zhao/media_data/ccmc/` (configured in `config.py:DATA_BASE`):
- `annotations/captions_train.json`, `captions_test.json` — COCO-format with 5 reference captions per image
- `images/ccmc_train/`, `images/ccmc_test/`, `images/ccmc_val/` — image files split by split prefix

## Architecture

### Model Wrappers (`models/`)

All models implement the `VLMWrapper` ABC (`models/base.py`):
- `load(device)` — load model + processor
- `generate(image_path, prompt, **kwargs) -> str` — zero-shot single-image captioning

Models with few-shot support additionally implement `generate_fewshot(test_image_path, prompt_template, example_images, example_captions, **kwargs)`.

Model registry is defined in `config.py:MODEL_REGISTRY` as `(short_name, hf_model_id, wrapper_class_name)` tuples. Runtime model lookup happens in `models/__init__.py:get_wrapper()` which maps short names to wrapper instances, with graceful handling for models requiring unavailable dependencies (DeepSeek-VL2, vLLM).

**Notable wrapper specifics:**
- **LLaVA** and **Qwen2VL** are the primary models — they support both HF and vLLM backends, plus few-shot `generate_fewshot()` with multi-image inputs
- **InternVL2/InternVL2.5** need monkey-patching for transformers ≥5.x compatibility and use the model's built-in `chat()` API rather than the standard processor pipeline
- **Qwen2VL** uses two processor instances: high-res for zero-shot, low-res for few-shot multi-image (avoids OOM)
- **vLLM wrappers** (`vllm_wrapper.py`) use `LLM.chat()` for batched offline inference as an alternative to HF `model.generate()`

### Data Loading (`data/dataset.py`)

`UICOTestDataset` wraps pycocotools COCO API:
- `image_ids` — sorted list of COCO image IDs (filterable for dev subsampling)
- `get_image_path(image_id)` — resolves COCO filenames to filesystem paths
- `get_references(image_id)` — returns all 5 reference captions
- `subsample(n, seed)` — returns a new dataset with a random subset (dev workflow)
- `all_references_dict()` — bulk reference extraction for evaluation

Image path resolution maps filename prefixes (`CCMC_train_*`, `CCMC_test_*`, `CCMC_val_*`) to subdirectories.

### Prompts (`prompts/templates.py`)

Four prompt variants:
- **A** (primary): concise description request
- **B**: structured 3-part analysis
- **C**: governance/urban-management persona
- **ZH**: Chinese prompt for Qwen models

Prompt A is used for all models. B/C are sensitivity analysis (LLaVA + Qwen only). ZH is supplemental for Chinese-optimized Qwen.

### Evaluation (`eval/`)

Two evaluation modes:
- **Ref-based** (`ref_based.py`): BLEU-1/4, METEOR, ROUGE-L, CIDEr-D, SPICE via `pycocoevalcap`. Also computes S_m composite score (Eq. 1 in paper: mean of B@4, M, R, C, S). Captions truncated to 50 words before SPICE to prevent Stanford parser OOM.
- **Ref-free** (`ref_free.py`): CLIPScore (cosine similarity of image/text CLIP embeddings) and RefCLIPScore (harmonic mean of CLIPScore and max reference CLIPScore). Uses `openai/clip-vit-large-patch14`.

### Few-Shot (`fewshot/`)

`fewshot/sampler.py` samples k diverse examples from the training set (one caption per image, fixed seed, cached to disk). Examples are pre-sampled once and reused across all test images.

`scripts/run_fewshot.py` uses the same checkpoint/resume pattern as `scripts/run_inference.py` but calls `wrapper.generate_fewshot()` which interleaves example images+captions before the test image.

### Training (`scripts/train_lora.py`)

QLoRA fine-tuning with multi-model support (`scripts/train_lora.py`):
- 4-bit NF4 quantization, LoRA rank=8 on q/k/v/o projections
- Vision encoder frozen, multimodal projector quantized (4-bit) but not LoRA-tuned
- Masked LM loss on caption tokens only (image tokens and user prompt masked with -100)
- 1 epoch, batch_size=1 with grad_accum=8, cosine schedule with warmup
- SwanLab logging

### Table Generation (`make_table.py`)

Generates LaTeX comparison tables from `outputs/all_metrics.json`, comparing VLM results against paper baselines (Transformer, SCST, AoANet, UIFormer, etc.) with bold-for-best formatting.

## Common Commands

```bash
# Download models (uses hf-mirror.com by default)
python download_models.py --dry-run     # list models
python download_models.py --model llava # single model

# Zero-shot inference
python scripts/run_inference.py --models blip2 llava --subsample 1000 --prompt A
python scripts/run_inference.py --models llava qwen2vl --prompt B          # sensitivity
python scripts/run_inference.py --models llava-vllm qwen2vl-vllm --prompt A # vLLM backend

# Evaluation (reads .jsonl predictions, writes metrics.json)
python scripts/run_eval.py --model blip2 --prompt A
python scripts/run_eval.py --model blip2 --prompt A --ref_free_only
python scripts/run_eval.py --all

# Few-shot
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5 --subsample 500
python scripts/eval_fewshot.py --model llava --k 1
python scripts/eval_fewshot.py --all

# Training
python scripts/train_lora.py --model llava                   # default LLaVA-1.5-7B
python scripts/train_lora.py --model llava-next --epochs 3   # LLaVA-NeXT

# LoRA inference
python scripts/inference_lora.py --model llava

# Table generation
python make_table.py
```

## Output Structure

```
outputs/
  {model_name}/
    predictions_prompt_{a,b,c,zh}.jsonl     # zero-shot
    predictions_fewshot_k{1,3,5}.jsonl      # few-shot
    metrics_prompt_{a,b,c,zh}.json           # per-prompt eval
  all_metrics.json                           # aggregate zero-shot results
  fewshot_all_metrics.json                   # aggregate few-shot results
  fewshot_cache/                             # cached few-shot example selections
  {model}-lora/                              # QLoRA adapter weights
  tables/                                    # generated LaTeX tables
```

## Key Config (`config.py`)

- `DEV_MODELS = ["blip2", "llava"]` — lightweight models for fast dev iteration
- `SENSITIVITY_MODELS = ["llava", "qwen2vl"]` — models used for prompt B/C sensitivity
- `DEV_SAMPLE_SIZE = 1000` — subsample size for Phase 1 dev
- `MAX_NEW_TOKENS = 128` — generation budget for all models
- `RANDOM_SEED = 42` — used everywhere for reproducibility

## Development Notes

- Inference scripts support checkpoint/resume: they track processed `image_id`s in the output `.jsonl`, so interrupted runs pick up where they left off
- All scripts accept `--subsample` for fast dev iteration on a subset of images
- Most model wrappers need ~16-24GB VRAM; vLLM backends can trade memory for speed
- Data paths in `config.py` are absolute and machine-specific — update `DATA_BASE` for different environments
