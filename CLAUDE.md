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

Key dependencies: Python 3.10, `torch==2.5.1`, `transformers>=4.46`, `vllm`, `pycocotools`, `pycocoevalcap`.

## Package Import Issue

The code imports itself as `vlm_eval` (e.g. `from vlm_eval.config import ...`) but the repo directory is named `uico_vlm`. The run scripts work around this by inserting the workspace root into `sys.path`. For imports to work, create a symlink in the workspace root:

```bash
ln -s /home/uesr/zhaoyeping/workspace-code/uico_vlm /home/uesr/zhaoyeping/workspace-code/vlm_eval
```

Without this, `from vlm_eval.xxx` imports will fail.

## Data

Data lives at `/home/uesr/zhao/media_data/ccmc/` (configured in `config/__init__.py:DATA_BASE`):
- `annotations/captions_train.json`, `captions_test.json` — COCO-format with 5 reference captions per image
- `images/ccmc_train/`, `images/ccmc_test/`, `images/ccmc_val/` — image files split by split prefix

## Architecture

### Config (`config/`)

Central configuration lives in `config/__init__.py`:
- `MODEL_REGISTRY` — list of `(short_name, hf_model_id, wrapper_class_name)` tuples
- `DATA_BASE`, `OUTPUT_DIR`, `MAX_NEW_TOKENS`, `RANDOM_SEED`, etc.

LoRA training configs live in `config/training.py` (`TrainingConfig` dataclass + `MODEL_LORA_CONFIGS` dict). Currently supports llava, llava-next, and qwen2vl, each with model-specific `target_modules`.

### Model Wrappers (`models/`)

All models implement the `VLMWrapper` ABC (`models/base.py`):
- `load(device)` — load model + processor
- `generate(image_path, prompt, **kwargs) -> str` — zero-shot single-image captioning
- `model_name` (property) — short identifier used for output directory naming
- `_strip_and_decode(output_ids, inputs)` — shared helper: strips input tokens from output, decodes

Few-shot support follows a Template Method pattern on the base class:
- `supports_fewshot` (property) — True if the wrapper overrides `_build_fewshot_inputs`
- `generate_fewshot(test_image_path, prompt_template, example_images, example_captions, **kwargs)` — orchestrated by the base class, delegates to `_build_fewshot_inputs` and `_get_fewshot_processor`
- `_build_fewshot_inputs(content_blocks, all_images)` — model-specific tokenization (LLaVA vs Qwen APIs differ)
- `_get_fewshot_processor()` — optionally returns a different processor (e.g. low-res for Qwen2VL to avoid OOM)
- Content blocks are built by `fewshot/content.py:build_fewshot_images_and_content()` which interleaves example images+captions before the test image, with a mode flag `embed_images` (True for Qwen-style inline PIL, False for LLaVA-style placeholders)

Runtime model lookup: `models/__init__.py:get_wrapper()` maps short names to wrapper instances, with graceful handling for models requiring unavailable dependencies (DeepSeek-VL2 requires `deepseek_vl2` package; vLLM models require `vllm`; Idefics3 may fail on older transformers).

**Notable wrapper specifics:**
- **LLaVA** and **Qwen2VL** are the primary models — they support both HF and vLLM backends, plus few-shot
- **InternVL2/InternVL2.5** share a base class (`models/_internvl_base.py`) for monkey-patching transformers ≥5.x compatibility; they use the model's built-in `chat()` API rather than the standard processor pipeline. Their chat implementation is vendored in `.internvl2_pkg/internvl2_chat/`
- **Qwen2VL** (`qwen2vl` short name maps to `Qwen/Qwen2.5-VL-7B-Instruct`) uses two processor instances: high-res (`min_pixels=256*28*28, max_pixels=1280*28*28`) for zero-shot, low-res (`min_pixels=128*28*28, max_pixels=256*28*28`) for few-shot (avoids OOM on 24GB)
- **Qwen3VL** wrapper exists in `models/__init__.py:get_wrapper()` but is not yet in `MODEL_REGISTRY`
- **vLLM wrappers** (`models/vllm_wrapper.py`) use `LLM.chat()` for batched offline inference as an alternative to HF `model.generate()`

**Shared QLoRA utilities** (`models/lora.py`):
- `make_lora_config(r, alpha, dropout, target_modules)` — build PEFT LoraConfig
- `load_qlora_model(model_class, model_id, lora_config, device)` — load 4-bit quantized base + LoRA for training
- `load_qlora_for_inference(model_class, model_id, lora_dir, device)` — load 4-bit base + trained adapters for inference

### Data Loading (`data/`)

`data/dataset.py` — `UICOTestDataset` wraps pycocotools COCO API for evaluation:
- `image_ids` — sorted list of COCO image IDs (filterable for dev subsampling)
- `get_image_path(image_id)` — resolves COCO filenames to filesystem paths via `resolve_image_path()` which maps filename prefixes (`CCMC_train_*`, `CCMC_test_*`, `CCMC_val_*`) to subdirectories
- `get_references(image_id)` — returns all 5 reference captions
- `subsample(n, seed)` — returns a new dataset with a random subset (dev workflow)
- `all_references_dict()` — bulk reference extraction for evaluation
- `load_test_dataset(subsample, seed)` — convenience factory

`data/training_dataset.py` — `UICOInstructionDataset` for QLoRA fine-tuning:
- One randomly-selected caption per image (fixed seed for reproducibility)
- Uses `processor.apply_chat_template()` to build instruction-formatted conversations
- **Masking strategy (critical):** labels before the assistant turn are set to -100. The user-prefix length is determined by tokenizing the user-only conversation WITH the image (not just the text), ensuring image placeholder expansion is consistent between user-only and full sequences. Image-specific token IDs (e.g. LLaVA's `<image>`, Qwen's `<|vision_pad|>/<|image_pad|>`) are also masked to -100 via auto-detection in `_detect_image_token_ids()`
- `collate_fn(processor, batch)` — pads variable-length sequences, handles `mm_token_type_ids` (Qwen2.5-VL MRoPE), stacks `pixel_values` and `image_grid_thw`

### Prompts (`prompts/templates.py`)

Three prompt variants (Prompt ZH removed in `e86a5d4`):
- **A** (primary): open-ended what+where description — used for all models
- **B**: same content as A, structured format (`Violation:` / `Location:` fields)
- **C**: same format as A, adds "why" justification

Prompt design is grounded in GT caption analysis (median 9 words, what+where structure). Each pair (A→B, A→C) varies a single dimension for clean attribution. See `docs/research-notes/2026-06-04-prompt-gt-alignment-analysis.md` for rationale.

**Important:** B and C are evaluated with **ref-free metrics only** (CLIPScore, RefCLIPScore). Their format/content differences mechanically deflate n-gram metrics, so comparing A/B/C via ref-based metrics would conflate format noise with sensitivity signal.

Few-shot prompt (`PROMPT_FEWSHOT`) is aligned with Prompt A style.

### Evaluation (`eval/`)

Two evaluation modes:
- **Ref-based** (`ref_based.py`): BLEU-1/4, METEOR, ROUGE-L, CIDEr-D, SPICE via `pycocoevalcap`. Also computes S_m composite score (Eq. 1 in paper: mean of B@4, M, R, C, S). Captions truncated to 50 words before SPICE to prevent Stanford parser OOM.
- **Ref-free** (`ref_free.py`): `CLIPScorer` class computes CLIPScore (cosine similarity of image/text CLIP embeddings) and RefCLIPScore (harmonic mean of CLIPScore and max reference CLIPScore). Uses `openai/clip-vit-large-patch14`. Loaded with `local_files_only=True` (offline mode).

### Few-Shot (`fewshot/`)

`fewshot/sampler.py` — samples k diverse examples from the training set:
- Groups captions by image, picks one random caption per image (fixed seed)
- Cached to disk at `outputs/fewshot_cache/fewshot_examples_k{k}_seed{seed}.json`
- Examples are pre-sampled once and reused across all test images

`fewshot/content.py` — shared content-block construction for few-shot inference: `build_fewshot_images_and_content()` interleaves example images+captions before the test image. Supports two modes: Qwen-style (embed PIL objects in content blocks) and LLaVA-style (image placeholders with separate image list).

`scripts/run_fewshot.py` uses the same checkpoint/resume pattern as `scripts/run_inference.py` but calls `wrapper.generate_fewshot()`. Few-shot-capable models are auto-discovered by checking `wrapper.supports_fewshot`.

### Training (`scripts/train_lora.py`)

QLoRA fine-tuning with multi-model support:
- 4-bit NF4 quantization with double quantization, bfloat16 compute dtype
- LoRA defaults: r=16, alpha=32, dropout=0.05 on model-specific target_modules
- Vision encoder frozen, multimodal projector quantized (4-bit) but not LoRA-tuned
- Masked LM loss on caption tokens only (image tokens and user prompt masked with -100)
- Defaults: 2 epochs, batch_size=1, grad_accum=8, lr=2e-4, cosine schedule with 10% warmup
- NaN/Inf loss detection aborts training early (safety net for numerical instability)
- Structured JSONL training log (`outputs/{model}-lora/training.log`) for cross-session agent sync — each line is a JSON object with timestamp, event type, step, loss, lr
- SwanLab logging (disable with `--no_swanlab`)
- Model-specific configs in `config/training.py:MODEL_LORA_CONFIGS` (llava, llava-next, qwen2vl)
- Qwen2.5-VL uses limited target_modules (q/k/v/o only, 4 total) due to NaN with 7 modules under QLoRA 4-bit
- Qwen2.5-VL training uses low-res processor (`min_pixels=128*28*28, max_pixels=256*28*28`) to fit 24GB VRAM

### Table Generation (`make_table.py`)

**Note:** This script is referenced in the paper workflow but does not yet exist in the repository. It is planned to generate LaTeX comparison tables from `outputs/zeroshot_all_metrics.json`.

### Research Notes (`docs/research-notes/`)

Contains design rationale documents:
- `2026-06-04-prompt-gt-alignment-analysis.md` — prompt design grounded in GT caption analysis
- `2026-06-04-vlm-baseline-plan.md` — VLM baseline comparison plan
- `2026-06-05-lora-nan-analysis.md` — root-cause analysis of NaN loss during QLoRA training

### Vendored Code (`.internvl2_pkg/`)

`internvl2_chat/` — vendored copy of the InternVL2 chat model implementation (modeling, tokenization, configuration, conversation templates) used by InternVL2 and InternVL2.5 wrappers.

## Common Commands

```bash
# Download models (uses hf-mirror.com by default)
python download_models.py --dry-run     # list models
python download_models.py --model llava # single model

# Zero-shot inference (--overwrite re-generates existing predictions)
python scripts/run_inference.py --models blip2 llava --subsample 1000 --prompt A
python scripts/run_inference.py --models llava qwen2vl --prompt B          # sensitivity (ref-free eval only)
python scripts/run_inference.py --models llava-vllm qwen2vl-vllm --prompt A # vLLM backend
python scripts/run_inference.py --models llava --prompt A --overwrite       # re-run, discard old predictions

# Prompt B format compliance checker
python scripts/check_format_compliance.py outputs/llava/predictions_prompt_b.jsonl

# Evaluation (reads .jsonl predictions, writes metrics.json)
python scripts/run_eval.py --model blip2 --prompt A
python scripts/run_eval.py --model blip2 --prompt A --ref_free_only   # skip n-gram metrics
python scripts/run_eval.py --all                                      # all models/prompts (B/C = ref-free only)

# Few-shot
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5 --subsample 500
python scripts/eval_fewshot.py --model llava --k 1
python scripts/eval_fewshot.py --all

# Training (default: 2 epochs, lora_r=16, lora_alpha=32, lr=2e-4, batch=1×8)
python scripts/train_lora.py --model llava
python scripts/train_lora.py --model llava-next --epochs 3 --lora_r 8
python scripts/train_lora.py --model qwen2vl --max_samples 500 --no_swanlab   # quick test
python scripts/train_lora.py --model qwen2vl --lr 1e-4 --epochs 5             # custom hparams

# LoRA inference
python scripts/inference_lora.py --model llava

# Table generation (TBD — script not yet created)
# python make_table.py
```

## Output Structure

```
outputs/
  {model_name}/
    predictions_prompt_{a,b,c}.jsonl          # zero-shot
    predictions_fewshot_k{1,3,5}.jsonl        # few-shot
    metrics_prompt_{a,b,c}.json               # per-prompt eval
    metrics_fewshot_k{1,3,5}.json             # per-k few-shot eval
  zeroshot_all_metrics.json                    # aggregate zero-shot results
  fewshot_all_metrics.json                    # aggregate few-shot results
  fewshot_cache/                              # cached few-shot example selections
  {model}-lora/                               # QLoRA adapter weights
    training.log                              # structured JSONL training log
  tables/                                     # generated LaTeX tables (TBD)
```

## Key Config (`config/__init__.py`)

- `SENSITIVITY_MODELS = ["llava", "qwen2vl"]` — models used for prompt B/C sensitivity
- `MAX_NEW_TOKENS = 128` — generation budget for all models
- `RANDOM_SEED = 42` — used everywhere for reproducibility
- `CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"` — CLIP model for ref-free eval
- `VLLM_GPU_MEMORY_UTILIZATION = 0.9` — vLLM memory fraction
- `VLLM_MAX_MODEL_LEN = 2048`, `VLLM_MAX_NUM_SEQS = 1`, `VLLM_ENFORCE_EAGER = True` — vLLM conservative settings

## Development Notes

- Inference scripts support checkpoint/resume: they track processed `image_id`s in the output `.jsonl`, so interrupted runs pick up where they left off
- Use `--overwrite` to ignore existing checkpoints and regenerate from scratch
- All scripts accept `--subsample` for fast dev iteration on a subset of images
- Most model wrappers need ~16-24GB VRAM; vLLM backends can trade memory for speed
- Data paths in `config/__init__.py` are absolute and machine-specific — update `DATA_BASE` for different environments
- Prompt A is the only prompt used for all-model evaluation; B/C are sensitivity-only (LLaVA + Qwen2VL, ref-free metrics only)
- `qwen2vl` short name maps to `Qwen/Qwen2.5-VL-7B-Instruct` (naming is historical)
- `eval/ref_free.py:CLIPScorer` uses `local_files_only=True` — models must be pre-downloaded
- Training uses bfloat16 throughout (torch_dtype + compute dtype) to prevent NaN
- Qwen2.5-VL requires lower resolution (few-shot processor / training processor) to fit within 24GB VRAM
- The `scripts/run_eval.py --all` flag hardcodes which combos get ref-free-only (B/C on llava+qwen2vl) and which get both (A on all models)
