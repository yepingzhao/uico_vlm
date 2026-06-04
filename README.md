# UICO-VLM: Urban Incivility Captioning with Vision-Language Models

A benchmark evaluating 15+ vision-language models on the task of describing civic norm violations in urban scenes, using the UICO dataset (CCMC images in COCO format). Covers zero-shot, few-shot, and QLoRA fine-tuned inference.

**Paper:** TCSVT 2026 submission — compares VLM baselines against specialized captioning models (Transformer, SCST, AoANet, UIFormer, etc.).

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate uico_vlm

# Install coco-caption (if not already in environment.yml)
pip install git+https://github.com/ruotianluo/coco-caption.git
```

**Key dependencies:** Python 3.10, PyTorch 2.5.1, transformers ≥4.46, vLLM, pycocotools.

### Import Path Setup

The code imports itself as `vlm_eval`. Create a symlink in the workspace root:

```bash
ln -s /path/to/uico_vlm /path/to/vlm_eval
```

Without this, `from vlm_eval.xxx` imports will fail.

## Data

Data lives at the path configured in `config.py:DATA_BASE` (default: `/home/uesr/zhao/media_data/ccmc/`).

```
DATA_BASE/
├── annotations/
│   ├── captions_train.json    # training set, 5 reference captions per image
│   └── captions_test.json     # test set, 5 reference captions per image
└── images/
    ├── ccmc_train/            # training images
    ├── ccmc_test/             # test images
    └── ccmc_val/              # validation images
```

Update `DATA_BASE` in `config.py` for your environment.

## Model Zoo

| Short Name      | HuggingFace Model ID                         | Wrapper Class       |
|-----------------|----------------------------------------------|---------------------|
| `blip2`         | `Salesforce/blip2-flan-t5-xl`                | `BLIP2Wrapper`      |
| `instructblip`  | `Salesforce/instructblip-vicuna-7b`          | `InstructBLIPWrapper`|
| `llava`         | `llava-hf/llava-1.5-7b-hf`                  | `LLaVAWrapper`      |
| `internvl2`     | `OpenGVLab/InternVL2-8B`                     | `InternVL2Wrapper`  |
| `qwen2vl`       | `Qwen/Qwen2.5-VL-7B-Instruct`                | `Qwen2VLWrapper`    |
| `phi35-vision`  | `microsoft/Phi-3.5-vision-instruct`          | `Phi35VisionWrapper`|
| `phi4-mm`       | `microsoft/Phi-4-multimodal-instruct`        | `Phi4MultimodalWrapper`|
| `paligemma2`    | `google/paligemma2-3b-ft-docci-448`          | `PaliGemma2Wrapper` |
| `minicpm-v`     | `openbmb/MiniCPM-V-2_6`                      | `MiniCPMVWrapper`   |
| `deepseek-vl2`  | `deepseek-ai/deepseek-vl2-small`             | `DeepSeekVL2Wrapper`|
| `llava-next`    | `llava-hf/llava-v1.6-mistral-7b-hf`          | `LLaVANeXTWrapper`  |
| `idefics3`      | `HuggingFaceM4/Idefics3-8B-Llama3`           | `Idefics3Wrapper`   |
| `internvl25`    | `OpenGVLab/InternVL2_5-8B`                   | `InternVL25Wrapper` |
| `pixtral`       | `mistralai/Pixtral-12B-2409`                 | `PixtralWrapper`    |
| `llama32-vision`| `meta-llama/Llama-3.2-11B-Vision-Instruct`   | `Llama32VisionWrapper`|

**vLLM backends** (append `-vllm` to the short name): `llava-vllm`, `qwen2vl-vllm`. These use batched offline inference via `LLM.chat()` as an alternative to HF `model.generate()`.

**Dev models** for fast iteration: `blip2`, `llava` (configured in `config.py:DEV_MODELS`).

## Usage

### 1. Download Models

```bash
python download_models.py --dry-run          # list all models
python download_models.py --model llava      # download a single model
python download_models.py                    # download all 15 models
```

Uses `hf-mirror.com` by default (set `HF_ENDPOINT` to override).

### 2. Zero-Shot Inference

```bash
# Phase 1 (dev): 1000 images, 2 lightweight models
python scripts/run_inference.py --models blip2 llava --subsample 1000 --prompt A

# Phase 2 (full): all images, all models
python scripts/run_inference.py --models blip2 instructblip llava internvl2 qwen2vl --prompt A

# Sensitivity analysis: prompt variants B/C on LLaVA + Qwen2.5-VL
python scripts/run_inference.py --models llava qwen2vl --prompt B
python scripts/run_inference.py --models llava qwen2vl --prompt C

# Chinese prompt (Qwen2.5-VL only)
python scripts/run_inference.py --models qwen2vl --prompt ZH

# vLLM backend: faster inference
python scripts/run_inference.py --models llava-vllm qwen2vl-vllm --prompt A
```

Supports checkpoint/resume — interrupted runs pick up where they left off.

### 3. Evaluation

```bash
# Evaluate a single model/prompt pair
python scripts/run_eval.py --model blip2 --prompt A

# Reference-free metrics only (skip BLEU/METEOR/ROUGE/CIDEr/SPICE)
python scripts/run_eval.py --model blip2 --prompt A --ref_free_only

# Evaluate all models, all prompts
python scripts/run_eval.py --all
```

### 4. Few-Shot Inference

```bash
# Quick test: 3 images, k=1
python scripts/run_fewshot.py --models llava --k 1 --subsample 3

# Dev run: 500 images, k=1,3,5
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5 --subsample 500

# Full run
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5

# Evaluate few-shot results
python scripts/eval_fewshot.py --model llava --k 1
python scripts/eval_fewshot.py --all
```

Few-shot examples are pre-sampled once from the training set (fixed seed, cached to disk) and reused across all test images.

### 5. QLoRA Fine-Tuning

```bash
python -m vlm_eval.train.train_llava_lora
```

Fine-tunes LLaVA-1.5-7B with:
- 4-bit NF4 quantization (QLoRA)
- LoRA rank=8 on q/k/v/o attention projections
- Vision encoder frozen, multimodal projector quantized (not tuned)
- Masked LM loss on caption tokens only
- SwanLab logging

### 6. LoRA Inference

```bash
python scripts/inference_lora.py
```

### 7. Table Generation

```bash
python make_table.py
```

Generates LaTeX comparison tables from `outputs/all_metrics.json`, comparing VLM results against paper baselines with bold-for-best formatting.

## Prompt Variants

| Key | Description | Usage |
|-----|-------------|-------|
| `A` | Concise: "Describe any urban incivility..." | Primary — all models |
| `B` | Structured: 3-part analysis (type, location, why) | Sensitivity — LLaVA + Qwen2.5-VL |
| `C` | Governance: urban management inspector persona | Sensitivity — LLaVA + Qwen2.5-VL |
| `ZH` | Chinese: 请描述这张图片中存在的城市不文明现象 | Supplemental — Qwen2.5-VL |

## Evaluation Metrics

### Reference-Based
Computed via `pycocoevalcap`:

- **BLEU-1 / BLEU-4** — n-gram precision
- **METEOR** — synonym-aware matching
- **ROUGE-L** — longest common subsequence
- **CIDEr-D** — consensus-based image description
- **SPICE** — scene graph semantic matching
- **S_m** — composite score: mean of B@4, M, R, C, S (Eq. 1 in paper)

Captions are truncated to 50 words before SPICE evaluation to prevent Stanford parser OOM.

### Reference-Free
Computed via CLIP (`openai/clip-vit-large-patch14`):

- **CLIPScore** — cosine similarity of image and text embeddings
- **RefCLIPScore** — harmonic mean of CLIPScore and max reference CLIPScore

## Output Structure

```
outputs/
├── {model_name}/
│   ├── predictions_prompt_a.jsonl          # zero-shot predictions
│   ├── predictions_prompt_b.jsonl          # sensitivity (B prompt)
│   ├── predictions_fewshot_k1.jsonl        # few-shot (k=1)
│   ├── predictions_fewshot_k3.jsonl        # few-shot (k=3)
│   ├── metrics_prompt_a.json               # per-prompt evaluation
│   └── ...
├── all_metrics.json                        # aggregate zero-shot results
├── fewshot_all_metrics.json                # aggregate few-shot results
├── fewshot_cache/                          # cached few-shot example selections
├── llava-lora/                             # QLoRA adapter weights
└── tables/                                 # generated LaTeX tables
```

## Configuration

Edit `config.py` for:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_BASE` | Path to CCMC data | `/home/uesr/zhao/media_data/ccmc` |
| `DEV_MODELS` | Lightweight models for fast iteration | `["blip2", "llava"]` |
| `DEV_SAMPLE_SIZE` | Subsampling size for dev runs | `1000` |
| `MAX_NEW_TOKENS` | Generation budget | `128` |
| `RANDOM_SEED` | Global random seed | `42` |
| `CLIP_MODEL_NAME` | CLIP model for ref-free eval | `openai/clip-vit-large-patch14` |
| `VLLM_GPU_MEMORY_UTILIZATION` | vLLM memory fraction | `0.9` |

## Architecture

- `models/` — VLM wrappers implementing a common `generate()` interface, with optional `generate_fewshot()` for few-shot capable models
- `data/` — COCO dataset loader with subsampling and image path resolution
- `prompts/` — prompt templates (A/B/C/ZH)
- `eval/` — reference-based metrics (BLEU, METEOR, ROUGE, CIDEr, SPICE) and reference-free metrics (CLIPScore, RefCLIPScore)
- `fewshot/` — static-example few-shot sampler with disk caching
- `train/` — QLoRA fine-tuning for LLaVA-1.5-7B
- `config.py` — central configuration

## License

This project is for academic research purposes. See the paper for citation details.

## Citation

If you use this work, please cite the corresponding TCSVT 2026 paper.
