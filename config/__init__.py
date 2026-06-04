"""Central configuration for VLM evaluation pipeline."""

import os

# --- Base paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# --- Data paths ---
DATA_BASE = "/home/uesr/zhao/media_data/ccmc"
ANNOTATIONS_DIR = os.path.join(DATA_BASE, "annotations")
IMAGES_DIR = os.path.join(DATA_BASE, "images")
TEST_ANN_FILE = os.path.join(ANNOTATIONS_DIR, "captions_test.json")
IMAGES_BASE_DIR = os.path.join(DATA_BASE, "images")

# --- Model registry ---
# Each entry: (short_name, hf_model_id, wrapper_class_name)
MODEL_REGISTRY = [
    ("blip2",           "Salesforce/blip2-flan-t5-xl",            "BLIP2Wrapper"),
    ("instructblip",    "Salesforce/instructblip-vicuna-7b",      "InstructBLIPWrapper"),
    ("llava",           "llava-hf/llava-1.5-7b-hf",              "LLaVAWrapper"),
    ("internvl2",       "OpenGVLab/InternVL2-8B",                 "InternVL2Wrapper"),
    ("qwen2vl",         "Qwen/Qwen2.5-VL-7B-Instruct",            "Qwen2VLWrapper"),
    # New VLM baselines
    ("phi35-vision",    "microsoft/Phi-3.5-vision-instruct",       "Phi35VisionWrapper"),
    ("phi4-mm",         "microsoft/Phi-4-multimodal-instruct",     "Phi4MultimodalWrapper"),
    ("paligemma2",      "google/paligemma2-3b-ft-docci-448",      "PaliGemma2Wrapper"),
    ("minicpm-v",       "openbmb/MiniCPM-V-2_6",                  "MiniCPMVWrapper"),
    ("deepseek-vl2",    "deepseek-ai/deepseek-vl2-small",          "DeepSeekVL2Wrapper"),
    ("llava-next",      "llava-hf/llava-v1.6-mistral-7b-hf",      "LLaVANeXTWrapper"),
    ("idefics3",        "HuggingFaceM4/Idefics3-8B-Llama3",       "Idefics3Wrapper"),
    ("internvl25",      "OpenGVLab/InternVL2_5-8B",               "InternVL25Wrapper"),
    ("pixtral",         "mistralai/Pixtral-12B-2409",             "PixtralWrapper"),
    ("llama32-vision",  "meta-llama/Llama-3.2-11B-Vision-Instruct","Llama32VisionWrapper"),
]

# Phase 1 development models (lightweight, fast verification)
DEV_MODELS = ["blip2", "llava"]

# Prompt templates are defined in prompts/templates.py (single source of truth).
# Sensitivity analysis: only run B/C on these models
SENSITIVITY_MODELS = ["llava", "qwen2vl"]

# --- Evaluation ---
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# --- Phase control ---
DEV_SAMPLE_SIZE = 1000   # number of images for Phase 1 dev validation
RANDOM_SEED = 42

# --- Inference ---
MAX_NEW_TOKENS = 128
BATCH_SIZE = 1           # single-image inference for VLM generation

# --- Beam search (off for most VLM zero-shot; use greedy) ---
DO_SAMPLE = False
TEMPERATURE = 1.0

# --- vLLM backend ---
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_MAX_MODEL_LEN = 2048
VLLM_MAX_NUM_SEQS = 1
VLLM_ENFORCE_EAGER = True
VLLM_LIMIT_MM_PER_PROMPT = {"image": 1}
