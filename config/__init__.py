"""Central configuration for VLM evaluation pipeline."""

import os
from pathlib import Path

# --- Load .env if present ---
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _sep, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip().strip("'\"")
                if _key not in os.environ:
                    os.environ[_key] = _val

from config.models import MODEL_REGISTRY, SENSITIVITY_MODELS  # noqa: F401 — re-exported for backward compatibility

# --- Base paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# --- Data paths ---
DATA_BASE = "/home/uesr/zhao/media_data/ccmc"
ANNOTATIONS_DIR = os.path.join(DATA_BASE, "annotations")
TEST_ANN_FILE = os.path.join(ANNOTATIONS_DIR, "captions_test.json")
IMAGES_BASE_DIR = os.path.join(DATA_BASE, "images")
VAL_IMAGES_DIR = os.path.join(IMAGES_BASE_DIR, "ccmc_val")

# Prompt templates are defined in config/prompts.py (single source of truth).

# --- Evaluation ---
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# --- Phase control ---
RANDOM_SEED = 42

# --- Inference ---
MAX_NEW_TOKENS = 128

# --- vLLM backend ---
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_MAX_MODEL_LEN = 2048
VLLM_MAX_NUM_SEQS = 1
VLLM_ENFORCE_EAGER = True
VLLM_LIMIT_MM_PER_PROMPT = {"image": 1}
