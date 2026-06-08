"""Training configuration and model registry for QLoRA fine-tuning."""

import os
from dataclasses import dataclass
from typing import Dict

from config import DATA_BASE


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for QLoRA fine-tuning.

    model_id, model_class_name, processor_class_name, and target_modules
    are resolved from MODEL_LORA_CONFIGS based on --model CLI arg.
    """
    # Resolved from registry
    model_id: str = ""
    model_class_name: str = ""
    processor_class_name: str = ""
    target_modules: tuple = ()
    output_dir: str = ""

    # Data
    train_ann_file: str = os.path.join(DATA_BASE, "annotations", "captions_train.json")
    max_samples: int = 0

    # LoRA (community-standard for QLoRA VLM captioning)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1   # 10% — Qwen2.5-VL community standard
    num_epochs: int = 1
    max_grad_norm: float = 1.0

    # Checkpoint
    save_steps: int = 2000
    logging_steps: int = 50

    # Device
    device: str = "cuda:0"
    seed: int = 42


MODEL_LORA_CONFIGS: Dict[str, dict] = {
    # Community-standard QLoRA config for VLM captioning:
    # - r=16 (mid-range for complex cross-modal tasks, QLoRA compensation)
    # - alpha=32 (2×r, standard scaling)
    # - target_modules: all 7 linear projections (attention + MLP)
    #   per VLM fine-tuning best practices (Qwen-VL / LLaVA guides)
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_class_name": "LlavaForConditionalGeneration",
        "processor_class_name": "AutoProcessor",
        # 4 attention-only modules — unified with Qwen3VL/InternVL35
        # for fair cross-model LoRA comparison.
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    },
    "llava-next": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class_name": "LlavaNextForConditionalGeneration",
        "processor_class_name": "LlavaNextProcessor",
        # 4 attention-only modules: match Qwen2VL conservative config
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class_name": "Qwen2_5_VLForConditionalGeneration",
        "processor_class_name": "AutoProcessor",
        # Qwen2.5-VL diverges (NaN) with 7 target modules under QLoRA 4-bit.
        # Stick to attention-only projections; LLaVA is stable with 7.
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    },
    "phi35-vision": {
        "model_id": "microsoft/Phi-3.5-vision-instruct",
        "model_class_name": "AutoModelForCausalLM",
        "processor_class_name": "AutoProcessor",
        # Phi-3.5 uses combined projections: qkv_proj + gate_up_proj
        "target_modules": ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
        "trust_remote_code": True,
        # Phi-3.5 config defaults to flash_attention_2; config.json patched to
        # sdpa on disk. Use local_files_only to avoid network hangs.
        "model_kwargs": {"local_files_only": True},
    },
    "internvl2": {
        "model_id": "OpenGVLab/InternVL2-8B",
        "model_class_name": "AutoModelForCausalLM",
        "processor_class_name": "AutoProcessor",
        # InternLM2 backbone uses non-standard naming
        "target_modules": ("wqkv", "wo", "w1", "w2", "w3"),
        "trust_remote_code": True,
        "model_kwargs": {"local_files_only": True},
    },
    "qwen3vl": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_class_name": "Qwen3VLForConditionalGeneration",
        "processor_class_name": "AutoProcessor",
        # Same attention-only strategy as Qwen2VL (4 modules) — avoids NaN
        # risk observed with 7 modules under QLoRA 4-bit on Qwen backbones.
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "model_kwargs": {"local_files_only": True},
    },
    # InternVL3/3.5: Qwen2/Qwen3 LLM backbone + InternViT vision.
    # Chat template = internvl2_5 (same <img>/IMG_CONTEXT format as InternVL2).
    # Uses InternVLChatModel → AutoModel with trust_remote_code.
    # Target modules follow Qwen naming (not InternLM2 naming).
    "internvl3": {
        "model_id": "OpenGVLab/InternVL3-8B",
        "model_class_name": "AutoModel",
        "processor_class_name": "AutoProcessor",
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "trust_remote_code": True,
        "model_kwargs": {"local_files_only": True},
    },
    "internvl35": {
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "model_class_name": "AutoModel",
        "processor_class_name": "AutoProcessor",
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "trust_remote_code": True,
        "model_kwargs": {"local_files_only": True},
    },
}


def get_lora_config(model_name: str) -> dict:
    """Resolve a model short-name to its LoRA training config.

    Raises ValueError if the model is not supported for LoRA fine-tuning.
    """
    if model_name not in MODEL_LORA_CONFIGS:
        raise ValueError(
            f"Unknown or unsupported model for LoRA: {model_name}. "
            f"Available: {list(MODEL_LORA_CONFIGS.keys())}"
        )
    return MODEL_LORA_CONFIGS[model_name]
