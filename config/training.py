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
    warmup_ratio: float = 0.03
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
        "target_modules": (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    },
    "llava-next": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class_name": "LlavaNextForConditionalGeneration",
        "processor_class_name": "LlavaNextProcessor",
        "target_modules": (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class_name": "Qwen2_5_VLForConditionalGeneration",
        "processor_class_name": "AutoProcessor",
        "target_modules": (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
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
