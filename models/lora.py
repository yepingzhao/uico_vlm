"""QLoRA model loading utilities shared by training and inference."""

import torch
from transformers import BitsAndBytesConfig, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


def make_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05,
                     target_modules: list = None) -> LoraConfig:
    """Build a standard LoRA config for causal LM fine-tuning."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )


def load_qlora_model(model_class, model_id: str, lora_config: LoraConfig,
                     device: str = "cuda:0"):
    """Load 4-bit quantized base model with LoRA adapters for training.

    Returns the PEFT-wrapped model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = model_class.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    return model


def load_qlora_for_inference(model_class, model_id: str, lora_dir: str,
                             device: str = "cuda:0"):
    """Load a QLoRA model with adapters for inference.

    Returns (model, processor) tuple.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = model_class.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    processor = AutoProcessor.from_pretrained(lora_dir)
    return model, processor
