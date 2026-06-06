"""QLoRA model loading utilities shared by training and inference."""

import os

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


def _patch_bitsandbytes_compat():
    """Monkey-patch bitsandbytes for models missing all_tied_weights_keys.

    InternVL3/3.5 (and potentially other custom-model VLMs) lack the
    all_tied_weights_keys attribute that bitsandbytes 4-bit quantization
    accesses during model loading. Provide a fallback to prevent crashes.

    Idempotent — safe to call multiple times.
    """
    import transformers.quantizers.base as _qbase
    if not hasattr(_qbase, "_uico_patched"):
        _orig = _qbase.get_keys_to_not_convert
        def _safe_get_keys(model):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            return _orig(model)
        _qbase.get_keys_to_not_convert = _safe_get_keys
        _qbase._uico_patched = True


def load_qlora_model(model_class, model_id: str, lora_config: LoraConfig,
                     device: str = "cuda:0", trust_remote_code: bool = False,
                     model_kwargs: dict = None):
    """Load 4-bit quantized base model with LoRA adapters for training.

    Returns the PEFT-wrapped model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    extra = dict(trust_remote_code=trust_remote_code)
    if model_kwargs:
        extra.update(model_kwargs)

    _patch_bitsandbytes_compat()

    base_model = model_class.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        **extra,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    return model


def load_qlora_for_inference(model_class, model_id: str, lora_dir: str,
                             device: str = "cuda:0",
                             trust_remote_code: bool = False,
                             model_kwargs: dict = None):
    """Load a QLoRA model with adapters for inference.

    Returns (model, processor) tuple.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    extra = dict(trust_remote_code=trust_remote_code)
    if model_kwargs:
        extra.update(model_kwargs)

    _patch_bitsandbytes_compat()

    base_model = model_class.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        **extra,
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    # Load processor from lora_dir if available, otherwise from model_id
    # (early-stopped checkpoints may not have processor files saved)
    preprocessor_path = os.path.join(lora_dir, "preprocessor_config.json")
    if os.path.exists(preprocessor_path):
        processor = AutoProcessor.from_pretrained(lora_dir)
    else:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
            local_files_only=True)
    return model, processor
