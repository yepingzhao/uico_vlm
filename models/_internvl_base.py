"""Shared base for InternVL2 and InternVL2.5 wrappers.

Both use the model's built-in chat() API with a CLIPImageProcessor
and InternLM2Tokenizer loaded from the HF cache snapshot.
"""

import sys

import torch
from transformers import AutoModel, CLIPImageProcessor

from .base import VLMWrapper


class InternVLBase(VLMWrapper):
    """Common logic for InternVL family models."""

    # Subclasses must define:
    #   model_id: str (class attribute)
    #   model_name: str (property)

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self._img_processor = None

    def _load_tokenizer(self, snap_dir: str):
        """Load InternLM2Tokenizer from snapshot directory."""
        if snap_dir not in sys.path:
            sys.path.insert(0, snap_dir)
        from tokenization_internlm2 import InternLM2Tokenizer
        self._tokenizer = InternLM2Tokenizer.from_pretrained(
            snap_dir, trust_remote_code=True,
        )

    def _load_image_processor(self):
        """Load CLIPImageProcessor with standard preprocessing."""
        self._img_processor = CLIPImageProcessor(
            size=448, crop_size=448,
            do_center_crop=True, do_normalize=True, do_resize=True,
        )

    def load_lora(self, lora_dir: str, device: str = "cuda:0"):
        """Load 4-bit quantized base + LoRA adapters with InternVL processor setup.

        InternVL uses a tokenizer + image_processor pair (not a unified processor),
        so we override the default load_lora() to handle the tokenizer/
        image_processor separation and model-specific monkey-patching.
        """
        import torch
        from transformers import (
            BitsAndBytesConfig,
            AutoModel,
        )
        from peft import PeftModel

        from config.training import get_lora_config
        from models.lora import _patch_bitsandbytes_compat

        cfg = get_lora_config(self._lora_config_key)
        model_id = cfg["model_id"]
        trust_remote_code = cfg.get("trust_remote_code", False)
        model_kwargs = cfg.get("model_kwargs", {})
        local_only = model_kwargs.get("local_files_only", False)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        _patch_bitsandbytes_compat()

        # Monkey-patch for InternVL compat
        import transformers.modeling_utils as _mu
        _orig_gbc = getattr(_mu, "get_total_byte_count", None)
        _orig_move = getattr(
            _mu.PreTrainedModel, "_move_missing_keys_from_meta_to_device",
            None)
        _was_patched = hasattr(_mu, "_internvl_lora_patched")

        if not _was_patched and _orig_gbc is not None:
            def _patched_gbc(model, accelerator_device_map, hf_quantizer):
                if not hasattr(model, "all_tied_weights_keys"):
                    return {}
                return _orig_gbc(model, accelerator_device_map, hf_quantizer)

            def _patched_move(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    tied = getattr(self, "_tied_weights_keys", None)
                    self.all_tied_weights_keys = (
                        tied if tied is not None else {}
                    )
                return _orig_move(self, *args, **kwargs)

        extra = {"trust_remote_code": trust_remote_code}
        if local_only:
            extra["local_files_only"] = True

        # Restore patches if previously applied (may have been removed by
        # a prior load's finally block)
        if not _was_patched and _orig_gbc is not None:
            _mu.get_total_byte_count = _patched_gbc
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
            _mu._internvl_lora_patched = True

        try:
            base_model = AutoModel.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=torch.bfloat16,
                **extra,
            )
        finally:
            # Restore originals; InternVL only needs patching during loading
            if not _was_patched and _orig_gbc is not None:
                _mu.get_total_byte_count = _orig_gbc
                _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move
                del _mu._internvl_lora_patched

        self._model = PeftModel.from_pretrained(base_model, lora_dir)
        self._model.eval()
        self._device = device

        # Load tokenizer (subclass-specific) and image processor
        from models.utils import find_snapshot_dir
        snap_dir = find_snapshot_dir(model_id)

        self._load_tokenizer(snap_dir)
        self._load_image_processor()

        # Set img_context_token_id on inner model for generate()
        _img_ctx = self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        _inner = self._model
        for _step in ("base_model", "model"):
            _inner = getattr(_inner, _step, _inner)
        if hasattr(_inner, "img_context_token_id"):
            _inner.img_context_token_id = _img_ctx

    def _load_model(self, model_id: str, device: str):
        """Load the AutoModel with standard config."""
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        """Generate caption using the model's chat() API.

        Subclasses may override _format_question() to customize the
        prompt format (e.g. InternVL2.5 prepends <image>\\n).
        """
        self._validate_image(image_path)
        from PIL import Image

        gen_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "do_sample": False,
        }

        image = Image.open(image_path).convert("RGB")
        pixel_values = self._img_processor(images=image, return_tensors="pt")
        pixel_values = pixel_values["pixel_values"].to(self._device)

        response = self._model.chat(
            self._tokenizer,
            pixel_values=pixel_values.to(torch.float16),
            question=self._format_question(prompt),
            generation_config=gen_config,
        )
        return response.strip()

    def _format_question(self, prompt: str) -> str:
        """Format the user prompt for chat().

        Override in subclasses if the model expects a specific format.
        Default: pass through as-is (InternVL2 behavior).
        """
        return prompt
