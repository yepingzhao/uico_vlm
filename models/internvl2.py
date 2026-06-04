"""InternVL2-8B wrapper.

Uses the model's built-in chat interface which handles
image preprocessing and generation internally.
"""

import sys

import torch
from transformers import AutoModel

from .base import VLMWrapper
from .utils import find_snapshot_dir


class InternVL2Wrapper(VLMWrapper):

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self._img_processor = None

    @property
    def model_name(self) -> str:
        return "internvl2"

    def load(self, device: str = "cuda:0"):
        self._device = device

        # Monkey-patch transformers for InternVL2 compatibility with
        # transformers >= 5.x (missing all_tied_weights_keys attribute).
        import transformers.modeling_utils as _mu
        if not hasattr(_mu, "_internvl2_patched"):
            _orig_gbc = _mu.get_total_byte_count
            _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device

            def _patched_gbc(model, accelerator_device_map, hf_quantizer):
                if not hasattr(model, "all_tied_weights_keys"):
                    return {}
                return _orig_gbc(model, accelerator_device_map, hf_quantizer)

            def _patched_move(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    tied = getattr(self, "_tied_weights_keys", None)
                    self.all_tied_weights_keys = tied if tied is not None else {}
                return _orig_move(self, *args, **kwargs)

            _mu.get_total_byte_count = _patched_gbc
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
            _mu._internvl2_patched = True

        model_id = "OpenGVLab/InternVL2-8B"

        # Load slow tokenizer directly to avoid the tiktoken fast-converter
        # bug in transformers >=4.46 when used with InternVL2's custom code.
        snap_dir = find_snapshot_dir(model_id)
        if snap_dir not in sys.path:
            sys.path.insert(0, snap_dir)
        from tokenization_internlm2 import InternLM2Tokenizer
        self._tokenizer = InternLM2Tokenizer.from_pretrained(
            snap_dir, trust_remote_code=True,
        )

        # Load image processor using CLIP-standard preprocessing
        from transformers import CLIPImageProcessor
        self._img_processor = CLIPImageProcessor(
            size=448, crop_size=448,
            do_center_crop=True, do_normalize=True, do_resize=True,
        )

        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        from PIL import Image

        # Use plain dict for generation_config — InternVL2's chat() does
        # item assignment (config['eos_token_id'] = ...) which fails on
        # the GenerationConfig object in newer transformers.
        gen_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "do_sample": False,
        }

        # Preprocess image to pixel_values tensor (chat() expects a tensor)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self._img_processor(images=image, return_tensors="pt")
        pixel_values = pixel_values["pixel_values"].to(self._device)

        response = self._model.chat(
            self._tokenizer,
            pixel_values=pixel_values.to(torch.float16),
            question=prompt,
            generation_config=gen_config,
        )
        return response.strip()
