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

    def _load_model(self, model_id: str, device: str):
        """Load the AutoModel with standard config.

        Uses the local snapshot path (via find_snapshot_dir) to ensure
        custom modeling code is found in offline mode (HF_HUB_OFFLINE=1).
        """
        from .utils import find_snapshot_dir
        snap_dir = find_snapshot_dir(model_id)
        self._model = AutoModel.from_pretrained(
            snap_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device,
            low_cpu_mem_usage=True,
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
