"""InternVL2.5-8B wrapper.

Same architecture as InternVL2-8B with updated training and
dynamic resolution support. Uses the model's built-in chat() API.
"""

import os
import sys

import torch
from transformers import AutoModel

from .base import VLMWrapper
from .utils import find_snapshot_dir


class InternVL25Wrapper(VLMWrapper):

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._img_processor = None
        self._device = "cuda:0"

    @property
    def model_name(self) -> str:
        return "internvl25"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "OpenGVLab/InternVL2_5-8B"

        # Load slow tokenizer to avoid tiktoken fast-converter bug
        snap_dir = find_snapshot_dir(model_id)
        if snap_dir not in sys.path:
            sys.path.insert(0, snap_dir)
        from tokenization_internlm2 import InternLM2Tokenizer
        self._tokenizer = InternLM2Tokenizer.from_pretrained(
            snap_dir, trust_remote_code=True,
        )

        # Image processor using CLIP-standard preprocessing
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
            question=f"<image>\n{prompt}",
            generation_config=gen_config,
        )
        return response.strip()
