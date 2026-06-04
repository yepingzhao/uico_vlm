"""MiniCPM-V-2.6 (8B) wrapper.

Uses the model's built-in chat() API which handles image preprocessing
and generation internally.
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import VLMWrapper


class MiniCPMVWrapper(VLMWrapper):

    def __init__(self):
        super().__init__()
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return "minicpm-v"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "openbmb/MiniCPM-V-2_6"

        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        ).eval().cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        msgs = [
            {"role": "user", "content": [image, prompt]},
        ]

        response = self._model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self._tokenizer,
        )
        return response.strip()
