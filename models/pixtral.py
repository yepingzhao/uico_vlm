"""Pixtral-12B wrapper.

Uses device_map="auto" for multi-GPU (fits dual 4090 24GB).
"""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import VLMWrapper


class PixtralWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "pixtral"

    def load(self, device: str = "cuda:0"):
        # Pixtral 12B needs dual GPU — use device_map="auto" regardless of device param
        model_id = "mistralai/Pixtral-12B-2409"

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)
