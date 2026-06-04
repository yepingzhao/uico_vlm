"""Idefics3-8B-Llama3 wrapper."""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from .base import VLMWrapper


class Idefics3Wrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "idefics3"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "HuggingFaceM4/Idefics3-8B-Llama3"

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
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
            images=[image],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)
