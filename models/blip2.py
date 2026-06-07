"""BLIP-2 FLAN-T5-XL wrapper."""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from .base import VLMWrapper


class BLIP2Wrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "blip2"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "Salesforce/blip2-flan-t5-xl"
        self._processor = Blip2Processor.from_pretrained(model_id, local_files_only=True)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, local_files_only=True,
        ).to(device)
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._device, torch.float16
        )
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._processor.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
