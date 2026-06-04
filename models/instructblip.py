"""InstructBLIP Vicuna-7B wrapper."""

import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from .base import VLMWrapper


class InstructBLIPWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "instructblip"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "Salesforce/instructblip-vicuna-7b"
        self._processor = InstructBlipProcessor.from_pretrained(model_id)
        self._model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
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
        # Strip input tokens — keep only generated response
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
