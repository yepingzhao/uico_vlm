"""PaliGemma 2 (3B, DOCCI fine-tuned) wrapper.

Uses task prefix "caption en" for detailed image descriptions.
"""

import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

from .base import VLMWrapper


class PaliGemma2Wrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "paligemma2"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "google/paligemma2-3b-ft-docci-448"

        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

        self._processor = PaliGemmaProcessor.from_pretrained(model_id)

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        # PaliGemma2 uses task prefixes; append user prompt as context
        task_prompt = f"caption en: {prompt}"

        inputs = self._processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        ).to(torch.bfloat16).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )

        # Strip input tokens
        input_len = inputs["input_ids"].shape[1]
        return self._processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
