"""LLaVA-NeXT (LLaVA-1.6) Mistral-7B wrapper."""

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .base import VLMWrapper


class LLaVANeXTWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "llava-next"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        self._processor = LlavaNextProcessor.from_pretrained(model_id)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=image, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """LLaVA-style: images passed separately, not embedded in content."""
        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self._processor(
            images=all_images, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)
