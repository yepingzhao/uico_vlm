"""LLaVA-1.5 Vicuna-7B wrapper."""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import VLMWrapper


class LLaVAWrapper(VLMWrapper):

    _lora_config_key = "llava"

    @property
    def model_name(self) -> str:
        return "llava"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "llava-hf/llava-1.5-7b-hf"
        self._processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, local_files_only=True,
        ).to(device)
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        # Build conversation with system + user message
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
        # Strip input tokens, keep only generated response
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
