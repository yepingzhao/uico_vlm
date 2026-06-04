"""Qwen2.5-VL-7B-Instruct wrapper."""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .base import VLMWrapper


class Qwen2VLWrapper(VLMWrapper):

    def __init__(self):
        super().__init__()
        self._fewshot_processor = None

    @property
    def model_name(self) -> str:
        return "qwen2vl"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self._model.eval()
        # min_pixels/max_pixels control resolution; set to reasonable defaults
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        # Low-res processor for few-shot multi-image (avoids OOM on 24GB)
        self._fewshot_processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        # Build chat messages following Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)

    def generate_fewshot(
        self,
        test_image_path: str,
        prompt_template: str,
        example_images: list,
        example_captions: list,
        **kwargs,
    ) -> str:
        from ._fewshot import build_fewshot_images_and_content

        fewshot_processor = self._fewshot_processor

        all_images, content_blocks = build_fewshot_images_and_content(
            test_image_path, prompt_template, example_images, example_captions,
            embed_images=True,
        )

        messages = [{"role": "user", "content": content_blocks}]
        text = fewshot_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = fewshot_processor(
            text=[text],
            images=all_images,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs, processor=fewshot_processor)
