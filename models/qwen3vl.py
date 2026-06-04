"""Qwen3-VL-8B-Instruct wrapper (2025)."""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import VLMWrapper


class Qwen3VLWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "qwen3vl"

    def load(self, device: str = "cuda:0"):
        self._device = device
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
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
        """Generate caption with k-shot in-context examples.

        Uses fewer pixels than single-image mode to fit k+1 images
        in 24GB VRAM.
        """
        from PIL import Image as PILImage

        # Build multi-image messages (same format as Qwen2.5-VL)
        content_blocks = []
        all_images = []

        for i, (ex_img_path, ex_caption) in enumerate(
            zip(example_images, example_captions)
        ):
            ex_img = PILImage.open(ex_img_path).convert("RGB")
            all_images.append(ex_img)
            content_blocks.append({"type": "image", "image": ex_img})
            content_blocks.append({
                "type": "text",
                "text": f"Example {i + 1}: {ex_caption}",
            })

        test_img = PILImage.open(test_image_path).convert("RGB")
        all_images.append(test_img)
        content_blocks.append({"type": "image", "image": test_img})
        content_blocks.append({
            "type": "text",
            "text": prompt_template,
        })

        messages = [{"role": "user", "content": content_blocks}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
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
        return self._strip_and_decode(output_ids, inputs)
