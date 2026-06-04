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

    def generate_fewshot(
        self,
        test_image_path: str,
        prompt_template: str,
        example_images: list,
        example_captions: list,
        **kwargs,
    ) -> str:
        """Generate caption with k-shot in-context examples."""
        from PIL import Image

        all_images = []
        content_blocks = []

        for i, (ex_img_path, ex_caption) in enumerate(
            zip(example_images, example_captions)
        ):
            ex_img = Image.open(ex_img_path).convert("RGB")
            all_images.append(ex_img)
            content_blocks.append({"type": "image"})
            content_blocks.append({
                "type": "text",
                "text": f"Example {i + 1}: {ex_caption}",
            })

        test_img = Image.open(test_image_path).convert("RGB")
        all_images.append(test_img)
        content_blocks.append({"type": "image"})
        content_blocks.append({"type": "text", "text": prompt_template})

        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=all_images,
            text=formatted,
            return_tensors="pt",
        ).to(self._device, torch.float16)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)
