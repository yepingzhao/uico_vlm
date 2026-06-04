"""LLaVA-1.5 Vicuna-7B wrapper."""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import VLMWrapper


class LLaVAWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "llava"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "llava-hf/llava-1.5-7b-hf"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
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
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()

    def generate_fewshot(
        self,
        test_image_path: str,
        prompt_template: str,
        example_images: list,
        example_captions: list,
        **kwargs,
    ) -> str:
        """Generate caption with k-shot in-context examples.

        Args:
            test_image_path: Path to the test image.
            prompt_template: Text template with {instruction} placeholder.
            example_images: List of paths to example images.
            example_captions: List of example captions (same length).
            **kwargs: Passed to model.generate().

        Returns:
            Generated caption string.
        """
        from PIL import Image

        # Build multi-image conversation
        all_images = []
        content_blocks = []

        # Add example pairs: [image, text], [image, text], ...
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

        # Add test image + instruction
        test_img = Image.open(test_image_path).convert("RGB")
        all_images.append(test_img)
        content_blocks.append({"type": "image"})
        content_blocks.append({
            "type": "text",
            "text": prompt_template,
        })

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
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
