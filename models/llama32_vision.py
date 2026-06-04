"""Llama 3.2 11B Vision Instruct wrapper.

Uses device_map="auto" for multi-GPU (fits dual 4090 24GB).
Note: requires HuggingFace login for gated repo access.
    huggingface-cli login
"""

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from .base import VLMWrapper


class Llama32VisionWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "llama32-vision"

    def load(self, device: str = "cuda:0"):
        # 11B needs dual GPU — use device_map="auto"
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
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
        # add_special_tokens=False avoids double BOS bug
        inputs = self._processor(
            text=text,
            images=image,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._model.device)

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
