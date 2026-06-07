"""Phi-3.5-Vision-Instruct (4B) wrapper."""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import VLMWrapper


class Phi35VisionWrapper(VLMWrapper):

    _lora_config_key = "phi35-vision"

    @property
    def model_name(self) -> str:
        return "phi35-vision"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "microsoft/Phi-3.5-vision-instruct"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4,  # lower resolution to save VRAM
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        text = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=text, images=image, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
                temperature=None,
            )
        return self._strip_and_decode(output_ids, inputs)
