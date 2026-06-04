"""Phi-4-Multimodal-Instruct (5.6B) wrapper.

Requires vision-lora adapter for visual input processing.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import VLMWrapper


class Phi4MultimodalWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "phi4-mm"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "microsoft/Phi-4-multimodal-instruct"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
        )

        # Load vision LoRA adapter — required for visual input
        self._model.load_adapter(
            model_id,
            adapter_name="vision",
            adapter_kwargs={"subfolder": "vision-lora"},
        )
        self._model.set_adapter("vision")
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
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
