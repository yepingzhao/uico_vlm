"""DeepSeek-VL2-Small (3B) wrapper.

Requires the DeepSeek-VL2 package:
    pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps
    pip install attrdict timm 'transformers<4.48.0'
"""

import torch
from PIL import Image

from .base import VLMWrapper


class DeepSeekVL2Wrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "deepseek-vl2"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "deepseek-ai/deepseek-vl2-small"

        from deepseek_vl2.models import (
            DeepseekVLV2Processor,
            DeepseekVLV2ForCausalLM,
        )
        from transformers import AutoModelForCausalLM

        self._processor = DeepseekVLV2Processor.from_pretrained(model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(torch.bfloat16).cuda().eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        from deepseek_vl2.utils.io import load_pil_images

        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)

        prepare_inputs = self._processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self._model.device)

        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        with torch.no_grad():
            outputs = self._model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self._model.language_model.config.eos_token_id,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )

        return self._model.language_model.tokenizer.decode(
            outputs[0].cpu(), skip_special_tokens=True
        ).strip()
