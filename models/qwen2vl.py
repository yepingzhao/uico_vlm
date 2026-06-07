"""Qwen2.5-VL-7B-Instruct wrapper."""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .base import VLMWrapper


class Qwen2VLWrapper(VLMWrapper):

    _lora_config_key = "qwen2vl"

    def __init__(self):
        super().__init__()
        self._fewshot_processor = None

    @property
    def model_name(self) -> str:
        return "qwen2vl"

    def load(self, device: str = "cuda:0"):
        import os as _os
        from pathlib import Path as _Path

        self._device = device
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        # Resolve local cache path — avoids HF Hub network calls
        # (transformers cached_files() requires _commit_hash to use
        # try_to_load_from_cache, which isn't available in offline mode)
        cache_root = _os.path.expanduser("~/.cache/huggingface/hub")
        ref_file = _os.path.join(
            cache_root,
            "models--Qwen--Qwen2.5-VL-7B-Instruct",
            "refs",
            "main",
        )
        if _os.path.isfile(ref_file):
            commit = _Path(ref_file).read_text().strip()
            snapshot = _os.path.join(
                cache_root,
                "models--Qwen--Qwen2.5-VL-7B-Instruct",
                "snapshots",
                commit,
            )
        else:
            snapshot = model_id  # fallback

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            snapshot, torch_dtype=torch.float16
        ).to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            snapshot,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        # Low-res processor for few-shot multi-image (avoids OOM on 24GB)
        self._fewshot_processor = AutoProcessor.from_pretrained(
            snapshot,
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

    _fewshot_embed_images = True

    def _get_fewshot_processor(self):
        """Return low-res processor for few-shot to avoid OOM on 24GB GPUs."""
        return self._fewshot_processor

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """Qwen-style: embed PIL images in content, list-based processor call."""
        processor = self._get_fewshot_processor()
        messages = [{"role": "user", "content": content_blocks}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
