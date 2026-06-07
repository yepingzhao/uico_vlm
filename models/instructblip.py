"""InstructBLIP Vicuna-7B wrapper.

Offline-loading workaround: InstructBlipProcessor.from_pretrained() internally
calls AutoTokenizer → AutoConfig.from_pretrained(), and with subfolder='qformer_tokenizer'
the kwargs (including local_files_only) get forwarded to AutoConfig which looks for
config.json in the wrong subdirectory. We bypass this by loading sub-components
individually and assembling the processor manually.
"""

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers.models.blip.image_processing_blip import BlipImageProcessor

from .base import VLMWrapper

# Cache path — resolve once at import time so the snapshot hash is immutable.
_CACHE_BASE = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Salesforce--instructblip-vicuna-7b",
    "snapshots",
    "19103d0c5b5263c8a7891012e08573439fb6607f",
)


class InstructBLIPWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "instructblip"

    def load(self, device: str = "cuda:0"):
        self._device = device
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        model_id = "Salesforce/instructblip-vicuna-7b"

        # --- Manually assemble processor (bypass broken from_pretrained) ---
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        qformer_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(_CACHE_BASE, "qformer_tokenizer"), local_files_only=True,
        )
        image_processor = BlipImageProcessor.from_pretrained(model_id, local_files_only=True)
        self._processor = InstructBlipProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            qformer_tokenizer=qformer_tokenizer,
            num_query_tokens=32,
        )

        # --- Model ---
        self._model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, local_files_only=True,
        ).to(device)
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._device, torch.float16
        )
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)
