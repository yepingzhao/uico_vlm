"""InternVL3.5-8B wrapper.

Same architecture family as InternVL2/2.5/3 with chat() API.
Key difference from InternVL2/2.5: uses Qwen2Tokenizer (not InternLM2Tokenizer),
so _load_tokenizer is overridden to use AutoTokenizer.

Supports zero-shot and few-shot inference via chat() API with history.
"""

import torch
from PIL import Image
from transformers import AutoTokenizer

from ._internvl_base import InternVLBase
from .utils import find_snapshot_dir


class InternVL35Wrapper(InternVLBase):

    model_id = "OpenGVLab/InternVL3_5-8B"

    @property
    def model_name(self) -> str:
        return "internvl35"

    def load(self, device: str = "cuda:0"):
        self._device = device

        # Monkey-patch transformers compat for InternVL models that lack
        # all_tied_weights_keys (same issue as InternVL2, needed for the
        # InternVLChatModel architecture).
        import transformers.modeling_utils as _mu
        _orig_gbc = getattr(_mu, "get_total_byte_count", None)
        _orig_move = getattr(
            _mu.PreTrainedModel, "_move_missing_keys_from_meta_to_device", None)
        _was_patched = hasattr(_mu, "_internvl35_patched")

        if not _was_patched and _orig_gbc is not None:
            def _patched_gbc(model, accelerator_device_map, hf_quantizer):
                if not hasattr(model, "all_tied_weights_keys"):
                    return {}
                return _orig_gbc(model, accelerator_device_map, hf_quantizer)

            def _patched_move(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    tied = getattr(self, "_tied_weights_keys", None)
                    self.all_tied_weights_keys = tied if tied is not None else {}
                return _orig_move(self, *args, **kwargs)

        snap_dir = find_snapshot_dir(self.model_id)
        self._load_tokenizer(snap_dir)
        self._load_image_processor()

        if not _was_patched and _orig_gbc is not None:
            _mu.get_total_byte_count = _patched_gbc
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
            _mu._internvl35_patched = True
        try:
            self._load_model(self.model_id, device)
        finally:
            if not _was_patched and _orig_gbc is not None:
                _mu.get_total_byte_count = _orig_gbc
                _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move
                del _mu._internvl35_patched

    def _load_tokenizer(self, snap_dir: str):
        """Load Qwen2Tokenizer via AutoTokenizer (not InternLM2Tokenizer).

        InternVL3.5 uses a Qwen-family LLM backbone, so the tokenizer is
        Qwen2Tokenizer rather than InternLM2Tokenizer used by InternVL2/2.5.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(
            snap_dir, trust_remote_code=True,
        )

    # --- Few-shot support ---

    @property
    def supports_fewshot(self) -> bool:
        return True

    def generate_fewshot(
        self,
        test_image_path: str,
        prompt_template: str,
        example_images: list,
        example_captions: list,
        **kwargs,
    ) -> str:
        """Generate caption using few-shot in-context examples via chat() API.

        InternVL's processor-free chat() API takes a history of
        (question, answer) tuples. We build history from example images
        and captions, concatenate all pixel_values, and pass
        num_patches_list to distinguish per-image patch counts.
        """
        gen_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "do_sample": False,
        }

        history = []
        all_pixel_values = []
        num_patches_list = []

        # Build history from example images/captions
        for ex_img_path, ex_caption in zip(example_images, example_captions):
            ex_img = Image.open(ex_img_path).convert("RGB")
            pv = self._img_processor(images=ex_img, return_tensors="pt")
            pv = pv["pixel_values"]  # [1, 3, 448, 448]
            all_pixel_values.append(pv)
            num_patches_list.append(pv.shape[0])
            # When history is non-None, chat() does NOT auto-prepend
            # <image>\n — we must include it explicitly.
            history.append(("<image>\nDescribe this image:", ex_caption))

        # Test image
        test_img = Image.open(test_image_path).convert("RGB")
        pv = self._img_processor(images=test_img, return_tensors="pt")
        pv = pv["pixel_values"]
        all_pixel_values.append(pv)
        num_patches_list.append(pv.shape[0])

        pixel_values = torch.cat(all_pixel_values, dim=0).to(self._device)
        question = f"<image>\n{prompt_template}"

        response = self._model.chat(
            self._tokenizer,
            pixel_values=pixel_values.to(torch.float16),
            question=question,
            generation_config=gen_config,
            history=history,
            num_patches_list=num_patches_list,
        )
        return response.strip()
