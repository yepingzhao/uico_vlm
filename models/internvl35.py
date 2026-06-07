"""InternVL3.5-8B wrapper.

Same architecture family as InternVL2/2.5/3 with chat() API.
Key difference from InternVL2/2.5: uses Qwen2Tokenizer (not InternLM2Tokenizer),
so _load_tokenizer is overridden to use AutoTokenizer.
"""

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
