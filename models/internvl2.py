"""InternVL2-8B wrapper.

Uses the model's built-in chat interface which handles
image preprocessing and generation internally.
"""

from ._internvl_base import InternVLBase
from .utils import find_snapshot_dir


class InternVL2Wrapper(InternVLBase):

    model_id = "OpenGVLab/InternVL2-8B"

    @property
    def model_name(self) -> str:
        return "internvl2"

    def load(self, device: str = "cuda:0"):
        self._device = device

        # Monkey-patch transformers for InternVL2 compatibility with
        # transformers >= 5.x (missing all_tied_weights_keys attribute).
        # Only apply when the patched API exists (skip for transformers 4.x).
        import transformers.modeling_utils as _mu
        if not hasattr(_mu, "_internvl2_patched"):
            if hasattr(_mu, "get_total_byte_count"):
                _orig_gbc = _mu.get_total_byte_count
                _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device

                def _patched_gbc(model, accelerator_device_map, hf_quantizer):
                    if not hasattr(model, "all_tied_weights_keys"):
                        return {}
                    return _orig_gbc(model, accelerator_device_map, hf_quantizer)

                def _patched_move(self, *args, **kwargs):
                    if not hasattr(self, "all_tied_weights_keys"):
                        tied = getattr(self, "_tied_weights_keys", None)
                        self.all_tied_weights_keys = tied if tied is not None else {}
                    return _orig_move(self, *args, **kwargs)

                _mu.get_total_byte_count = _patched_gbc
                _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
            _mu._internvl2_patched = True

        snap_dir = find_snapshot_dir(self.model_id)
        self._load_tokenizer(snap_dir)
        self._load_image_processor()
        self._load_model(self.model_id, device)
