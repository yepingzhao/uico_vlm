"""InternVL2.5-8B wrapper.

Same architecture as InternVL2-8B with updated training and
dynamic resolution support. Uses the model's built-in chat() API.
"""

from ._internvl_base import InternVLBase
from .utils import find_snapshot_dir


class InternVL25Wrapper(InternVLBase):

    model_id = "OpenGVLab/InternVL2_5-8B"

    @property
    def model_name(self) -> str:
        return "internvl25"

    def load(self, device: str = "cuda:0"):
        self._device = device

        snap_dir = find_snapshot_dir(self.model_id)
        self._load_tokenizer(snap_dir)
        self._load_image_processor()
        self._load_model(self.model_id, device)

    def _format_question(self, prompt: str) -> str:
        """InternVL2.5 expects <image>\\n prefix in the question."""
        return f"<image>\n{prompt}"
