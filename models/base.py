"""Abstract base class for VLM wrappers."""

from abc import ABC, abstractmethod
import os


class VLMWrapper(ABC):
    """Unified interface for zero-shot VLM inference."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cuda:0"

    @abstractmethod
    def load(self, device: str = "cuda:0"):
        """Load model and processor onto the specified device."""
        ...

    @abstractmethod
    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        """Generate a caption for a single image given a prompt.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt describing the task.

        Returns:
            Generated caption string.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used for output directory naming."""
        ...

    def _validate_image(self, image_path: str):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
