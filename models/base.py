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

    def _strip_and_decode(self, output_ids, inputs, processor=None):
        """Strip input tokens from output and decode to string.

        Removes the input prompt tokens from the generated output, then
        decodes the remaining generated token IDs to a stripped string.

        Args:
            output_ids: Full model output token IDs (batch_size, seq_len).
            inputs: The tokenizer output dict containing "input_ids".
            processor: The processor/tokenizer to use for decoding.
                       Defaults to self._processor.

        Returns:
            Decoded, stripped caption string.
        """
        proc = processor if processor is not None else self._processor
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return proc.decode(generated_ids[0], skip_special_tokens=True).strip()
