"""GenerationStrategy ABC and concrete strategy implementations.

Each strategy encapsulates HOW to generate a caption from an image path.
The InferenceRunner only calls prepare() / generate(path) / cleanup() —
it has zero knowledge of models, prompts, or generation mechanisms.
"""

from abc import ABC, abstractmethod

from config import MAX_NEW_TOKENS


class GenerationStrategy(ABC):
    """Encapsulates model loading and per-image generation.

    Subclasses implement:
      - prepare(device): load model, processor, and mode-specific context
      - generate(image_path) -> str: produce caption
      - cleanup(): release GPU memory
    """

    @property
    def label(self) -> str:
        """Human-readable label for progress display (e.g. "llava", "qwen2vl/k=1")."""
        return self.__class__.__name__

    def prepare(self, device: str):
        """Load model and mode-specific context. Called once before the loop."""
        pass

    @abstractmethod
    def generate(self, image_path: str) -> str:
        """Generate a caption for a single image."""
        ...

    def cleanup(self):
        """Release GPU resources. Called once after the loop."""
        import torch
        torch.cuda.empty_cache()


# ── Zero-Shot ──

class ZeroShotStrategy(GenerationStrategy):
    """Loads a model wrapper and generates with a fixed prompt."""

    def __init__(self, model_name: str, prompt: str,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self._model_name = model_name
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._wrapper = None

    @property
    def label(self) -> str:
        return self._model_name

    def prepare(self, device: str):
        from models import get_wrapper
        self._wrapper = get_wrapper(self._model_name)
        self._wrapper.load(device=device)

    def generate(self, image_path: str) -> str:
        return self._wrapper.generate(
            image_path, self._prompt, max_new_tokens=self._max_new_tokens
        )

    def cleanup(self):
        if self._wrapper is not None:
            del self._wrapper
            self._wrapper = None
        super().cleanup()


# ── Few-Shot ──

class FewShotStrategy(GenerationStrategy):
    """Loads a model wrapper with few-shot examples and generates."""

    def __init__(self, model_name: str, prompt_template: str, k: int,
                 example_images: list, example_captions: list,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self._model_name = model_name
        self._prompt_template = prompt_template
        self._k = k
        self._example_images = example_images
        self._example_captions = example_captions
        self._max_new_tokens = max_new_tokens
        self._wrapper = None

    @property
    def label(self) -> str:
        return f"{self._model_name}/k={self._k}"

    def prepare(self, device: str):
        from models import get_wrapper
        self._wrapper = get_wrapper(self._model_name)
        self._wrapper.load(device=device)

    def generate(self, image_path: str) -> str:
        return self._wrapper.generate_fewshot(
            test_image_path=image_path,
            prompt_template=self._prompt_template,
            example_images=self._example_images,
            example_captions=self._example_captions,
            max_new_tokens=self._max_new_tokens,
        )

    def cleanup(self):
        if self._wrapper is not None:
            del self._wrapper
            self._wrapper = None
        super().cleanup()


# ── LoRA ──

class LoRAStrategy(GenerationStrategy):
    """Loads a QLoRA-adapted model and generates with a fixed prompt."""

    def __init__(self, model_name: str, lora_dir: str, prompt: str,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self._model_name = model_name
        self._lora_dir = lora_dir
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._wrapper = None

    @property
    def label(self) -> str:
        return self._model_name

    def prepare(self, device: str):
        from models import get_wrapper
        self._wrapper = get_wrapper(self._model_name)
        if not self._wrapper.supports_lora:
            raise ValueError(
                f"Model '{self._model_name}' does not support LoRA inference"
            )
        self._wrapper.load_lora(self._lora_dir, device=device)

    def generate(self, image_path: str) -> str:
        return self._wrapper.generate(
            image_path, self._prompt, max_new_tokens=self._max_new_tokens
        )

    def cleanup(self):
        if self._wrapper is not None:
            del self._wrapper
            self._wrapper = None
        super().cleanup()
