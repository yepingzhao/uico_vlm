"""Abstract base class for VLM wrappers."""

from abc import ABC, abstractmethod
import os
import torch


class VLMWrapper(ABC):
    """Unified interface for zero-shot VLM inference."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cuda:0"
        # Separate image processor for wrappers that don't use unified
        # processor (InternVL family).
        self._image_processor = None

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

    def _strip_and_decode(self, output_ids, inputs, processor=None) -> str:
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

    # --- LoRA support ---

    _lora_config_key: str = None
    """Set in subclasses that support LoRA (key into MODEL_LORA_CONFIGS)."""

    @property
    def supports_lora(self) -> bool:
        """True if this wrapper can load LoRA adapters for inference."""
        return self._lora_config_key is not None

    def load_lora(self, lora_dir: str, device: str = "cuda:0"):
        """Load 4-bit quantized base model + LoRA adapters for inference.

        Default implementation uses load_qlora_for_inference from models/lora.py.
        Wrappers with special processor requirements (InternVL family) override
        this method.

        After loading, self._model is a PeftModel and the existing generate()
        method works unchanged for standard models.
        """
        from config.training import get_lora_config
        from models.lora import load_qlora_for_inference

        cfg = get_lora_config(self._lora_config_key)
        model_id = cfg["model_id"]
        class_name = cfg["model_class_name"]
        trust_remote_code = cfg.get("trust_remote_code", False)
        model_kwargs = cfg.get("model_kwargs")

        # Resolve model class (all MODEL_LORA_CONFIGS entries use classes
        # that exist in transformers, so hasattr is always True)
        import transformers
        model_class = getattr(transformers, class_name)

        self._model, self._processor = load_qlora_for_inference(
            model_class, model_id, lora_dir, device,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
        )
        self._device = device

    # --- Few-shot support (Template Method) ---

    _fewshot_embed_images: bool = False
    """Set True in Qwen-family wrappers to embed PIL images in content blocks."""

    @property
    def supports_fewshot(self) -> bool:
        """True if this wrapper overrides _build_fewshot_inputs."""
        return type(self)._build_fewshot_inputs is not VLMWrapper._build_fewshot_inputs

    def _get_fewshot_processor(self):
        """Return the processor for few-shot inference.

        Override in Qwen2VL to return a low-resolution processor to avoid OOM.
        """
        return self._processor

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """Build tokenized model inputs from few-shot content blocks and images.

        Subclasses MUST override this — the implementation varies by model family
        (LLaVA vs Qwen processor APIs differ).
        """
        raise NotImplementedError(
            f"{self.model_name} does not support few-shot inference"
        )

    def generate_fewshot(
        self,
        test_image_path: str,
        prompt_template: str,
        example_images: list,
        example_captions: list,
        **kwargs,
    ) -> str:
        """Generate a caption using few-shot in-context examples.

        Args:
            test_image_path: Path to the test image.
            prompt_template: Text prompt placed after the examples.
            example_images: List of paths to example images.
            example_captions: List of example captions (same length as images).

        Returns:
            Generated caption string.
        """
        from models.fewshot.content import build_fewshot_images_and_content

        all_images, content_blocks = build_fewshot_images_and_content(
            test_image_path, prompt_template, example_images, example_captions,
            embed_images=self._fewshot_embed_images,
        )

        processor = self._get_fewshot_processor()
        inputs = self._build_fewshot_inputs(content_blocks, all_images)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs, processor=processor)
