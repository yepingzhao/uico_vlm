"""TrainingModelAdapter — encapsulates model-specific behavior for QLoRA training.

Parallel to core/inference/strategies.py:GenerationStrategy on the inference side.
Each concrete adapter handles: processor loading, post-load setup, dataset
configuration, training forward routing, and validation inference.

Model families:
  Standard HF pipeline — LLaVA, LLaVA-NeXT, Qwen2VL, Qwen3VL
  InternVL pipeline — InternVL2/3/3.5 (separate tokenizer + image processor)
  Phi-3.5 pipeline — <|image_1|> tag, chat_template on processor
"""

from abc import ABC, abstractmethod

import torch
from PIL import Image


class TrainingModelAdapter(ABC):
    """Encapsulates model-specific behaviour for the QLoRA training loop.

    Subclasses implement the five extension points.  The TrainingRunner
    owns the loop; it delegates every model-specific decision to the adapter.
    """

    # -- Processor -------------------------------------------------------

    @abstractmethod
    def load_processor(self, model_id: str, model_cfg: dict) -> dict:
        """Load and return processor(s) for this model.

        Returns:
            dict with keys:
              - ``processor``: HF processor / tokenizer
              - ``image_processor``: CLIPImageProcessor | None
        """
        ...

    # -- Post-load hook --------------------------------------------------

    def setup_post_load(self, model, processor, image_processor, device: str):
        """Model-specific setup after QLoRA model + processor are loaded.

        Called once, before the training loop.  Default: no-op.
        Subclasses override for InternVL img_context_token_id, etc.
        """
        pass

    # -- Dataset ---------------------------------------------------------

    def get_dataset_kwargs(self) -> dict:
        """Extra kwargs forwarded to UICOInstructionDataset(...).

        Default: no extra flags (standard HF pipeline).
        """
        return {}

    # -- Forward pass routing --------------------------------------------

    @property
    def use_base_model_forward(self) -> bool:
        """Should the training loop call ``model.base_model(**kwargs)``?

        InternVL-family models route through PeftModel.base_model
        (= LoraModel) because PeftModelForCausalLM.forward() injects
        inputs_embeds that InternVLChatModel does not accept.
        """
        return False

    # -- Validation inference --------------------------------------------

    @abstractmethod
    def validation_generate(
        self, model, processor, image_processor,
        image: Image.Image, prompt: str, device: str,
    ) -> str:
        """Generate a caption for one validation image.

        Used by the training loop for mid-training mode-collapse detection.
        ``model`` is the QLoRA-adapted model (in eval mode when called).
        ``processor`` may be a tokenizer (InternVL) or a standard HF processor.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
#  Standard HF VLM family
# ═══════════════════════════════════════════════════════════════════════

class StandardAdapter(TrainingModelAdapter):
    """Baseline adapter for models using the standard HF processor pipeline.

    Loads an AutoProcessor, uses structured-content chat templates, and
    calls ``model(**kwargs)`` for both training and validation.
    """

    def __init__(self, processor_kwargs: dict | None = None):
        self._processor_kwargs = processor_kwargs or {}

    def load_processor(self, model_id: str, model_cfg: dict) -> dict:
        from transformers import AutoProcessor

        kwargs = dict(self._processor_kwargs)
        if model_cfg.get("model_kwargs", {}).get("local_files_only"):
            kwargs["local_files_only"] = True
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            **kwargs,
        )
        return {"processor": processor, "image_processor": None}

    def validation_generate(
        self, model, processor, image_processor,
        image: Image.Image, prompt: str, device: str,
    ) -> str:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        inputs = processor(
            images=image, text=text_prompt, return_tensors="pt",
        ).to(device, torch.bfloat16)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=128, do_sample=False)
        generated_ids = output_ids[:, input_len:]
        tok = (
            processor.tokenizer if hasattr(processor, "tokenizer")
            else processor
        )
        return tok.decode(generated_ids[0], skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════
#  LLaVA-NeXT
# ═══════════════════════════════════════════════════════════════════════

class LLaVANeXTAdapter(StandardAdapter):
    """LLaVA-NeXT dynamic high resolution → smaller image to fit 24 GB."""

    def __init__(self):
        super().__init__(
            processor_kwargs=dict(
                size={"shortest_edge": 168, "longest_edge": 336},
            )
        )


# ═══════════════════════════════════════════════════════════════════════
#  Qwen2VL / Qwen3VL
# ═══════════════════════════════════════════════════════════════════════

QWEN_LOW_RES_KWARGS = dict(
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)

class QwenVLAdapter(StandardAdapter):
    """Qwen2.5-VL / Qwen3-VL — low-res processor to avoid OOM on 24 GB."""

    def __init__(self):
        super().__init__(processor_kwargs=dict(QWEN_LOW_RES_KWARGS))


# ═══════════════════════════════════════════════════════════════════════
#  InternVL family
# ═══════════════════════════════════════════════════════════════════════

_INTERNVL_NUM_IMAGE_TOKENS = 256  # 448 px, patch=14, downsample=0.5


class InternVLAdapter(TrainingModelAdapter):
    """Adapter for InternVL2, InternVL3, and InternVL3.5.

    These models use a separate AutoTokenizer + AutoImageProcessor
    (AutoProcessor returns only the tokenizer), route training forward
    through ``model.base_model()``, and use ``<img><IMG_CONTEXT>`` token
    expansion for validation.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    # -- Processor -------------------------------------------------------

    def load_processor(self, model_id: str, model_cfg: dict) -> dict:
        from transformers import AutoTokenizer, AutoImageProcessor

        extra = {}
        if model_cfg.get("model_kwargs", {}).get("local_files_only"):
            extra["local_files_only"] = True
        trust = model_cfg.get("trust_remote_code", False)

        # InternVL3.5: force-load tokenizer directly (InternVLProcessor
        # crashes on TokenizersBackend — missing start_image_token etc.)
        if self._model_name == "internvl35":
            processor = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust, **extra)
        else:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust, **extra)

        image_processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=trust, **extra)

        # InternVL3.5: force single-patch mode for training stability.
        # Dynamic patches produce variable-length IMG_CONTEXT (256–3072
        # tokens) → extreme sequence-length variation → QLoRA instability.
        if self._model_name == "internvl35":
            image_processor.max_patches = 1

        return {"processor": processor, "image_processor": image_processor}

    # -- Post-load -------------------------------------------------------

    def setup_post_load(self, model, processor, image_processor, device: str):
        """Set img_context_token_id on the innermost InternVLChatModel."""
        img_ctx = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        inner = model
        for step in ("base_model", "model"):
            inner = getattr(inner, step, inner)
        if hasattr(inner, "img_context_token_id"):
            inner.img_context_token_id = img_ctx
            print(f"[InternVL] img_context_token_id={img_ctx} "
                  f"set on {type(inner).__name__}")

    # -- Dataset ---------------------------------------------------------

    def get_dataset_kwargs(self) -> dict:
        return {
            "is_internvl2": True,
            "num_image_tokens": _INTERNVL_NUM_IMAGE_TOKENS,
        }

    # -- Forward ---------------------------------------------------------

    @property
    def use_base_model_forward(self) -> bool:
        return True

    # -- Validation ------------------------------------------------------

    def validation_generate(
        self, model, processor, image_processor,
        image: Image.Image, prompt: str, device: str,
    ) -> str:
        # InternVL2/3/3.5: expand <image> → <img><IMG_CONTEXT>×256</img>
        # BEFORE apply_chat_template (which returns token IDs, not text).
        img_tokens = (
            f"<img>{'<IMG_CONTEXT>' * _INTERNVL_NUM_IMAGE_TOKENS}</img>"
        )
        conv = [{"role": "user", "content": f"{img_tokens}\n{prompt}"}]
        result = processor.apply_chat_template(
            conv, add_generation_prompt=True)

        # Normalize: output may be BatchEncoding / dict / list
        if hasattr(result, "data") and "input_ids" in result.data:
            ids_list = result.data["input_ids"]
        elif isinstance(result, dict) and "input_ids" in result:
            ids_list = result["input_ids"]
        elif hasattr(result, "ids"):
            ids_list = result.ids
        else:
            ids_list = result
        if (isinstance(ids_list, list) and len(ids_list) > 0
                and isinstance(ids_list[0], list)):
            ids_list = ids_list[0]

        input_ids = torch.tensor([ids_list], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids)
        img_outputs = image_processor(images=image, return_tensors="pt")
        pixel_values = img_outputs["pixel_values"].to(
            device, torch.bfloat16)

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
            )
        # InternVL.generate() returns ONLY generated tokens
        tok = (
            processor.tokenizer if hasattr(processor, "tokenizer")
            else processor
        )
        return tok.decode(
            output_ids[0], skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════
#  Phi-3.5-Vision
# ═══════════════════════════════════════════════════════════════════════

class Phi35Adapter(TrainingModelAdapter):
    """Phi-3.5-Vision adapter.

    Phi3VProcessor lacks a chat_template attribute (it lives on the
    tokenizer).  We copy it so ``apply_chat_template`` works.  The chat
    template uses ``<|image_1|>`` as the image placeholder.
    """

    def load_processor(self, model_id: str, model_cfg: dict) -> dict:
        # Phi-3.5-Vision + transformers >= 4.49: DynamicCache.get_max_length()
        # is called by the vendored modeling_phi3_v.py but does not exist on
        # the upstream class.  Patch once on first use.
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None

        from transformers import AutoProcessor

        kwargs = {}
        if model_cfg.get("model_kwargs", {}).get("local_files_only"):
            kwargs["local_files_only"] = True
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            **kwargs,
        )
        # Make chat_template accessible at processor level
        processor.chat_template = processor.tokenizer.chat_template
        return {"processor": processor, "image_processor": None}

    def get_dataset_kwargs(self) -> dict:
        return {"is_phi35": True}

    def validation_generate(
        self, model, processor, image_processor,
        image: Image.Image, prompt: str, device: str,
    ) -> str:
        conv = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        text_prompt = processor.apply_chat_template(
            conv, add_generation_prompt=True)
        inputs = processor(
            images=image, text=text_prompt, return_tensors="pt",
        ).to(device, torch.bfloat16)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=128, do_sample=False)
        generated_ids = output_ids[:, input_len:]
        tok = (
            processor.tokenizer if hasattr(processor, "tokenizer")
            else processor
        )
        return tok.decode(generated_ids[0], skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════

_ADAPTER_REGISTRY = {
    "llava": StandardAdapter(),
    "llava-next": LLaVANeXTAdapter(),
    "qwen2vl": QwenVLAdapter(),
    "qwen3vl": QwenVLAdapter(),
    "internvl2": InternVLAdapter("internvl2"),
    "internvl3": InternVLAdapter("internvl3"),
    "internvl35": InternVLAdapter("internvl35"),
    "phi35-vision": Phi35Adapter(),
}


def get_training_adapter(model_name: str) -> TrainingModelAdapter:
    """Resolve a model short-name to its TrainingModelAdapter."""
    if model_name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown or unsupported model for LoRA training: {model_name}. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return _ADAPTER_REGISTRY[model_name]
