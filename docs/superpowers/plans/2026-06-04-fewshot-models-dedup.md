# Fewshot/Models Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicate functionality between `fewshot/` and `models/` directories by moving `_fewshot.py` into `fewshot/`, lifting `generate_fewshot()` boilerplate into the base class, centralizing the few-shot prompt, deduplicating path resolution, and auto-discovering few-shot models.

**Architecture:** Five independent changes: (1) module move `_fewshot.py` → `fewshot/content.py`, (2) Template Method pattern in `VLMWrapper` base class to eliminate ~60 lines of wrapper boilerplate, (3) prompt text moved to `prompts/templates.py`, (4) `sampler.py` reuses `dataset.py` path resolution, (5) `eval_fewshot.py` discovers models via `supports_fewshot` property instead of hardcoded list.

**Tech Stack:** Python 3.x, PyTorch, HuggingFace Transformers

---

### Task 1: Create `fewshot/content.py` from `models/_fewshot.py`

**Files:**
- Create: `fewshot/content.py`
- Modify: `fewshot/__init__.py`

- [ ] **Step 1: Copy `_fewshot.py` to `fewshot/content.py` with updated docstring**

```python
"""Shared few-shot content-block construction logic.

Four models support few-shot (LLaVA, LLaVA-NeXT, Qwen2VL, Qwen3VL).
They all interleave example images and captions the same way but
differ in how images are attached to content blocks and how the
processor is invoked.

This module provides:
  - build_fewshot_images_and_content: builds (all_images, content_blocks)
"""

from PIL import Image


def build_fewshot_images_and_content(
    test_image_path: str,
    prompt_template: str,
    example_images: list,
    example_captions: list,
    *,
    embed_images: bool = False,
):
    """Build (all_images, content_blocks) for few-shot inference.

    Args:
        test_image_path: Path to the test image.
        prompt_template: Text prompt with the instruction.
        example_images: List of paths to example images.
        example_captions: List of example captions (same length).
        embed_images: If True, embed PIL Image objects in content blocks
                      (Qwen-style). If False, use {"type": "image"} placeholder
                      (LLaVA-style), with images passed separately.

    Returns:
        Tuple of (all_images: list[Image], content_blocks: list[dict]).
    """
    all_images = []
    content_blocks = []

    for i, (ex_img_path, ex_caption) in enumerate(
        zip(example_images, example_captions)
    ):
        ex_img = Image.open(ex_img_path).convert("RGB")
        all_images.append(ex_img)
        if embed_images:
            content_blocks.append({"type": "image", "image": ex_img})
        else:
            content_blocks.append({"type": "image"})
        content_blocks.append({
            "type": "text",
            "text": f"Example {i + 1}: {ex_caption}",
        })

    test_img = Image.open(test_image_path).convert("RGB")
    all_images.append(test_img)
    if embed_images:
        content_blocks.append({"type": "image", "image": test_img})
    else:
        content_blocks.append({"type": "image"})
    content_blocks.append({
        "type": "text",
        "text": prompt_template,
    })

    return all_images, content_blocks
```

- [ ] **Step 2: Update `fewshot/__init__.py` docstring**

Replace existing content with:

```python
"""Few-shot in-context learning for VLM evaluation.

Modules:
  - sampler: example selection from the training set
  - content: shared content-block construction for few-shot prompts
"""
```

- [ ] **Step 3: Commit**

```bash
git add fewshot/content.py fewshot/__init__.py
git commit -m "refactor: move _fewshot.py content to fewshot/content.py"
```

---

### Task 2: Centralize few-shot prompt into `prompts/templates.py`

**Files:**
- Modify: `prompts/templates.py`

- [ ] **Step 1: Add `PROMPT_FEWSHOT` constant**

Add after `PROMPT_ZH` (before `PROMPT_MAP`):

```python
# Few-shot prompt: instruction placed after example images
PROMPT_FEWSHOT = (
    "Now describe any urban incivility or civic norm violations "
    "visible in the image above in one or two sentences."
)
```

- [ ] **Step 2: Add `PROMPT_FEWSHOT` to `PROMPT_MAP`**

```python
PROMPT_MAP = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
    "ZH": PROMPT_ZH,
    "FS": PROMPT_FEWSHOT,
}
```

- [ ] **Step 3: Commit**

```bash
git add prompts/templates.py
git commit -m "refactor: centralize few-shot prompt into prompts/templates.py"
```

---

### Task 3: Make path resolution in `data/dataset.py` reusable

**Files:**
- Modify: `data/dataset.py`

- [ ] **Step 1: Rename `_resolve_image_path` to `resolve_image_path`**

Rename the function (drop the leading underscore) so it's part of the public API:

```python
def resolve_image_path(images_base_dir: str, file_name: str) -> str:
    """Map a COCO file_name to its actual filesystem path."""
    # file_name example: "CCMC_train_000000026952.jpg"
    parts = file_name.rsplit("_", 2)  # ["CCMC", "train", "000000026952.jpg"]
    if len(parts) >= 2:
        prefix = parts[0] + "_" + parts[1]  # "CCMC_train"
    else:
        prefix = file_name.split("_")[0] + "_" + file_name.split("_")[1]

    subdir = PREFIX_TO_SUBDIR.get(prefix, "ccmc_test")
    return os.path.join(images_base_dir, subdir, file_name)
```

- [ ] **Step 2: Update the internal call site in `UICOTestDataset.get_image_path`**

```python
def get_image_path(self, image_id: int) -> str:
    return resolve_image_path(self.images_base_dir, self._img_id_to_file[image_id])
```

- [ ] **Step 3: Commit**

```bash
git add data/dataset.py
git commit -m "refactor: make resolve_image_path public in data/dataset.py"
```

---

### Task 4: Deduplicate path resolution in `fewshot/sampler.py`

**Files:**
- Modify: `fewshot/sampler.py`

- [ ] **Step 1: Replace `resolve_image_path` function with import**

Delete the `resolve_image_path` function (lines 41-53) and add this import at the top:

```python
from data.dataset import resolve_image_path
```

The final imports in `sampler.py` should be:

```python
"""Sample few-shot examples from the UICO training set.

For reproducibility, we sample once with a fixed seed and reuse the same
examples for all test images (static-example approach).
"""

import json
import os
import random
from typing import List, Tuple

from config import DATA_BASE, RANDOM_SEED
from data.dataset import resolve_image_path
```

- [ ] **Step 2: Commit**

```bash
git add fewshot/sampler.py
git commit -m "refactor: reuse data.dataset.resolve_image_path in sampler.py"
```

---

### Task 5: Add Template Method to `VLMWrapper` base class

**Files:**
- Modify: `models/base.py`

- [ ] **Step 1: Add `torch` import**

Add `import torch` after the existing imports:

```python
"""Abstract base class for VLM wrappers."""

from abc import ABC, abstractmethod
import os
import torch
```

- [ ] **Step 2: Add few-shot hooks and `generate_fewshot()` to `VLMWrapper`**

Add the following methods after `_strip_and_decode` (before the end of the class):

```python
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
        from fewshot.content import build_fewshot_images_and_content

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
```

- [ ] **Step 3: Commit**

```bash
git add models/base.py
git commit -m "refactor: add generate_fewshot Template Method to VLMWrapper base"
```

---

### Task 6: Simplify `llava.py` — delete `generate_fewshot()`, add `_build_fewshot_inputs()`

**Files:**
- Modify: `models/llava.py`

- [ ] **Step 1: Delete `generate_fewshot()` method and add `_build_fewshot_inputs()`**

Replace the `generate_fewshot` method (lines 55-86) with:

```python
    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """LLaVA-style: images passed separately, not embedded in content."""
        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self._processor(
            images=all_images, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)
```

- [ ] **Step 2: Verify the final file**

The complete `llava.py` should be:

```python
"""LLaVA-1.5 Vicuna-7B wrapper."""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import VLMWrapper


class LLaVAWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "llava"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "llava-hf/llava-1.5-7b-hf"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        # Build conversation with system + user message
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=image, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        # Strip input tokens, keep only generated response
        return self._strip_and_decode(output_ids, inputs)

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """LLaVA-style: images passed separately, not embedded in content."""
        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self._processor(
            images=all_images, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)
```

- [ ] **Step 3: Commit**

```bash
git add models/llava.py
git commit -m "refactor: replace generate_fewshot with _build_fewshot_inputs in LLaVA"
```

---

### Task 7: Simplify `llava_next.py` — same pattern

**Files:**
- Modify: `models/llava_next.py`

- [ ] **Step 1: Delete `generate_fewshot()` method and add `_build_fewshot_inputs()`**

Replace the `generate_fewshot` method (lines 56-87) with:

```python
    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """LLaVA-style: images passed separately, not embedded in content."""
        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self._processor(
            images=all_images, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)
```

- [ ] **Step 2: Verify the final file**

The complete `llava_next.py` should be:

```python
"""LLaVA-NeXT (LLaVA-1.6) Mistral-7B wrapper."""

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .base import VLMWrapper


class LLaVANeXTWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "llava-next"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        self._processor = LlavaNextProcessor.from_pretrained(model_id)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        self._model.eval()

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=image, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """LLaVA-style: images passed separately, not embedded in content."""
        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self._processor(
            images=all_images, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)
```

- [ ] **Step 3: Commit**

```bash
git add models/llava_next.py
git commit -m "refactor: replace generate_fewshot with _build_fewshot_inputs in LLaVA-NeXT"
```

---

### Task 8: Simplify `qwen2vl.py` — delete `generate_fewshot()`, add hooks

**Files:**
- Modify: `models/qwen2vl.py`

- [ ] **Step 1: Delete `generate_fewshot()` and add `_build_fewshot_inputs()` + `_get_fewshot_processor()`**

Replace the `generate_fewshot` method (lines 69-103) with:

```python
    _fewshot_embed_images = True

    def _get_fewshot_processor(self):
        """Return low-res processor for few-shot to avoid OOM on 24GB GPUs."""
        return self._fewshot_processor

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """Qwen-style: embed PIL images in content, list-based processor call."""
        processor = self._get_fewshot_processor()
        messages = [{"role": "user", "content": content_blocks}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
```

- [ ] **Step 2: Verify the final file**

The complete `qwen2vl.py` should be:

```python
"""Qwen2.5-VL-7B-Instruct wrapper."""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .base import VLMWrapper


class Qwen2VLWrapper(VLMWrapper):

    def __init__(self):
        super().__init__()
        self._fewshot_processor = None

    @property
    def model_name(self) -> str:
        return "qwen2vl"

    def load(self, device: str = "cuda:0"):
        self._device = device
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self._model.eval()
        # min_pixels/max_pixels control resolution; set to reasonable defaults
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        # Low-res processor for few-shot multi-image (avoids OOM on 24GB)
        self._fewshot_processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        # Build chat messages following Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)

    _fewshot_embed_images = True

    def _get_fewshot_processor(self):
        """Return low-res processor for few-shot to avoid OOM on 24GB GPUs."""
        return self._fewshot_processor

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """Qwen-style: embed PIL images in content, list-based processor call."""
        processor = self._get_fewshot_processor()
        messages = [{"role": "user", "content": content_blocks}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
```

- [ ] **Step 3: Commit**

```bash
git add models/qwen2vl.py
git commit -m "refactor: replace generate_fewshot with hooks in Qwen2VL"
```

---

### Task 9: Simplify `qwen3vl.py` — delete `generate_fewshot()`, add hooks

**Files:**
- Modify: `models/qwen3vl.py`

- [ ] **Step 1: Delete `generate_fewshot()` and add `_build_fewshot_inputs()`**

Replace the `generate_fewshot` method (lines 65-97) with:

```python
    _fewshot_embed_images = True

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """Qwen-style: embed PIL images in content, list-based processor call."""
        messages = [{"role": "user", "content": content_blocks}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
```

- [ ] **Step 2: Verify the final file**

The complete `qwen3vl.py` should be:

```python
"""Qwen3-VL-8B-Instruct wrapper (2025)."""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import VLMWrapper


class Qwen3VLWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "qwen3vl"

    def load(self, device: str = "cuda:0"):
        self._device = device
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)
        image = Image.open(image_path).convert("RGB")

        # Build chat messages following Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        return self._strip_and_decode(output_ids, inputs)

    _fewshot_embed_images = True

    def _build_fewshot_inputs(self, content_blocks: list, all_images: list):
        """Qwen-style: embed PIL images in content, list-based processor call."""
        messages = [{"role": "user", "content": content_blocks}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
```

- [ ] **Step 3: Commit**

```bash
git add models/qwen3vl.py
git commit -m "refactor: replace generate_fewshot with hooks in Qwen3VL"
```

---

### Task 10: Update `scripts/run_fewshot.py` — import prompt from templates

**Files:**
- Modify: `scripts/run_fewshot.py`

- [ ] **Step 1: Add import of `PROMPT_FEWSHOT`**

Add `PROMPT_FEWSHOT` to the existing `from prompts.templates import ...` line (currently there's none — add a new import):

Add this line after the existing `from config import` block:

```python
from prompts.templates import PROMPT_FEWSHOT
```

- [ ] **Step 2: Delete the inline `FEWSHOT_PROMPT` constant**

Delete these lines:

```python
# Few-shot prompt template: k examples + instruction
FEWSHOT_PROMPT = (
    "Now describe any urban incivility or civic norm violations "
    "visible in the image above in one or two sentences."
)
```

- [ ] **Step 3: Replace `FEWSHOT_PROMPT` usage with `PROMPT_FEWSHOT`**

In the `run_fewshot` function, change:

```python
prompt_template=FEWSHOT_PROMPT,
```

to:

```python
prompt_template=PROMPT_FEWSHOT,
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_fewshot.py
git commit -m "refactor: use PROMPT_FEWSHOT from prompts.templates in run_fewshot.py"
```

---

### Task 11: Update `scripts/eval_fewshot.py` — auto-discover few-shot models

**Files:**
- Modify: `scripts/eval_fewshot.py`

- [ ] **Step 1: Replace hardcoded `FEWSHOT_MODELS` with discovery function**

Delete:

```python
# Models that support few-shot inference
FEWSHOT_MODELS = ["llava", "qwen2vl", "llava-next"]
FEWSHOT_K_VALUES = [1, 3, 5]
```

Replace with:

```python
FEWSHOT_K_VALUES = [1, 3, 5]


def _get_fewshot_models():
    """Discover which registered models support few-shot inference.

    Returns:
        List of model short names that implement generate_fewshot.
    """
    from models import get_wrapper

    candidates = ["llava", "llava-next", "qwen2vl", "qwen3vl"]
    available = []
    for name in candidates:
        try:
            wrapper = get_wrapper(name)
            if wrapper.supports_fewshot:
                available.append(name)
        except (ValueError, ImportError):
            pass
    return available
```

- [ ] **Step 2: Replace `FEWSHOT_MODELS` usage with `_get_fewshot_models()`**

In `__main__`, change:

```python
if args.all:
    combos = [(m, k) for m in FEWSHOT_MODELS for k in FEWSHOT_K_VALUES]
```

to:

```python
if args.all:
    fewshot_models = _get_fewshot_models()
    print(f"[Discover] Few-shot models: {fewshot_models}")
    combos = [(m, k) for m in fewshot_models for k in FEWSHOT_K_VALUES]
```

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_fewshot.py
git commit -m "refactor: auto-discover few-shot models via supports_fewshot property"
```

---

### Task 12: Delete `models/_fewshot.py`

**Files:**
- Delete: `models/_fewshot.py`

- [ ] **Step 1: Verify no remaining references to `_fewshot`**

```bash
grep -r "_fewshot" /home/uesr/zhaoyeping/workspace-code/uico_vlm/models/ /home/uesr/zhaoyeping/workspace-code/uico_vlm/scripts/ /home/uesr/zhaoyeping/workspace-code/uico_vlm/fewshot/
```

Expected: No matches (all wrappers now import from `fewshot.content`, scripts import from `prompts.templates`).

- [ ] **Step 2: Delete the file**

```bash
git rm models/_fewshot.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove models/_fewshot.py (moved to fewshot/content.py)"
```

---

### Task 13: Validation

**Files:**
- None (verification only)

- [ ] **Step 1: Verify imports work with a quick Python import check**

```bash
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm && python -c "
import sys; sys.path.insert(0, '.')
from fewshot.content import build_fewshot_images_and_content
from prompts.templates import PROMPT_FEWSHOT
from data.dataset import resolve_image_path
from models.base import VLMWrapper
print('All imports OK')
print('supports_fewshot is defined:', hasattr(VLMWrapper, 'supports_fewshot'))
print('generate_fewshot is defined:', hasattr(VLMWrapper, 'generate_fewshot'))
"
```

Expected output:
```
All imports OK
supports_fewshot is defined: True
generate_fewshot is defined: True
```

- [ ] **Step 2: Verify wrapper `supports_fewshot` and instantiation**

```bash
cd /home/uesr/zhaoyeping/workspace-code/uico_vlm && python -c "
import sys; sys.path.insert(0, '.')
from models import get_wrapper

for name in ['llava', 'llava-next', 'qwen2vl', 'qwen3vl', 'blip2']:
    w = get_wrapper(name)
    print(f'{name}: supports_fewshot={w.supports_fewshot}')
"
```

Expected output:
```
llava: supports_fewshot=True
llava-next: supports_fewshot=True
qwen2vl: supports_fewshot=True
qwen3vl: supports_fewshot=True
blip2: supports_fewshot=False
```

- [ ] **Step 3: Run zero-shot regression test (requires GPU)**

```bash
python scripts/run_inference.py --models llava --subsample 5 --prompt A
```

Expected: completes without errors, generates `outputs/llava/predictions_prompt_a.jsonl`.

- [ ] **Step 4: Run few-shot test (requires GPU)**

```bash
python scripts/run_fewshot.py --models llava --k 1 --subsample 5
```

Expected: completes without errors, generates `outputs/llava/predictions_fewshot_k1.jsonl`.

- [ ] **Step 5: Run few-shot eval test**

```bash
python scripts/eval_fewshot.py --model llava --k 1
```

Expected: shows discovered models, computes metrics without errors.

- [ ] **Step 6: Tag validation commit**

```bash
git add -A
git diff --staged --stat  # should be empty if all previous commits captured changes
```

---

## Validation Summary

```bash
# Quick smoke test (no GPU needed)
python -c "from fewshot.content import build_fewshot_images_and_content; from prompts.templates import PROMPT_FEWSHOT; from data.dataset import resolve_image_path; print('OK')"

# Full validation (requires GPU)
python scripts/run_inference.py --models llava --subsample 5 --prompt A
python scripts/run_fewshot.py --models llava qwen2vl --k 1 --subsample 5
python scripts/eval_fewshot.py --all --ref_free_only
```
