# Design: Eliminate `fewshot/` / `models/` Duplication

**Date**: 2026-06-04
**Branch**: `refactor/eliminate-model-wrapper-duplication`
**Status**: approved

## Summary

Eliminate duplicate functionality between `fewshot/` and `models/` directories:

1. Move `models/_fewshot.py` → `fewshot/content.py`
2. Lift `generate_fewshot()` boilerplate into `VLMWrapper` base class via Template Method
3. Centralize the few-shot prompt into `prompts/templates.py`
4. Deduplicate image path resolution between `sampler.py` and `dataset.py`
5. Auto-discover few-shot-capable models instead of hardcoding the list

## Motivation

Five concrete duplications were identified:

| # | Duplication | Location |
|---|------------|----------|
| 1 | `generate_fewshot()` boilerplate (~15 lines × 4 wrappers, near-identical) | `llava.py`, `llava_next.py`, `qwen2vl.py`, `qwen3vl.py` |
| 2 | `FEWSHOT_MODELS` hardcoded list (stale: missing `qwen3vl`) | `eval_fewshot.py:27` |
| 3 | `FEWSHOT_PROMPT` inlined in script instead of in prompt module | `run_fewshot.py:39-42` |
| 4 | `resolve_image_path()` duplicates `data/dataset.py` prefix→subdir logic | `sampler.py:41-53` |
| 5 | `_fewshot.py` lives under `models/` but is a shared utility — wrong home | `models/_fewshot.py` |

## Design

### 1. Module Reorganization

```
fewshot/
  __init__.py      # unchanged
  sampler.py       # unchanged (example selection from training data)
  content.py       # NEW — moved from models/_fewshot.py, drop _ prefix
models/
  _fewshot.py      # DELETED
```

Import changes in wrappers: `from ._fewshot import ...` → `from fewshot.content import ...`

Rationale: `build_fewshot_images_and_content` is a content-construction utility used by model wrappers. It belongs in the `fewshot/` package alongside `sampler.py`, not under `models/`.

### 2. Template Method for `generate_fewshot()`

The four `generate_fewshot()` implementations differ in only two axes:

| Axis | LLaVA family | Qwen family |
|------|-------------|-------------|
| `embed_images` | `False` | `True` |
| Processor API style | `processor(images=..., text=...)` | `processor(text=[text], images=..., padding=True)` |

**Solution:** Provide default `generate_fewshot()` in `VLMWrapper` base class. Subclasses override a single method `_build_fewshot_inputs(content_blocks, all_images)` plus declarative class attributes.

Base class additions in `models/base.py`:

```python
class VLMWrapper(ABC):
    _fewshot_embed_images: bool = False

    @property
    def supports_fewshot(self) -> bool:
        return type(self)._build_fewshot_inputs is not VLMWrapper._build_fewshot_inputs

    def _get_fewshot_processor(self):
        return self._processor

    def _build_fewshot_inputs(self, content_blocks, all_images):
        raise NotImplementedError(
            f"{self.model_name} does not support few-shot inference"
        )

    def generate_fewshot(self, test_image_path, prompt_template,
                         example_images, example_captions, **kwargs) -> str:
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

Wrapper changes (LLaVA family — applies to both `llava.py` and `llava_next.py`):

```python
class LLaVAWrapper(VLMWrapper):
    # delete: generate_fewshot()  (~30 lines removed)

    def _build_fewshot_inputs(self, content_blocks, all_images):
        conversation = [{"role": "user", "content": content_blocks}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self._processor(
            images=all_images, text=formatted, return_tensors="pt"
        ).to(self._device, torch.float16)
```

Wrapper changes (Qwen2VL — `qwen2vl.py`):

```python
class Qwen2VLWrapper(VLMWrapper):
    _fewshot_embed_images = True

    # delete: generate_fewshot()  (~30 lines removed)

    def _get_fewshot_processor(self):
        return self._fewshot_processor  # low-res processor for OOM avoidance

    def _build_fewshot_inputs(self, content_blocks, all_images):
        processor = self._get_fewshot_processor()
        messages = [{"role": "user", "content": content_blocks}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
```

Wrapper changes (Qwen3VL — `qwen3vl.py`):

```python
class Qwen3VLWrapper(VLMWrapper):
    _fewshot_embed_images = True

    # delete: generate_fewshot()  (~30 lines removed)
    # _get_fewshot_processor() not overridden — uses self._processor

    def _build_fewshot_inputs(self, content_blocks, all_images):
        messages = [{"role": "user", "content": content_blocks}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(
            text=[text], images=all_images, return_tensors="pt", padding=True
        ).to(self._device)
```

Net effect: ~60 lines of boilerplate removed from wrappers, replaced by ~35 lines in base class + ~10 lines per wrapper for `_build_fewshot_inputs`.

### 3. Prompt Centralization

Add to `prompts/templates.py`:

```python
PROMPT_FEWSHOT = (
    "Now describe any urban incivility or civic norm violations "
    "visible in the image above in one or two sentences."
)

PROMPT_MAP = {
    ...
    "FS": PROMPT_FEWSHOT,
}
```

`run_fewshot.py` switches from inline `FEWSHOT_PROMPT` to `from prompts.templates import PROMPT_FEWSHOT`.

### 4. Path Resolution Dedup

`fewshot/sampler.py:resolve_image_path()` duplicates `data/dataset.py:_resolve_image_path()`. Remove the `sampler.py` copy and import from `data.dataset`:

```python
# sampler.py — replace resolve_image_path() with:
from data.dataset import _resolve_image_path as resolve_image_path
```

Or, for a cleaner public API, rename `_resolve_image_path` → `resolve_image_path` in `dataset.py` (drop the underscore) since it's now used by two modules.

### 5. Auto-Discover Few-Shot Models

`eval_fewshot.py` hardcodes `FEWSHOT_MODELS = ["llava", "qwen2vl", "llava-next"]`. Replace with:

```python
def _get_fewshot_models():
    """Discover which registered models support few-shot inference."""
    from models import get_wrapper
    from models import get_wrapper as _gw
    # Query all known model names; keep those that implement generate_fewshot
    candidates = ["llava", "llava-next", "qwen2vl", "qwen3vl"]
    available = []
    for name in candidates:
        try:
            w = _gw(name)
            if w.supports_fewshot:
                available.append(name)
        except (ValueError, ImportError):
            pass
    return available
```

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `models/_fewshot.py` | DELETE | Moved to `fewshot/content.py` |
| `fewshot/content.py` | CREATE | Content from old `_fewshot.py` |
| `models/base.py` | UPDATE | Add `generate_fewshot()`, `supports_fewshot`, hooks |
| `models/llava.py` | UPDATE | Delete `generate_fewshot()`, add `_build_fewshot_inputs()` |
| `models/llava_next.py` | UPDATE | Same as llava.py |
| `models/qwen2vl.py` | UPDATE | Delete `generate_fewshot()`, add hooks + `_build_fewshot_inputs()` |
| `models/qwen3vl.py` | UPDATE | Same pattern |
| `prompts/templates.py` | UPDATE | Add `PROMPT_FEWSHOT` |
| `fewshot/sampler.py` | UPDATE | Reuse `data.dataset` path resolution |
| `data/dataset.py` | UPDATE | Make `_resolve_image_path` public |
| `scripts/run_fewshot.py` | UPDATE | Import prompt from templates |
| `scripts/eval_fewshot.py` | UPDATE | Auto-discover few-shot models |

## Validation

```bash
# Zero-shot still works (regression check)
python scripts/run_inference.py --models llava qwen2vl --subsample 10 --prompt A

# Few-shot still works
python scripts/run_fewshot.py --models llava qwen2vl qwen3vl --k 1 --subsample 10

# Few-shot eval works with auto-discovery
python scripts/eval_fewshot.py --all --ref_free_only
```

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `generate_fewshot()` base class refactor breaks subtle model-specific behavior | Low | Identical logic, only moved. Validation run on 4 models catches regressions. |
| Import path changes break other scripts | Low | Grep for `_fewshot` references; only used in 4 wrapper files. |
| `supports_fewshot` property false positives | Low | Checks method override explicitly, not a flag that can be set accidentally. |
