# Merge Inference Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `scripts/run_zeroshot.py` and `scripts/run_fewshot.py` into a single `scripts/run_inference.py` with a `--mode` flag.

**Architecture:** Single script with `--mode zeroshot|fewshot` dispatch. Zeroshot mode loads `ZeroShotStrategy` + `PROMPT_MAP`; fewshot mode pre-samples examples via `sample_examples()`, then loads `FewShotStrategy`. Both share the existing `DatasetBundle` → `InferenceRunner` pipeline without modifying any `common/` modules.

**Tech Stack:** Python 3.10, argparse, existing `common/strategies.py` and `common/pipeline.py`

---

## Files

| File | Action |
|------|--------|
| `scripts/run_inference.py` | CREATE |
| `scripts/run_zeroshot.py` | DELETE |
| `scripts/run_fewshot.py` | DELETE |
| `CLAUDE.md` | UPDATE |
| `README.md` | UPDATE |

---

### Task 1: Create `scripts/run_inference.py`

**Files:**
- Create: `scripts/run_inference.py`

- [ ] **Step 1: Write the merged script**

```python
#!/usr/bin/env python3
"""Run VLM inference on UICO test set with checkpoint/resume support.

Usage:
    # Zero-shot
    python scripts/run_inference.py --mode zeroshot --models llava --prompt A
    python scripts/run_inference.py --mode zeroshot --models llava qwen2vl --prompt B
    python scripts/run_inference.py --mode zeroshot --models llava --prompt A --overwrite

    # Few-shot
    python scripts/run_inference.py --mode fewshot --models llava --k 1 --subsample 3
    python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5
    python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5 --subsample 500
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, RANDOM_SEED, MAX_NEW_TOKENS
from data.dataset import load_test_dataset
from common.dataset_bundle import DatasetBundle
from common.strategies import ZeroShotStrategy, FewShotStrategy
from common.pipeline import InferenceRunner
from config.prompts import PROMPT_MAP, PROMPT_FEWSHOT


def run_zeroshot(args):
    """Zero-shot inference across one or more models."""
    prompt_text = PROMPT_MAP[args.prompt]
    print(f"[Config] mode=zeroshot, models={args.models}, prompt={args.prompt}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")
    print(f"[Prompt] {args.prompt}: {prompt_text[:100]}...")

    ds = load_test_dataset(subsample=args.subsample, seed=RANDOM_SEED)
    bundle = DatasetBundle.from_dataset(ds)
    print(f"[Data] Loaded {len(bundle)} images")

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"[Model] {model_name}")
        print(f"{'='*60}")

        model_out_dir = os.path.join(OUTPUT_DIR, model_name)
        filename = f"predictions_prompt_{args.prompt.lower()}.jsonl"

        strategy = ZeroShotStrategy(
            model_name=model_name,
            prompt=prompt_text,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        runner = InferenceRunner(
            strategy=strategy,
            output_dir=model_out_dir,
            filename=filename,
            bundle=bundle,
            prompt_label=args.prompt,
        )
        runner.run(overwrite=args.overwrite, device=args.device)


def run_fewshot(args):
    """Few-shot inference across one or more models and k values."""
    print(f"[Config] mode=fewshot, models={args.models}, k={args.k}, "
          f"subsample={args.subsample or 'full'}, device={args.device}"
          f"{', overwrite' if args.overwrite else ''}")

    from models.fewshot import sample_examples

    ds = load_test_dataset(subsample=args.subsample, seed=RANDOM_SEED)
    bundle = DatasetBundle.from_dataset(ds)
    print(f"[Data] Loaded {len(bundle)} images")

    # Pre-sample examples for each k (shared across all models)
    fewshot_cache = os.path.join(OUTPUT_DIR, "fewshot_cache")
    examples_cache = {}
    for k in args.k:
        examples_cache[k] = sample_examples(
            k, seed=RANDOM_SEED, cache_dir=fewshot_cache)
        print(f"[FewShot] k={k}: sampled {len(examples_cache[k])} examples")

    for model_name in args.models:
        for k in args.k:
            print(f"\n{'='*60}")
            print(f"[FewShot] model={model_name}, k={k}")
            print(f"{'='*60}")

            model_out_dir = os.path.join(OUTPUT_DIR, model_name)
            filename = f"predictions_fewshot_k{k}.jsonl"

            example_images, example_captions = zip(*examples_cache[k])

            strategy = FewShotStrategy(
                model_name=model_name,
                prompt_template=PROMPT_FEWSHOT,
                k=k,
                example_images=list(example_images),
                example_captions=list(example_captions),
                max_new_tokens=MAX_NEW_TOKENS,
            )
            runner = InferenceRunner(
                strategy=strategy,
                output_dir=model_out_dir,
                filename=filename,
                bundle=bundle,
                prompt_label=f"fewshot_k{k}",
            )
            runner.run(overwrite=args.overwrite, device=args.device)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Inference on UICO Test Set")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["zeroshot", "fewshot"],
                        help="Inference mode.")
    parser.add_argument("--models", nargs="+", default=["llava"],
                        help="Model short names.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Number of images (0 = full test set).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing predictions before starting.")
    # Zeroshot-only
    parser.add_argument("--prompt", type=str, default="A",
                        choices=["A", "B", "C"],
                        help="Prompt template key (zeroshot only).")
    # Fewshot-only
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5],
                        help="Number of few-shot examples per model.")
    args = parser.parse_args()

    if args.mode == "zeroshot":
        run_zeroshot(args)
    else:
        run_fewshot(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('scripts/run_inference.py', doraise=True)"`
Expected: No output (no syntax errors).

- [ ] **Step 3: Verify --help output**

Run: `python scripts/run_inference.py --help`
Expected: Prints usage including `--mode`, `--models`, `--prompt`, `--k`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_inference.py
git commit -m "feat: add unified run_inference.py with --mode flag

Merges run_zeroshot.py and run_fewshot.py into a single script.
--mode zeroshot uses ZeroShotStrategy + PROMPT_MAP.
--mode fewshot uses FewShotStrategy + sample_examples() + PROMPT_FEWSHOT.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Delete old scripts

**Files:**
- Delete: `scripts/run_zeroshot.py`
- Delete: `scripts/run_fewshot.py`

- [ ] **Step 1: Delete run_zeroshot.py**

Run: `git rm scripts/run_zeroshot.py`

- [ ] **Step 2: Delete run_fewshot.py**

Run: `git rm scripts/run_fewshot.py`

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove run_zeroshot.py and run_fewshot.py

Replaced by unified scripts/run_inference.py with --mode flag.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update few-shot architecture reference (line 130)**

Replace:
```
`scripts/run_fewshot.py` uses the same checkpoint/resume pattern as `scripts/run_inference.py` but calls `wrapper.generate_fewshot()`. Few-shot-capable models are auto-discovered by checking `wrapper.supports_fewshot`.
```
With:
```
`scripts/run_inference.py` supports both zero-shot (`--mode zeroshot`) and few-shot (`--mode fewshot`) via the Strategy pattern. Few-shot mode calls `wrapper.generate_fewshot()`; few-shot-capable models are auto-discovered by checking `wrapper.supports_fewshot`.
```

- [ ] **Step 2: Update common commands — zero-shot examples (lines 176-179)**

Already reference `run_inference.py`. Add `--mode zeroshot` for clarity:

```
python scripts/run_inference.py --mode zeroshot --models blip2 llava --subsample 1000 --prompt A
python scripts/run_inference.py --mode zeroshot --models llava qwen2vl --prompt B          # sensitivity (ref-free eval only)
python scripts/run_inference.py --mode zeroshot --models llava-vllm qwen2vl-vllm --prompt A # vLLM backend
python scripts/run_inference.py --mode zeroshot --models llava --prompt A --overwrite       # re-run, discard old predictions
```

- [ ] **Step 3: Update common commands — few-shot examples (lines 190)**

Replace:
```
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5 --subsample 500
```
With:
```
python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5 --subsample 500
```

- [ ] **Step 4: Update eval scripts section**

The section header at line 170 "Evaluation" references these scripts. Verify the eval commands don't need changes (they call `run_eval.py`, not inference scripts).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for unified inference script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update zero-shot command examples (lines 87-100)**

Already reference `run_inference.py`. Add `--mode zeroshot` for consistency:

Lines 87-100 become:
```
python scripts/run_inference.py --mode zeroshot --models blip2 llava --subsample 1000 --prompt A
...
python scripts/run_inference.py --mode zeroshot --models llava-vllm qwen2vl-vllm --prompt A
```

- [ ] **Step 2: Update few-shot command examples (lines 122-128)**

Replace:
```
python scripts/run_fewshot.py --models llava --k 1 --subsample 3
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5 --subsample 500
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3 5
```
With:
```
python scripts/run_inference.py --mode fewshot --models llava --k 1 --subsample 3
python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5 --subsample 500
python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 5
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for unified inference script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Validation

```bash
# 1. Syntax check
python -c "import py_compile; py_compile.compile('scripts/run_inference.py', doraise=True)"

# 2. Help output — both modes accessible
python scripts/run_inference.py --help | grep -E "mode|zeroshot|fewshot"

# 3. Old scripts gone
ls scripts/run_zeroshot.py scripts/run_fewshot.py 2>&1  # should fail with "No such file"

# 4. No dangling references to old script names in active docs
grep -r "run_zeroshot\|run_fewshot" CLAUDE.md README.md  # should return empty

# 5. Integration smoke test (requires GPU, uses --help as dry-run substitute)
python scripts/run_inference.py --mode zeroshot --help > /dev/null
python scripts/run_inference.py --mode fewshot --help > /dev/null
```
