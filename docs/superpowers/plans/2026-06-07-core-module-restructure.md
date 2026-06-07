# Core Module Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `common/` into `core/{inference,training,evaluation}/`, extract `MODEL_REGISTRY` to `config/models.py`, add `data/__init__.py` public API, and update all import paths.

**Architecture:** Pure structural refactoring — no behavioral changes. Move files with `git mv`, split `common/evaluator.py` (310 lines) into `core/evaluation/metrics.py` + `core/evaluation/runner.py`, and fix a module-level monkey-patch in the training adapter.

**Tech Stack:** Python 3.10, no new dependencies

---

### Task 1: Create directory scaffolding

**Files:**
- Create: `core/__init__.py`
- Create: `core/inference/__init__.py`
- Create: `core/training/__init__.py`
- Create: `core/evaluation/__init__.py`

- [ ] **Step 1: Create directories and `core/__init__.py`**

```bash
mkdir -p core/inference core/training core/evaluation
```

- [ ] **Step 2: Write `core/__init__.py`**

```python
"""Engine layer — inference, training, and evaluation pipelines.

Sub-packages:
  - inference:   InferenceRunner, GenerationStrategy, checkpoint/resume
  - training:    TrainingRunner, TrainingModelAdapter, QLoRA loop
  - evaluation:  ref-based & ref-free metrics, CLIP scoring
"""
```

- [ ] **Step 3: Write `core/inference/__init__.py`**

```python
"""Inference pipeline — runner, strategies, and checkpoint/resume."""

from core.inference.runner import InferenceRunner
from core.inference.strategies import (
    FewShotStrategy,
    GenerationStrategy,
    LoRAStrategy,
    ZeroShotStrategy,
)
from core.inference.checkpoint import load_checkpoint

__all__ = [
    "InferenceRunner",
    "GenerationStrategy",
    "ZeroShotStrategy",
    "FewShotStrategy",
    "LoRAStrategy",
    "load_checkpoint",
]
```

- [ ] **Step 4: Write `core/training/__init__.py`**

```python
"""QLoRA training pipeline — runner and model-family adapters."""

from core.training.runner import TrainingRunner
from core.training.adapters import get_training_adapter

__all__ = [
    "TrainingRunner",
    "get_training_adapter",
]
```

- [ ] **Step 5: Write `core/evaluation/__init__.py`**

```python
"""Evaluation — ref-based metrics, CLIP scoring, and orchestration."""

from core.evaluation.runner import compute_metrics, load_predictions

__all__ = [
    "compute_metrics",
    "load_predictions",
]
```

- [ ] **Step 6: Verify directories exist**

Run: `ls -la core/ core/inference/ core/training/ core/evaluation/`
Expected: all four directories with `__init__.py` files

- [ ] **Step 7: Commit**

```bash
git add core/
git commit -m "feat: add core/ directory scaffolding with sub-package __init__.py files"
```

---

### Task 2: Extract MODEL_REGISTRY to config/models.py

**Files:**
- Create: `config/models.py`
- Modify: `config/__init__.py`

- [ ] **Step 1: Write `config/models.py`**

```python
"""Model registry and sensitivity analysis configuration."""

# Each entry: (short_name, hf_model_id, wrapper_class_name)
MODEL_REGISTRY = [
    ("blip2",           "Salesforce/blip2-flan-t5-xl",            "BLIP2Wrapper"),
    ("instructblip",    "Salesforce/instructblip-vicuna-7b",      "InstructBLIPWrapper"),
    ("llava",           "llava-hf/llava-1.5-7b-hf",              "LLaVAWrapper"),
    ("internvl2",       "OpenGVLab/InternVL2-8B",                 "InternVL2Wrapper"),
    ("qwen2vl",         "Qwen/Qwen2.5-VL-7B-Instruct",            "Qwen2VLWrapper"),
    # New VLM baselines
    ("phi35-vision",    "microsoft/Phi-3.5-vision-instruct",       "Phi35VisionWrapper"),
    ("phi4-mm",         "microsoft/Phi-4-multimodal-instruct",     "Phi4MultimodalWrapper"),
    ("paligemma2",      "google/paligemma2-3b-ft-docci-448",      "PaliGemma2Wrapper"),
    ("minicpm-v",       "openbmb/MiniCPM-V-2_6",                  "MiniCPMVWrapper"),
    ("deepseek-vl2",    "deepseek-ai/deepseek-vl2-small",          "DeepSeekVL2Wrapper"),
    ("llava-next",      "llava-hf/llava-v1.6-mistral-7b-hf",      "LLaVANeXTWrapper"),
    ("idefics3",        "HuggingFaceM4/Idefics3-8B-Llama3",       "Idefics3Wrapper"),
    ("internvl25",      "OpenGVLab/InternVL2_5-8B",               "InternVL25Wrapper"),
    ("internvl35",      "OpenGVLab/InternVL3_5-8B",              "InternVL35Wrapper"),
    ("pixtral",         "mistralai/Pixtral-12B-2409",             "PixtralWrapper"),
    ("llama32-vision",  "meta-llama/Llama-3.2-11B-Vision-Instruct","Llama32VisionWrapper"),
    ("qwen3vl",         "Qwen/Qwen3-VL-8B-Instruct",               "Qwen3VLWrapper"),
]

# Sensitivity analysis: only run B/C prompts on these models
SENSITIVITY_MODELS = ["llava", "qwen2vl"]
```

- [ ] **Step 2: Edit `config/__init__.py` — remove MODEL_REGISTRY and SENSITIVITY_MODELS, add re-export**

Remove lines 16-41 (the entire `MODEL_REGISTRY` list and `SENSITIVITY_MODELS`). Add this import near the top (after the `import os` line):

```python
from config.models import MODEL_REGISTRY, SENSITIVITY_MODELS  # noqa: F401 — re-exported for backward compatibility
```

The line should be inserted after line 1 (`import os`), before the Base paths section. The `# noqa: F401` is needed because the symbols are re-exported for consumers that do `from config import MODEL_REGISTRY`.

- [ ] **Step 3: Verify old import still works**

Run: `python -c "from config import MODEL_REGISTRY, SENSITIVITY_MODELS; print(len(MODEL_REGISTRY))"`
Expected: `17`

- [ ] **Step 4: Commit**

```bash
git add config/models.py config/__init__.py
git commit -m "refactor: extract MODEL_REGISTRY to config/models.py"
```

---

### Task 3: Move common/ files to core/ with git mv

**Files:**
- Move: `common/pipeline.py` → `core/inference/runner.py`
- Move: `common/strategies.py` → `core/inference/strategies.py`
- Move: `common/checkpoint.py` → `core/inference/checkpoint.py`
- Move: `common/training.py` → `core/training/runner.py`
- Move: `common/training_adapter.py` → `core/training/adapters.py`

- [ ] **Step 1: Move files with git mv**

```bash
git mv common/pipeline.py core/inference/runner.py
git mv common/strategies.py core/inference/strategies.py
git mv common/checkpoint.py core/inference/checkpoint.py
git mv common/training.py core/training/runner.py
git mv common/training_adapter.py core/training/adapters.py
```

- [ ] **Step 2: Update internal import in `core/inference/runner.py`**

The file currently imports from `common.checkpoint` and `common.strategies`. Edit lines 14-15:

Old:
```python
from common.checkpoint import load_checkpoint
from common.strategies import GenerationStrategy
```

New:
```python
from core.inference.checkpoint import load_checkpoint
from core.inference.strategies import GenerationStrategy
```

- [ ] **Step 3: Verify imports resolve**

Run: `python -c "from core.inference.runner import InferenceRunner; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add core/inference/runner.py core/inference/strategies.py core/inference/checkpoint.py core/training/runner.py core/training/adapters.py common/
git commit -m "refactor: move common/ files to core/{inference,training}/"
```

---

### Task 4: Split common/evaluator.py into core/evaluation/

**Files:**
- Create: `core/evaluation/metrics.py`
- Create: `core/evaluation/runner.py`
- Delete: `common/evaluator.py` (not yet deleted — will be handled in cleanup)

- [ ] **Step 1: Write `core/evaluation/metrics.py`**

This file contains all metric computation logic: constants, ref-based metrics, and CLIPScorer.

```python
"""Reference-based and reference-free metric computation.

Does NOT handle file I/O or orchestration — only "how to calculate".
"""

import json
import os
import sys
import tempfile
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import CLIP_MODEL_NAME

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum word count for prediction captions in ref-based metrics.
#: Captions exceeding this are truncated before evaluation.  Reference captions
#: in this dataset are ≤34 words (median 9), so words beyond ~50 cannot match
#: any reference n-gram.  Truncation also prevents OOM in the Stanford parser
#: used by SPICE when models generate very long captions (e.g. Qwen2.5-VL).
MAX_CAPTION_WORDS = 50


# ---------------------------------------------------------------------------
# Reference-based metrics
# ---------------------------------------------------------------------------

def _build_coco_result(predictions: Dict[int, str], max_words: int = MAX_CAPTION_WORDS) -> List[dict]:
    """Convert {image_id: caption} to COCO result format.

    Captions longer than *max_words* are truncated to keep SPICE's Stanford
    parser within memory limits.
    """
    result = []
    for img_id, caption in predictions.items():
        words = caption.split()
        if len(words) > max_words:
            caption = " ".join(words[:max_words])
        result.append({"image_id": img_id, "caption": caption})
    return result


def compute_ref_based_metrics(
    predictions: Dict[int, str],
    references: Dict[int, List[str]],
    cache_dir: str = None,
) -> Dict[str, float]:
    """Compute all reference-based metrics using pycocoevalcap.

    Args:
        predictions: {image_id: generated_caption}
        references: {image_id: [ref1, ref2, ...]}
        cache_dir: Optional directory for caching intermediate files.

    Returns:
        Dict with keys: BLEU-1, BLEU-4, METEOR, ROUGE_L, CIDEr, SPICE, S_m, S*_m
    """
    # Write predictions and references to temporary COCO-format JSON
    tmpdir = tempfile.mkdtemp(prefix="uico_vlm_eval_")
    try:
        pred_file = os.path.join(tmpdir, "predictions.json")
        ref_file = os.path.join(tmpdir, "references.json")

        # Predictions as list of {image_id, caption}
        pred_list = _build_coco_result(predictions)
        with open(pred_file, "w") as f:
            json.dump(pred_list, f)

        # References as {image_id: [captions]}
        with open(ref_file, "w") as f:
            json.dump(references, f)

        # Use pycocoevalcap's COCOEvalCap
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap

        # Create a minimal COCO ground-truth structure
        anns = []
        img_ids_set = set()
        ann_id = 0
        for img_id, refs in references.items():
            img_ids_set.add(img_id)
            for ref in refs:
                anns.append({
                    "image_id": img_id,
                    "id": ann_id,
                    "caption": ref,
                })
                ann_id += 1

        imgs = [{"id": img_id} for img_id in sorted(img_ids_set)]

        gt_file = os.path.join(tmpdir, "gt.json")
        with open(gt_file, "w") as f:
            json.dump({"images": imgs, "annotations": anns, "info": {}, "licenses": []}, f)

        coco = COCO(gt_file)
        coco_res = coco.loadRes(pred_file)

        # Run evaluation
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params["image_id"] = coco_res.getImgIds()
        try:
            coco_eval.evaluate()
        except Exception as e:
            print(f"  [WARN] Ref-based eval partially failed: {e}", file=sys.stderr)

        # Collect results
        metrics = {}
        for metric, score in coco_eval.eval.items():
            key = metric.upper().replace(" ", "_")
            metrics[key] = score * 100

        # Compute S_m (composite) per paper Eq.(1)
        sm_keys = ["BLEU_4", "METEOR", "ROUGE_L", "CIDER", "SPICE"]
        metrics["S_m"] = sum(metrics.get(k, 0.0) for k in sm_keys) / len(sm_keys)

        # Compute S*_m (SPICE-excluded composite) per zhang2021global
        sm_star_keys = ["BLEU_4", "METEOR", "ROUGE_L", "CIDER"]
        metrics["S*_m"] = sum(metrics.get(k, 0.0) for k in sm_star_keys) / len(sm_star_keys)

        return metrics

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reference-free metrics (CLIPScore)
# ---------------------------------------------------------------------------

class CLIPScorer:
    """Compute CLIPScore and RefCLIPScore for image-caption pairs."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda:0"):
        self.device = device
        self._model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self._model.eval()

    @torch.no_grad()
    def compute_refclipscore(
        self,
        image_paths: Dict[int, str],
        captions: Dict[int, str],
        references: Dict[int, List[str]],
    ) -> Dict[str, float]:
        """Compute RefCLIPScore: harmonic mean of CLIPScore and reference-anchored score.

        For each image, we compute CLIPScore(image, generated) and
        the max CLIPScore among references for that image.
        RefCLIPScore = 2 * CLIPScore * RefMax / (CLIPScore + RefMax).
        """
        if not image_paths:
            return {"RefCLIPScore": 0.0, "CLIPScore": 0.0, "RefMax": 0.0}

        clip_scores = []
        ref_max_scores = []
        refclip_scores = []
        img_ids = sorted(set(image_paths.keys()) & set(captions.keys()))

        for img_id in img_ids:
            img_path = image_paths[img_id]
            caption = captions[img_id]
            refs = references.get(img_id, [])

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # CLIPScore for generated caption
            inputs = self._processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self._model(**inputs)
            gen_sim = torch.nn.functional.cosine_similarity(
                outputs.image_embeds[0], outputs.text_embeds[0], dim=0
            ).item()
            clip_scores.append(gen_sim)

            # Max CLIPScore among references
            ref_max = 0.0
            if refs:
                ref_inputs = self._processor(
                    text=refs,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                ref_outputs = self._model(**ref_inputs)
                for i in range(len(refs)):
                    ref_sim = torch.nn.functional.cosine_similarity(
                        ref_outputs.image_embeds[0],
                        ref_outputs.text_embeds[i],
                        dim=0,
                    ).item()
                    ref_max = max(ref_max, ref_sim)
            ref_max_scores.append(ref_max)

            # Harmonic mean
            if gen_sim + ref_max > 0:
                rc = 2 * gen_sim * ref_max / (gen_sim + ref_max)
            else:
                rc = 0.0
            refclip_scores.append(rc)

        n = len(clip_scores)
        return {
            "CLIPScore": sum(clip_scores) / n if n else 0.0,
            "RefMax": sum(ref_max_scores) / n if n else 0.0,
            "RefCLIPScore": sum(refclip_scores) / n if n else 0.0,
        }
```

- [ ] **Step 2: Write `core/evaluation/runner.py`**

This file handles data loading and orchestration — it imports metrics computation from `core.evaluation.metrics`.

```python
"""Evaluation orchestration — load predictions, run metrics, save results."""

import json
import os
import sys

from config import CLIP_MODEL_NAME
from core.evaluation.metrics import CLIPScorer, compute_ref_based_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(filepath: str) -> dict:
    """Load predictions from JSONL file → {image_id: caption}."""
    preds = {}
    if not os.path.exists(filepath):
        print(f"[WARN] Predictions file not found: {filepath}")
        return preds
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            preds[record["image_id"]] = record["caption"]
    return preds


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: dict,
    references: dict,
    image_paths: dict,
    metrics_file: str,
    device: str = "cuda:0",
    skip_ref_based: bool = False,
) -> dict:
    """Compute ref-based and ref-free metrics, save to disk.

    Args:
        predictions: {image_id: caption} dict.
        references: {image_id: [ref_captions]} dict.
        image_paths: {image_id: filesystem_path} dict for CLIP scoring.
        metrics_file: Path to save the metrics JSON.
        device: CUDA device for CLIP scorer.
        skip_ref_based: If True, only compute ref-free metrics.

    Returns:
        Metrics dict (e.g. {"BLEU-1": 45.2, "CLIPScore": 0.72, ...}).
    """
    metrics = {}

    # Reference-based
    if not skip_ref_based:
        print("  Computing ref-based metrics...")
        ref_metrics = compute_ref_based_metrics(predictions, references)
        metrics.update(ref_metrics)
        for k, v in ref_metrics.items():
            print(f"    {k}: {v:.2f}")

        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved (ref-based) → {metrics_file}")

    # Reference-free
    print("  Computing ref-free metrics...")
    try:
        scorer = CLIPScorer(model_name=CLIP_MODEL_NAME, device=device)
        clip_metrics = scorer.compute_refclipscore(
            image_paths, predictions, references
        )
        metrics.update(clip_metrics)
        for k, v in clip_metrics.items():
            print(f"    {k}: {v:.4f}")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved (with ref-free) → {metrics_file}")
    except Exception as e:
        print(f"    [WARN] CLIP failed (ref-free metrics unavailable): {e}")

    return metrics
```

- [ ] **Step 3: Verify imports resolve**

Run:
```bash
python -c "from core.evaluation.metrics import compute_ref_based_metrics, CLIPScorer; print('metrics OK')"
python -c "from core.evaluation.runner import compute_metrics, load_predictions; print('runner OK')"
python -c "from core.evaluation import compute_metrics, load_predictions; print('package OK')"
```

Expected: all three print OK

- [ ] **Step 4: Commit**

```bash
git add core/evaluation/metrics.py core/evaluation/runner.py
git commit -m "refactor: split evaluator into core/evaluation/{metrics,runner}.py"
```

---

### Task 5: Fix monkey-patch and update imports in core/training/adapters.py

**Files:**
- Modify: `core/training/adapters.py`

- [ ] **Step 1: Move DynamicCache monkey-patch into Phi35Adapter.load_processor()**

The file currently has this at module level (around line 307-309):

```python
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None
```

Remove those 3 lines. Then inside `Phi35Adapter.load_processor()`, add at the beginning of the method:

```python
    def load_processor(self, model_id: str, model_cfg: dict) -> dict:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
        # ... existing code follows
```

The `from transformers import AutoProcessor` that was originally inside the method body should appear after this block. In the original file, the method body starts with `from transformers import AutoProcessor`. After the edit, the method should look like:

```python
    def load_processor(self, model_id: str, model_cfg: dict) -> dict:
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
```

- [ ] **Step 2: Verify import does NOT trigger the monkey-patch**

Run: `python -c "from core.training.adapters import Phi35Adapter; print('import OK (no side-effects)')"`
Expected: `import OK (no side-effects)`

- [ ] **Step 3: Commit**

```bash
git add core/training/adapters.py
git commit -m "fix: move DynamicCache monkey-patch into Phi35Adapter.load_processor()"
```

---

### Task 6: Update data/__init__.py and data/dataset.py

**Files:**
- Modify: `data/__init__.py`
- Modify: `data/dataset.py`

- [ ] **Step 1: Write `data/__init__.py` (replace empty file)**

```python
"""Data loading layer — COCO-format test/training datasets for UICO."""

from data.dataset import (
    DatasetBundle,
    UICOTestDataset,
    load_test_dataset,
    resolve_image_path,
)
from data.training_dataset import UICOInstructionDataset, collate_fn

__all__ = [
    "DatasetBundle",
    "UICOTestDataset",
    "UICOInstructionDataset",
    "collate_fn",
    "load_test_dataset",
    "resolve_image_path",
]
```

- [ ] **Step 2: Edit `data/dataset.py` — move lazy import to top-level**

Change lines 80-82 from:

```python
def load_test_dataset(subsample: int = 0, seed: int = 42) -> UICOTestDataset:
    """Convenience: load the full test set, optionally subsample."""
    from config import TEST_ANN_FILE, IMAGES_BASE_DIR
```

To:

```python
from config import TEST_ANN_FILE, IMAGES_BASE_DIR

def load_test_dataset(subsample: int = 0, seed: int = 42) -> UICOTestDataset:
    """Convenience: load the full test set, optionally subsample."""
```

The `from config import ...` is added at the top of the file (after line 6 `from pycocotools.coco import COCO`), and removed from the function body.

- [ ] **Step 3: Verify imports**

Run:
```bash
python -c "from data import UICOTestDataset, DatasetBundle, UICOInstructionDataset, collate_fn, load_test_dataset, resolve_image_path; print('data OK')"
```

Expected: `data OK`

- [ ] **Step 4: Commit**

```bash
git add data/__init__.py data/dataset.py
git commit -m "refactor: add data/__init__.py public API, lift lazy import to top-level"
```

---

### Task 7: Update script import paths

**Files:**
- Modify: `scripts/run_inference.py`
- Modify: `scripts/run_eval.py`
- Modify: `scripts/run_lora.py`

- [ ] **Step 1: Edit `scripts/run_inference.py` lines 26-27**

Old:
```python
from common.strategies import ZeroShotStrategy, FewShotStrategy
from common.pipeline import InferenceRunner
```

New:
```python
from core.inference.strategies import ZeroShotStrategy, FewShotStrategy
from core.inference.runner import InferenceRunner
```

- [ ] **Step 2: Edit `scripts/run_eval.py` line 29**

Old:
```python
from common.evaluator import load_predictions, compute_metrics
```

New:
```python
from core.evaluation.runner import load_predictions, compute_metrics
```

- [ ] **Step 3: Edit `scripts/run_lora.py` lines 31-34**

Old:
```python
from common.strategies import LoRAStrategy
from common.pipeline import InferenceRunner
from common.training_adapter import get_training_adapter
from common.training import TrainingRunner
```

New:
```python
from core.inference.strategies import LoRAStrategy
from core.inference.runner import InferenceRunner
from core.training.adapters import get_training_adapter
from core.training.runner import TrainingRunner
```

- [ ] **Step 4: Verify script import paths**

Run:
```bash
python scripts/run_inference.py --help
python scripts/run_eval.py --help
python scripts/run_lora.py --help
```

Expected: all three print help text without ImportError

- [ ] **Step 5: Commit**

```bash
git add scripts/run_inference.py scripts/run_eval.py scripts/run_lora.py
git commit -m "refactor: update script imports common/ → core/"
```

---

### Task 8: Remove old common/ directory

**Files:**
- Delete: `common/__init__.py`
- Delete: `common/evaluator.py` (already split, not needed)

- [ ] **Step 1: Check what remains in common/**

Run: `ls common/`
Expected: `__init__.py  evaluator.py  __pycache__`

(The other 5 files were already `git mv`'d in Task 3.)

- [ ] **Step 2: Remove remaining files**

```bash
git rm common/__init__.py common/evaluator.py
rm -rf common/__pycache__
```

- [ ] **Step 3: Verify common/ is gone**

Run: `ls common/ 2>&1`
Expected: `ls: cannot access 'common/': No such file or directory`

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove old common/ directory"
```

---

### Task 9: Final verification

- [ ] **Step 1: Full import smoke test**

```bash
python -c "
from config import MODEL_REGISTRY, OUTPUT_DIR, DATA_BASE
from config.models import MODEL_REGISTRY, SENSITIVITY_MODELS
from config.prompts import PROMPT_A, PROMPT_B, PROMPT_C, PROMPT_FEWSHOT, PROMPT_MAP
from config.training import TrainingConfig, MODEL_LORA_CONFIGS, get_lora_config
from core.inference import InferenceRunner, ZeroShotStrategy, FewShotStrategy, LoRAStrategy, load_checkpoint
from core.training import TrainingRunner, get_training_adapter
from core.evaluation import compute_metrics, load_predictions
from data import UICOTestDataset, DatasetBundle, UICOInstructionDataset, collate_fn, load_test_dataset, resolve_image_path
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 2: Check no file still imports from `common`**

```bash
grep -rn "from common\|import common" --include='*.py' . | grep -v __pycache__ | grep -v '.pyc' | grep -v 'docs/'
```

Expected: no output (empty)

- [ ] **Step 3: Check git status is clean**

Run: `git status`
Expected: clean working tree

- [ ] **Step 4: Commit (if any remaining changes)**

```bash
git add -A
git diff --cached --stat
# If anything staged:
git commit -m "chore: final cleanup after core/ restructure"
```
