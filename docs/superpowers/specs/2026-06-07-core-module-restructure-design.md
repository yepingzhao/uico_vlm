# Design: Core Module Restructure

**Date**: 2026-06-07
**Complexity**: Medium
**Goal**: Split `common/` into domain-specific sub-packages under `core/`, slim `config/`, and clarify `data/` public API.

## Motivation

Three problems:

1. **`common/` is a dumping ground** — 7 files (1440 lines) covering inference, training, and evaluation with no internal grouping.
2. **`config/__init__.py` mixes concerns** — paths, model registry, vLLM settings, evaluation config, and constants all in one file.
3. **`data/__init__.py` is empty** — no public API surface; consumers import from private submodules.

## Target Structure

```
config/                      # Pure configuration (3→4 files)
├── __init__.py              # Path constants + re-export from submodules
├── models.py                # NEW: MODEL_REGISTRY, SENSITIVITY_MODELS
├── prompts.py               # PROMPT_A/B/C/FS, PROMPT_MAP
└── training.py              # TrainingConfig, MODEL_LORA_CONFIGS, get_lora_config()

core/                        # Engine layer (was: common/)
├── __init__.py              # Package docstring
├── inference/               # Inference subsystem
│   ├── __init__.py          # Re-export public API
│   ├── runner.py            # InferenceRunner (was: common/pipeline.py)
│   ├── strategies.py        # GenerationStrategy + 3 concrete (was: common/strategies.py)
│   └── checkpoint.py        # load_checkpoint (was: common/checkpoint.py)
├── training/                # Training subsystem
│   ├── __init__.py          # Re-export public API
│   ├── runner.py            # TrainingRunner (was: common/training.py)
│   └── adapters.py          # TrainingModelAdapter + 5 adapters (was: common/training_adapter.py)
└── evaluation/              # Evaluation subsystem
    ├── __init__.py          # Re-export public API
    ├── runner.py            # NEW: load_predictions + compute_metrics (orchestration)
    └── metrics.py           # NEW: compute_ref_based_metrics + CLIPScorer (was: common/evaluator.py split)

data/                        # Data loading layer (2 files)
├── __init__.py              # Public API exports (was: empty)
├── dataset.py               # UICOTestDataset, DatasetBundle, load_test_dataset
└── training_dataset.py      # UICOInstructionDataset, collate_fn
```

## Key Design Decisions

### 1. `core/` namespace

Three sub-packages under `core/` instead of top-level `inference/`, `training/`, `evaluation/` packages.
- `core.inference.runner` reads as "engine inference runner" — self-documenting.
- Caps top-level package count at 4 (`config`, `core`, `data`, `models`) instead of 6.
- Risk mitigation: sub-package boundaries prevent `core/` from becoming another dumping ground.

### 2. `common/evaluator.py` split (310 → 2 files)

Boundary: metrics computation ("how to calculate") vs orchestration ("load data → calculate → save").

- `core/evaluation/metrics.py` (~200 lines): `compute_ref_based_metrics`, `CLIPScorer`, `_build_coco_result`, `MAX_CAPTION_WORDS`
- `core/evaluation/runner.py` (~60 lines): `load_predictions`, `compute_metrics`

### 3. `config/__init__.py` slim-down

Extract `MODEL_REGISTRY` + `SENSITIVITY_MODELS` → `config/models.py`.
Keep paths, inference settings, and VLLM constants in `__init__.py`.
Re-export `MODEL_REGISTRY` from `__init__.py` for backward compatibility.

### 4. Monkey-patch containment

`common/training_adapter.py` applies a module-level `DynamicCache` monkey-patch at import time.
Move into `Phi35Adapter.load_processor()` so it only runs when Phi-3.5 is actually used.

### 5. `data/training_dataset.py` kept as-is

`config/training.py`, `core/training/`, and `data/training_dataset.py` all deal with training.
Renaming `training_dataset.py` → `training.py` would create confusion with `config/training.py` in import contexts. The current name is explicit enough.

## File Changes

### CREATE

| File | Content |
|------|---------|
| `config/models.py` | MODEL_REGISTRY, SENSITIVITY_MODELS extracted from `config/__init__.py` |
| `core/__init__.py` | Package docstring |
| `core/inference/__init__.py` | Re-export InferenceRunner, ZeroShotStrategy, FewShotStrategy, LoRAStrategy, load_checkpoint |
| `core/training/__init__.py` | Re-export TrainingRunner, get_training_adapter |
| `core/evaluation/__init__.py` | Re-export compute_metrics, load_predictions |
| `core/evaluation/runner.py` | load_predictions + compute_metrics from `common/evaluator.py` |
| `core/evaluation/metrics.py` | compute_ref_based_metrics + CLIPScorer from `common/evaluator.py` |

### MOVE (git mv)

| From | To | Internal changes |
|------|----|-----------------|
| `common/pipeline.py` | `core/inference/runner.py` | Update internal imports |
| `common/strategies.py` | `core/inference/strategies.py` | Update `from config` import |
| `common/checkpoint.py` | `core/inference/checkpoint.py` | None needed |
| `common/training.py` | `core/training/runner.py` | Update imports from config/data/models |
| `common/training_adapter.py` | `core/training/adapters.py` | Move monkey-patch into Phi35Adapter method |

### MODIFY

| File | Change |
|------|--------|
| `config/__init__.py` | Remove MODEL_REGISTRY/SENSITIVITY_MODELS; add re-export from config.models |
| `data/__init__.py` | Add public API exports |
| `data/dataset.py` | Move lazy `from config import` to top-level |
| `scripts/run_inference.py` | `common.xxx` → `core.inference.xxx` |
| `scripts/run_eval.py` | `common.evaluator` → `core.evaluation.runner` |
| `scripts/run_lora.py` | `common.xxx` → `core.xxx` |

### DELETE

| File | Reason |
|------|--------|
| `common/__init__.py` | Package removed |
| `common/pipeline.py` | Moved → `core/inference/runner.py` |
| `common/strategies.py` | Moved → `core/inference/strategies.py` |
| `common/checkpoint.py` | Moved → `core/inference/checkpoint.py` |
| `common/evaluator.py` | Split → `core/evaluation/runner.py` + `metrics.py` |
| `common/training.py` | Moved → `core/training/runner.py` |
| `common/training_adapter.py` | Moved → `core/training/adapters.py` |

## Unaffected

- `config/prompts.py` — no imports changed
- `config/training.py` — no imports changed
- `data/training_dataset.py` — no internal changes
- `models/` (all) — no model file imports from `common`
- `download_models.py` — only imports `config.MODEL_REGISTRY` (backward-compatible)
- `models/fewshot/` — only imports from `config` and `data.dataset`

## Dependency Graph (post-restructure)

```
config/  ←  core/  ←  scripts/
  ↓          ↓
data/     models/
  ↑          ↑
  └──────────┘
```

- `models/` depends on `config` (registry, training config) and `data` (resolve_image_path)
- `core/` depends on `config`, `data`, and `models`
- `scripts/` depends on `core`, `config`, `data`
- `data/` depends on `config`
- `config/` depends on nothing

## Validation

```bash
# 1. All new import paths work
python -c "from core.inference import InferenceRunner, ZeroShotStrategy, FewShotStrategy, LoRAStrategy, load_checkpoint"
python -c "from core.training import TrainingRunner, get_training_adapter"
python -c "from core.evaluation import compute_metrics, load_predictions"
python -c "from config.models import MODEL_REGISTRY, SENSITIVITY_MODELS"
python -c "from data import UICOTestDataset, DatasetBundle, UICOInstructionDataset"

# 2. Backward-compatible re-exports
python -c "from config import MODEL_REGISTRY"

# 3. Script CLI entry points work
python scripts/run_inference.py --help
python scripts/run_eval.py --help
python scripts/run_lora.py --help
```

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Script import paths break | Medium | Each script touched, verified via `--help` |
| `models/` discovers new dependency on `common` | Low | Grep confirmed `models/` only imports `config`/`data` |
| evaluator split introduces circular import | Low | `metrics.py` has zero imports from `core/`; `runner.py` imports `metrics.py` one-way |
| `__pycache__` from old `common/` confuses pytest | Low | `git rm` old files; stale `.pyc` doesn't affect `import` |
