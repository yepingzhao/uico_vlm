# Design: eval/ Module Consolidation

**Date**: 2026-06-07
**Complexity**: Small

## Summary

Merge three files (`eval/ref_based.py`, `eval/ref_free.py`, `common/eval_core.py`) into a single `eval/evaluator.py`. This eliminates the artificial split between "pure metric computation" (in `eval/`) and "evaluation orchestration" (in `common/`), and removes `common/`'s dependency on `eval/`.

## Current vs Target

```
BEFORE:                              AFTER:
eval/                                eval/
  __init__.py (empty)                  __init__.py (re-exports)
  ref_based.py  (121 lines)            evaluator.py (~280 lines)
  ref_free.py   (101 lines)
common/                              common/
  eval_core.py  (85 lines) ← DELETE     (no eval dependency)
```

## evaluator.py Internal Structure

Three logical sections, section-commented:

1. **Data loading** — `load_predictions(filepath) -> dict` (from `common/eval_core.py`)
2. **Ref-based metrics** — `compute_ref_based_metrics()` + `_build_coco_result()` (from `eval/ref_based.py`)
3. **Ref-free metrics** — `CLIPScorer` class (from `eval/ref_free.py`)
4. **Orchestration** — `compute_metrics()` (from `common/eval_core.py`)

## Public API (via `__init__.py`)

```python
from eval import load_predictions, compute_metrics
```

`compute_ref_based_metrics` and `CLIPScorer` remain importable from `eval.evaluator` for advanced use, but are not the primary API.

## External Impact

| File | Change |
|---|---|
| `scripts/run_eval.py:29` | `from common.eval_core` → `from eval import load_predictions, compute_metrics` |
| `common/eval_core.py` | DELETE |
| `eval/ref_based.py` | DELETE |
| `eval/ref_free.py` | DELETE |
| `eval/__init__.py` | UPDATE — add re-exports |
| `eval/evaluator.py` | CREATE |

No other files import `common.eval_core`, `eval.ref_based`, or `eval.ref_free`.

## Verification

```bash
# Dry run: confirm no imports broken
python -c "from eval import load_predictions, compute_metrics; print('OK')"

# Integration: run eval on a small subset
python scripts/run_eval.py --model blip2 --prompt A --ref_free_only

# Check old import paths are gone
python -c "import common.eval_core"  # should fail
python -c "import eval.ref_based"    # should fail
python -c "import eval.ref_free"     # should fail
```

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `from eval.ref_based import ...` used externally | Low | grep confirms no references |
| `from common.eval_core import ...` used externally | Low | grep confirms only `scripts/run_eval.py` |
| Merged file too long | Low | ~280 lines, well within maintainable limits |
| Import order issues | Low | Imports are external-only (torch, PIL, pycocoevalcap, config), no intra-package deps |
