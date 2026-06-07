# Design: Merge run_zeroshot.py and run_fewshot.py into run_inference.py

**Date:** 2026-06-07
**Status:** approved

## Summary

Merge `scripts/run_zeroshot.py` and `scripts/run_fewshot.py` into a single `scripts/run_inference.py` with a `--mode` flag. The existing Strategy pattern (`common/strategies.py`) and inference loop (`common/pipeline.py`) already cleanly separate generation logic from orchestration, so the merge is a pure argparse consolidation — zero changes to the underlying abstractions.

## CLI Design

```
# Zero-shot
python scripts/run_inference.py --mode zeroshot --models llava --prompt A --subsample 500
python scripts/run_inference.py --mode zeroshot --models llava qwen2vl --prompt B --overwrite

# Few-shot
python scripts/run_inference.py --mode fewshot --models llava qwen2vl --k 1 3 --subsample 500
python scripts/run_inference.py --mode fewshot --models llava --k 1 --overwrite
```

### Argument Matrix

| Argument | Mode | Required | Default | Notes |
|----------|------|----------|---------|-------|
| `--mode` | both | yes | — | `choices=["zeroshot", "fewshot"]` |
| `--models` | both | no | `["llava"]` | nargs="+", shared |
| `--subsample` | both | no | `0` (full set) | shared |
| `--device` | both | no | `"cuda:0"` | shared |
| `--overwrite` | both | no | `False` | shared |
| `--prompt` | zeroshot | no | `"A"` | `choices=["A","B","C"]` |
| `--k` | fewshot | no | `[1, 3, 5]` | nargs="+", type=int |

- `--prompt` is silently ignored in fewshot mode (always uses `PROMPT_FEWSHOT`)
- `--k` is silently ignored in zeroshot mode

## Main Loop Structure

Both modes share the same outer pipeline:

```
load_test_dataset → DatasetBundle → for models → strategy → InferenceRunner.run()
```

### Zeroshot branch
```python
for model_name in args.models:
    strategy = ZeroShotStrategy(model_name, prompt_text, MAX_NEW_TOKENS)
    filename = f"predictions_prompt_{args.prompt.lower()}.jsonl"
    runner = InferenceRunner(strategy, model_out_dir, filename, bundle,
                             prompt_label=args.prompt)
    runner.run(overwrite=args.overwrite, device=args.device)
```

### Fewshot branch
```python
# Pre-sample examples once (shared across all models)
fewshot_cache = {}
for k in args.k:
    fewshot_cache[k] = sample_examples(k, seed=RANDOM_SEED, cache_dir=...)

for model_name in args.models:
    for k in args.k:
        example_images, example_captions = zip(*fewshot_cache[k])
        strategy = FewShotStrategy(model_name, PROMPT_FEWSHOT, k,
                                   list(example_images), list(example_captions),
                                   MAX_NEW_TOKENS)
        filename = f"predictions_fewshot_k{k}.jsonl"
        runner = InferenceRunner(strategy, model_out_dir, filename, bundle,
                                 prompt_label=f"fewshot_k{k}")
        runner.run(overwrite=args.overwrite, device=args.device)
```

## Error Handling

All error handling is inherited from existing components — no new handling added:

- **Invalid `--mode`** — argparse `choices` rejects with helpful message
- **Invalid `--prompt`** — argparse `choices` rejects
- **Model unsupported by mode** — `strategy.prepare()` raises naturally; `InferenceRunner` catches, logs empty caption, continues
- **Checkpoint/resume** — existing `InferenceRunner` / `checkpoint.py` handles transparently
- **GPU OOM** — Python propagates; user sees stack trace (existing behavior)

## Files Changed

| File | Action | Reason |
|------|--------|--------|
| `scripts/run_inference.py` | CREATE | Merged script (~120 lines) |
| `scripts/run_zeroshot.py` | DELETE | Replaced |
| `scripts/run_fewshot.py` | DELETE | Replaced |
| `CLAUDE.md` | UPDATE | Update all command examples |
| `README.md` | UPDATE | Update all command examples |

Files NOT changed:
- `common/strategies.py` — unchanged (merge validates the Strategy abstraction)
- `common/pipeline.py` — unchanged
- `common/dataset_bundle.py` — unchanged
- `scripts/run_lora.py` — unchanged (separate training+inference workflow)
- `docs/research-notes/*` — historical snapshots, intentionally preserved as-is
- `docs/superpowers/*` — historical plans, intentionally preserved as-is

## Validation

```bash
# Syntax check
python scripts/run_inference.py --help

# Dry-run on small subset (zeroshot)
python scripts/run_inference.py --mode zeroshot --models llava --prompt A --subsample 3

# Dry-run on small subset (fewshot)
python scripts/run_inference.py --mode fewshot --models llava --k 1 --subsample 3
```

## Constraints

- `run_lora.py` is NOT merged — it wraps training + inference together, with many training-specific hyperparams; merging it with pure inference scripts would create an overweight CLI
- No changes to `common/` — the Strategy pattern already proves correctness through its ability to support this merge with zero modifications
- Existing output directory structure and filename conventions are preserved exactly
