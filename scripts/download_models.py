#!/usr/bin/env python3
"""Download all VLM models from HuggingFace to local cache.

Usage:
    python download_models.py                # download all 15 models
    python download_models.py --dry-run      # list what would be downloaded
    python download_models.py --model blip2  # download a single model
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download
from config import MODEL_REGISTRY


def is_cached(model_id: str) -> bool:
    """Check if model files are already in HF cache."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = model_id.replace("/", "--")
    snapshots_dir = os.path.join(hf_home, "hub", "models--" + model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False
    subdirs = os.listdir(snapshots_dir)
    return len(subdirs) > 0


# Use HF mirror if official endpoint is unreachable
HF_ENDPOINT = os.environ.get(
    "HF_ENDPOINT",
    "https://hf-mirror.com",
)


def download_model(model_id: str, display_name: str) -> bool:
    """Download a single model. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"[Download] {display_name}")
    print(f"           {model_id}")
    print(f"           endpoint: {HF_ENDPOINT}")
    print(f"{'='*60}")

    if is_cached(model_id):
        print("  → Already cached, skipping.")
        return True

    try:
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            max_workers=4,
            endpoint=HF_ENDPOINT,
        )
        print("  ✓ Done.")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download VLM evaluation models")
    parser.add_argument("--model", type=str, help="Download a single model by short name")
    parser.add_argument("--dry-run", action="store_true", help="List models without downloading")
    args = parser.parse_args()

    # Filter models
    if args.model:
        models = [(n, hid, cls) for n, hid, cls in MODEL_REGISTRY if n == args.model]
        if not models:
            print(f"Unknown model: {args.model}")
            print(f"Available: {[n for n, _, _ in MODEL_REGISTRY]}")
            sys.exit(1)
    else:
        models = MODEL_REGISTRY

    print(f"{'[Dry run] ' if args.dry_run else ''}Models to download: {len(models)}")
    for name, hf_id, cls_name in models:
        cached = " (cached)" if is_cached(hf_id) else ""
        print(f"  {name:20s} → {hf_id}{cached}")

    if args.dry_run:
        return

    # Download
    success = 0
    fail = 0
    for name, hf_id, cls_name in models:
        if download_model(hf_id, f"{name} ({cls_name})"):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"Done: {success} succeeded, {fail} failed, {success + fail} total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
