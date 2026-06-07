"""Shared utilities for model wrappers and inference scripts."""

import glob
import os


def find_snapshot_dir(model_id: str) -> str:
    """Find the HF cache snapshot directory for a model.

    Args:
        model_id: HuggingFace model ID (e.g. "OpenGVLab/InternVL2-8B").

    Returns:
        Path to the latest snapshot directory in the HF cache.

    Raises:
        FileNotFoundError: If no snapshot directory exists for the model.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = model_id.replace("/", "--")
    snapshots_dir = os.path.join(hf_home, "hub", "models--" + model_dir, "snapshots")
    dirs = sorted(glob.glob(os.path.join(snapshots_dir, "*")))
    if not dirs:
        raise FileNotFoundError(
            f"No snapshot found for {model_id} in {snapshots_dir}"
        )
    return dirs[-1]


# load_checkpoint moved to core/inference/checkpoint.py (decouple I/O from models/)
