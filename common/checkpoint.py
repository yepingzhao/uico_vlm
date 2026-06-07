"""Prediction file checkpoint/resume helpers.

Moved from models/utils.py to keep models/ free of I/O concerns
not related to model loading/inference.
"""

import json
import os


def load_checkpoint(output_path: str) -> set:
    """Read existing predictions JSONL and return set of processed image_ids.

    Args:
        output_path: Path to the predictions JSONL file.

    Returns:
        Set of image_id integers already processed.
    """
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        processed.add(record["image_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return processed
