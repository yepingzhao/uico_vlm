"""Sample few-shot examples from the UICO training set.

For reproducibility, we sample once with a fixed seed and reuse the same
examples for all test images (static-example approach).
"""

import json
import os
import random
from typing import List, Tuple

from config import DATA_BASE, IMAGES_BASE_DIR, RANDOM_SEED
from data.dataset import resolve_image_path


def load_training_captions() -> List[Tuple[int, str, str]]:
    """Load training set images and captions.

    Returns:
        List of (image_id, file_name, caption) tuples.
        There are 5 captions per image, so images repeat.
    """
    ann_file = os.path.join(DATA_BASE, "annotations", "captions_train.json")
    with open(ann_file) as f:
        data = json.load(f)

    # Build image_id → file_name map
    id_to_file = {}
    for img in data["images"]:
        id_to_file[img["id"]] = img["file_name"]

    examples = []
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        file_name = id_to_file[image_id]
        caption = ann["caption"].strip()
        examples.append((image_id, file_name, caption))

    return examples


def sample_examples(
    k: int,
    seed: int = RANDOM_SEED,
    cache_dir: str = None,
) -> List[Tuple[str, str]]:
    """Sample k diverse examples (different images) from the training set.

    Args:
        k: Number of few-shot examples to sample.
        seed: Random seed for reproducibility.
        cache_dir: If provided, save/load examples from this dir.

    Returns:
        List of (image_path, caption) tuples.
    """
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"fewshot_examples_k{k}_seed{seed}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
            # Verify files still exist
            valid = all(os.path.exists(p) for p, _ in cached)
            if valid:
                return cached

    rng = random.Random(seed)
    all_examples = load_training_captions()

    # Group by image_id, pick one random caption per image
    by_image = {}
    for img_id, fname, caption in all_examples:
        by_image.setdefault(img_id, []).append((fname, caption))

    image_ids = list(by_image.keys())
    rng.shuffle(image_ids)

    selected = []
    for img_id in image_ids:
        if len(selected) >= k:
            break
        fname, caption = rng.choice(by_image[img_id])
        img_path = resolve_image_path(IMAGES_BASE_DIR, fname)
        if os.path.exists(img_path):
            selected.append((img_path, caption))

    if len(selected) < k:
        raise RuntimeError(
            f"Only found {len(selected)} valid training images (requested {k})"
        )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(selected, f)

    return selected
