"""COCO-format dataset loader for UICO test set."""

import json
import os
import random
from typing import List, Dict

from pycocotools.coco import COCO


# Image files are split across subdirectories by prefix:
#   CCMC_train_NNN.jpg → ccmc_train/
#   CCMC_test_NNN.jpg  → ccmc_test/
#   CCMC_val_NNN.jpg   → ccmc_val/
PREFIX_TO_SUBDIR = {
    "CCMC_train": "ccmc_train",
    "CCMC_test": "ccmc_test",
    "CCMC_val": "ccmc_val",
}


def resolve_image_path(images_base_dir: str, file_name: str) -> str:
    """Map a COCO file_name to its actual filesystem path."""
    # file_name example: "CCMC_train_000000026952.jpg"
    parts = file_name.rsplit("_", 2)  # ["CCMC", "train", "000000026952.jpg"]
    if len(parts) >= 2:
        prefix = parts[0] + "_" + parts[1]  # "CCMC_train"
    else:
        prefix = file_name.split("_")[0] + "_" + file_name.split("_")[1]

    subdir = PREFIX_TO_SUBDIR.get(prefix, "ccmc_test")
    return os.path.join(images_base_dir, subdir, file_name)


class UICOTestDataset:
    """Load UICO test set in COCO format, with optional sub-sampling for dev."""

    def __init__(self, ann_file: str, images_base_dir: str):
        self.ann_file = ann_file
        self.images_base_dir = images_base_dir
        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self._img_id_to_file: Dict[int, str] = {}
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            self._img_id_to_file[img_id] = img_info["file_name"]

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_image_path(self, image_id: int) -> str:
        return resolve_image_path(self.images_base_dir, self._img_id_to_file[image_id])

    def get_references(self, image_id: int) -> List[str]:
        """Return all 5 reference captions for an image."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        return [ann["caption"] for ann in anns]

    def subsample(self, n: int, seed: int = 42) -> "UICOTestDataset":
        """Return a new dataset with a random subset of n images."""
        rng = random.Random(seed)
        subset_ids = sorted(rng.sample(self.image_ids, min(n, len(self.image_ids))))
        new_ds = UICOTestDataset.__new__(UICOTestDataset)
        new_ds.ann_file = self.ann_file
        new_ds.images_base_dir = self.images_base_dir
        new_ds.coco = self.coco
        new_ds.image_ids = subset_ids
        new_ds._img_id_to_file = {k: self._img_id_to_file[k] for k in subset_ids}
        return new_ds

    def all_references_dict(self) -> Dict[int, List[str]]:
        """Return {image_id: [ref1, ref2, ...]} for all images in this dataset."""
        result = {}
        for img_id in self.image_ids:
            result[img_id] = self.get_references(img_id)
        return result


def load_test_dataset(subsample: int = 0, seed: int = 42) -> UICOTestDataset:
    """Convenience: load the full test set, optionally subsample."""
    from config import TEST_ANN_FILE, IMAGES_BASE_DIR
    ds = UICOTestDataset(TEST_ANN_FILE, IMAGES_BASE_DIR)
    if subsample > 0:
        ds = ds.subsample(subsample, seed)
    return ds
