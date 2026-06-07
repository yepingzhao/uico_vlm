"""DatasetBundle — value object that decouples dataset access from scripts.

Every inference script repeated the same pattern:
    ds = load_test_dataset(subsample=..., seed=...)
    image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}

DatasetBundle captures this in one place.
"""

from typing import Dict


class DatasetBundle:
    """Lightweight wrapper holding image IDs and their filesystem paths.

    Created from a UICOTestDataset and consumed by InferenceRunner.
    Runner only needs image_ids (for iteration order) and image_paths
    (for passing paths to the strategy) — it never touches the dataset
    or the COCO API directly.
    """

    def __init__(self, image_ids: list, image_paths: Dict[int, str]):
        self.image_ids = image_ids
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_ids)

    @classmethod
    def from_dataset(cls, ds):
        """Build a DatasetBundle from a UICOTestDataset instance.

        Args:
            ds: UICOTestDataset with .image_ids and .get_image_path().

        Returns:
            DatasetBundle ready for InferenceRunner.
        """
        image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}
        return cls(image_ids=list(ds.image_ids), image_paths=image_paths)
