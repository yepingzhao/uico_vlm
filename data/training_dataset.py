"""Training dataset and collate function for VLM instruction fine-tuning."""

import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_BASE
from data.dataset import _resolve_image_path

IMAGES_BASE_DIR = os.path.join(DATA_BASE, "images")


class UICOInstructionDataset(Dataset):
    """UICO training set in instruction-following format for VLMs.

    Each image has 5 reference captions. One caption is randomly
    selected per image at init (fixed seed) for stable training.
    """

    def __init__(
        self,
        ann_file: str,
        processor,
        user_prompt: str = (
            "In one sentence, describe any violation of urban order visible in "
            "this image. State what the problem is and where it is located."
        ),
        max_samples: int = 0,
        seed: int = 42,
    ):
        self.processor = processor
        self.user_prompt = user_prompt
        self.samples: List[tuple] = []

        with open(ann_file) as f:
            data = json.load(f)

        id_to_file: Dict[int, str] = {
            img["id"]: img["file_name"] for img in data["images"]
        }

        by_image: Dict[int, list] = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(
                ann["caption"].strip()
            )

        rng = torch.Generator().manual_seed(seed)
        image_ids = sorted(by_image.keys())
        if max_samples > 0:
            image_ids = image_ids[:max_samples]

        for img_id in image_ids:
            fname = id_to_file[img_id]
            path = _resolve_image_path(IMAGES_BASE_DIR, fname)
            if not os.path.exists(path):
                continue
            captions = by_image[img_id]
            idx = torch.randint(0, len(captions), (1,), generator=rng).item()
            self.samples.append((path, captions[idx]))

        print(f"[Dataset] {len(self.samples)} training examples "
              f"({len(image_ids)} images)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)

        # Mask user prompt + image tokens; train only on assistant response
        user_only = self.processor.apply_chat_template(
            [conversation[0]], add_generation_prompt=True
        )
        user_ids = self.processor.tokenizer(
            user_only, return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[:len(user_ids)] = -100
        labels[input_ids == 32000] = -100  # image placeholder tokens

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(processor, batch):
    """Pad variable-length sequences and stack pixel values."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_token_id = processor.tokenizer.pad_token_id

    input_ids_list, labels_list = [], []
    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id)])
            labs = torch.cat([labs, torch.full((pad_len,), -100)])
        input_ids_list.append(ids)
        labels_list.append(labs)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(input_ids_list).ne(pad_token_id).long(),
        "labels": torch.stack(labels_list),
    }
