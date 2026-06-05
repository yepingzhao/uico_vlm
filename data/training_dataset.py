"""Training dataset and collate function for VLM instruction fine-tuning."""

import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_BASE
from data.dataset import resolve_image_path as _resolve_image_path

IMAGES_BASE_DIR = os.path.join(DATA_BASE, "images")


def _detect_image_token_ids(processor) -> set:
    """Auto-detect image-related token IDs from a VLM processor.

    Different VLMs use different token IDs for image placeholders
    (LLaVA: 32000 for <image>, Qwen2.5-VL: 151654/151655 for
    <|vision_pad|>/<|image_pad|>). We probe the tokenizer for common
    image token names rather than hardcoding per-model IDs.
    """
    ids = set()
    tok = processor.tokenizer
    candidates = [
        "<image>",           # LLaVA / LLaVA-NeXT
        "<|image_pad|>",     # Qwen2.5-VL
        "<|vision_pad|>",    # Qwen2.5-VL
        "<|vision_start|>",  # Qwen2.5-VL
        "<|vision_end|>",    # Qwen2.5-VL
        "<image_soft_token>",  # InternVL2
    ]
    for token_name in candidates:
        tid = tok.convert_tokens_to_ids(token_name)
        if isinstance(tid, int) and tid != tok.unk_token_id:
            ids.add(tid)
    if not ids:
        # Fallback for unknown models
        ids = {32000}
    return ids


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
        self._image_token_ids = _detect_image_token_ids(processor)
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
            idx = int(torch.randint(0, len(captions), (1,), generator=rng).item())
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

        # Mask user prompt + image tokens; train only on assistant response.
        # CRITICAL: user-only prefix must be tokenized WITH the image so that
        # image placeholder expansion (e.g. <image> → 576 tokens in LLaVA) is
        # consistent between user_ids and input_ids. Otherwise the prefix length
        # misaligns and prompt text leaks into the loss (see commit message).
        user_only = self.processor.apply_chat_template(
            [conversation[0]], add_generation_prompt=True
        )
        user_inputs = self.processor(
            images=image, text=user_only, return_tensors="pt",
        )
        user_ids = user_inputs["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[:len(user_ids)] = -100
        # Mask any remaining image placeholder tokens (model-specific IDs)
        for tid in self._image_token_ids:
            labels[input_ids == tid] = -100

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0),
            "image_grid_thw": inputs.get("image_grid_thw", torch.tensor([[1, 1, 1]])).squeeze(0),
            "mm_token_type_ids": (
                inputs["mm_token_type_ids"].squeeze(0)
                if inputs.get("mm_token_type_ids") is not None
                else None
            ),
        }


def collate_fn(processor, batch):
    """Pad variable-length sequences and stack pixel values."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    image_grid_thw = torch.stack([item["image_grid_thw"] for item in batch])

    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_token_id = processor.tokenizer.pad_token_id

    # Check whether this model provides multimodal token type IDs (Qwen2.5-VL MRoPE)
    has_mm_tokens = all(
        item.get("mm_token_type_ids") is not None for item in batch
    )

    input_ids_list, labels_list, mask_list = [], [], []
    mm_token_list = [] if has_mm_tokens else None
    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        am = item["attention_mask"]
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id)])
            labs = torch.cat([labs, torch.full((pad_len,), -100)])
            am = torch.cat([am, torch.zeros(pad_len, dtype=am.dtype)])
        input_ids_list.append(ids)
        labels_list.append(labs)
        mask_list.append(am)
        if has_mm_tokens:
            mm = item["mm_token_type_ids"]
            if pad_len > 0:
                mm = torch.cat([mm, torch.zeros(pad_len, dtype=mm.dtype)])
            mm_token_list.append(mm)

    result = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels": torch.stack(labels_list),
    }
    if has_mm_tokens:
        result["mm_token_type_ids"] = torch.stack(mm_token_list)
    return result
