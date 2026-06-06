"""Training dataset and collate function for VLM instruction fine-tuning."""

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_BASE
from data.dataset import resolve_image_path as _resolve_image_path

IMAGES_BASE_DIR = os.path.join(DATA_BASE, "images")

# InternVL2 uses separate tokenizer + image processor (AutoProcessor returns
# only the tokenizer). The chat template expects plain-string content and the
# <image> placeholder must be expanded to <img><IMG_CONTEXT>×256</img>
# before tokenization.
_INTERNVL2_IMG_START = "<img>"
_INTERNVL2_IMG_END = "</img>"
_INTERNVL2_IMG_CONTEXT = "<IMG_CONTEXT>"
_INTERNVL2_NUM_IMAGE_TOKENS = 256  # 448px, patch=14, downsample=0.5


def _detect_image_token_ids(processor, is_internvl2: bool = False) -> set:
    """Auto-detect image-related token IDs from a VLM processor.

    Different VLMs use different token IDs for image placeholders
    (LLaVA: 32000 for <image>, Qwen2.5-VL: 151654/151655 for
    <|vision_pad|>/<|image_pad|>). We probe the tokenizer for common
    image token names rather than hardcoding per-model IDs.
    """
    ids = set()
    # Resolve the tokenizer: for InternVL2, processor IS the tokenizer
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if is_internvl2:
        # InternVL2 uses <img>, </img>, <IMG_CONTEXT> as image tokens
        for t in (_INTERNVL2_IMG_START, _INTERNVL2_IMG_END, _INTERNVL2_IMG_CONTEXT):
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != tok.unk_token_id:
                ids.add(tid)
        return ids
    candidates = [
        "<image>",           # LLaVA / LLaVA-NeXT
        "<|image_pad|>",     # Qwen2.5-VL
        "<|vision_pad|>",    # Qwen2.5-VL
        "<|vision_start|>",  # Qwen2.5-VL
        "<|vision_end|>",    # Qwen2.5-VL
        "<image_soft_token>",  # InternVL2 (legacy)
    ]
    for token_name in candidates:
        tid = tok.convert_tokens_to_ids(token_name)
        if isinstance(tid, int) and tid != tok.unk_token_id:
            ids.add(tid)
    # Phi-3.5-Vision: <|image_1|> tokenizes to multiple sub-tokens
    phi_image_ids = tok.encode("<|image_1|>", add_special_tokens=False)
    ids.update(phi_image_ids)
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
        is_internvl2: bool = False,
        num_image_tokens: int = _INTERNVL2_NUM_IMAGE_TOKENS,
    ):
        self.processor = processor
        self.user_prompt = user_prompt
        self.is_internvl2 = is_internvl2
        self.num_image_tokens = num_image_tokens
        self._image_token_ids = _detect_image_token_ids(
            processor, is_internvl2=is_internvl2)
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

        if self.is_internvl2:
            return self._getitem_internvl2(image, caption)
        return self._getitem_standard(image, caption)

    def _getitem_standard(self, image: Image.Image, caption: str) -> dict:
        """Standard VLM processing: structured content + processor(image, text)."""
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
        user_only = self.processor.apply_chat_template(
            [conversation[0]], add_generation_prompt=True
        )
        user_inputs = self.processor(
            images=image, text=user_only, return_tensors="pt",
        )
        user_ids = user_inputs["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[:len(user_ids)] = -100
        for tid in self._image_token_ids:
            labels[input_ids == tid] = -100

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0),
            "image_grid_thw": inputs.get("image_grid_thw", torch.tensor([[1, 1, 1]])).squeeze(0),
            "image_sizes": inputs.get("image_sizes"),
            "mm_token_type_ids": (
                inputs["mm_token_type_ids"].squeeze(0)
                if inputs.get("mm_token_type_ids") is not None
                else None
            ),
            "image_flags": None,
        }

    def _getitem_internvl2(self, image: Image.Image, caption: str) -> dict:
        """InternVL2-specific processing.

        InternVL2's AutoProcessor returns only the tokenizer (no image
        processor), and apply_chat_template returns token IDs (not text)
        because bos_token renders as an integer. We must:
        (1) expand <image> → <img><IMG_CONTEXT>×256</img> before the
            chat template,
        (2) convert the template's token-ID list to tensors directly,
        (3) process images via CLIPImageProcessor,
        (4) emit image_flags for the model forward.
        """
        tokenizer = self.processor  # processor IS the tokenizer for InternVL2
        image_processor = self.image_processor

        img_tokens = (
            f"{_INTERNVL2_IMG_START}"
            f"{_INTERNVL2_IMG_CONTEXT * self.num_image_tokens}"
            f"{_INTERNVL2_IMG_END}"
        )

        # 1. Build user-only token IDs (for label masking)
        user_conv = [
            {"role": "user", "content": f"{img_tokens}\n{self.user_prompt}"},
        ]
        user_ids = tokenizer.apply_chat_template(
            user_conv, add_generation_prompt=True)
        user_ids = torch.tensor(user_ids)

        # 2. Build full conversation token IDs
        full_conv = [
            {"role": "user", "content": f"{img_tokens}\n{self.user_prompt}"},
            {"role": "assistant", "content": caption},
        ]
        input_ids = tokenizer.apply_chat_template(
            full_conv, add_generation_prompt=False)
        input_ids = torch.tensor(input_ids)

        # 3. Process image with CLIP image processor
        img_outputs = image_processor(images=image, return_tensors="pt")
        pixel_values = img_outputs["pixel_values"].squeeze(0)
        # pixel_values shape: [3, 448, 448]

        # 4. Build labels: mask user prefix + image tokens
        labels = input_ids.clone()
        labels[:len(user_ids)] = -100
        for tid in self._image_token_ids:
            labels[input_ids == tid] = -100

        # 5. image_flags: marks samples with images (always 1 for single-image)
        image_flags = torch.tensor([1])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "image_grid_thw": None,
            "image_sizes": None,
            "mm_token_type_ids": None,
            "image_flags": image_flags,
        }


def collate_fn(processor, batch):
    """Pad variable-length sequences and stack pixel values."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # image_grid_thw: None for InternVL2 (no dynamic resolution)
    has_grid = all(
        item.get("image_grid_thw") is not None for item in batch
    )
    image_grid_thw = (
        torch.stack([item["image_grid_thw"] for item in batch])
        if has_grid else None
    )

    # LLaVA-NeXT dynamic high resolution: stack image_sizes if present
    has_image_sizes = all(
        item.get("image_sizes") is not None for item in batch
    )
    image_sizes = (
        torch.stack([item["image_sizes"].squeeze(0) for item in batch])
        if has_image_sizes else None
    )

    # InternVL2 image_flags: marks which batch elements contain images
    has_image_flags = all(
        item.get("image_flags") is not None for item in batch
    )
    image_flags = (
        torch.stack([item["image_flags"] for item in batch])
        if has_image_flags else None
    )

    max_len = max(item["input_ids"].size(0) for item in batch)
    # Resolve pad_token_id: for InternVL2, processor IS the tokenizer
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_token_id = tok.pad_token_id

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
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels": torch.stack(labels_list),
    }
    if image_grid_thw is not None:
        result["image_grid_thw"] = image_grid_thw
    if has_image_sizes:
        result["image_sizes"] = image_sizes
    if has_mm_tokens:
        result["mm_token_type_ids"] = torch.stack(mm_token_list)
    if has_image_flags:
        result["image_flags"] = image_flags
    return result
