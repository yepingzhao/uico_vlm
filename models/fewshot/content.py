"""Shared few-shot content-block construction logic.

Four models support few-shot (LLaVA, LLaVA-NeXT, Qwen2VL, Qwen3VL).
They all interleave example images and captions the same way but
differ in how images are attached to content blocks and how the
processor is invoked.

This module provides:
  - build_fewshot_images_and_content: builds (all_images, content_blocks)
"""

from PIL import Image


def build_fewshot_images_and_content(
    test_image_path: str,
    prompt_template: str,
    example_images: list,
    example_captions: list,
    *,
    embed_images: bool = False,
):
    """Build (all_images, content_blocks) for few-shot inference.

    Args:
        test_image_path: Path to the test image.
        prompt_template: Text prompt with the instruction.
        example_images: List of paths to example images.
        example_captions: List of example captions (same length).
        embed_images: If True, embed PIL Image objects in content blocks
                      (Qwen-style). If False, use {"type": "image"} placeholder
                      (LLaVA-style), with images passed separately.

    Returns:
        Tuple of (all_images: list[Image], content_blocks: list[dict]).
    """
    all_images = []
    content_blocks = []

    for i, (ex_img_path, ex_caption) in enumerate(
        zip(example_images, example_captions)
    ):
        ex_img = Image.open(ex_img_path).convert("RGB")
        all_images.append(ex_img)
        if embed_images:
            content_blocks.append({"type": "image", "image": ex_img})
        else:
            content_blocks.append({"type": "image"})
        content_blocks.append({
            "type": "text",
            "text": f"Example {i + 1}: {ex_caption}",
        })

    test_img = Image.open(test_image_path).convert("RGB")
    all_images.append(test_img)
    if embed_images:
        content_blocks.append({"type": "image", "image": test_img})
    else:
        content_blocks.append({"type": "image"})
    content_blocks.append({
        "type": "text",
        "text": prompt_template,
    })

    return all_images, content_blocks
