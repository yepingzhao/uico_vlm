#!/usr/bin/env python3
"""Run zero-shot VLM inference on UICO test set with checkpoint/resume support.

Usage:
    # Phase 1 (dev): 1000 images, 2 models
    python run_inference.py --models blip2 llava --subsample 1000 --prompt A

    # Phase 2 (full): 3500 images, all models
    python run_inference.py --models blip2 instructblip llava internvl2 qwen2vl --prompt A

    # Sensitivity analysis: B/C prompts on LLaVA + Qwen2.5-VL
    python run_inference.py --models llava qwen2vl --prompt B
    python run_inference.py --models llava qwen2vl --prompt C
    python run_inference.py --models qwen2vl --prompt ZH

    # vLLM backend: faster inference for LLaVA + Qwen2.5-VL
    python run_inference.py --models llava-vllm qwen2vl-vllm --prompt A
    python run_inference.py --models qwen2vl qwen2vl-vllm --prompt A  # compare HF vs vLLM
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OUTPUT_DIR,
    MODEL_REGISTRY,
    PROMPTS,
    DEV_SAMPLE_SIZE,
    RANDOM_SEED,
    MAX_NEW_TOKENS,
)
from data.dataset import load_test_dataset
from prompts.templates import PROMPT_MAP
from models.utils import load_checkpoint


# Map short name → wrapper class
def _get_wrapper(name: str):
    from models.blip2 import BLIP2Wrapper
    from models.instructblip import InstructBLIPWrapper
    from models.llava import LLaVAWrapper
    from models.internvl2 import InternVL2Wrapper
    from models.qwen2vl import Qwen2VLWrapper
    from models.qwen3vl import Qwen3VLWrapper
    from models.phi35_vision import Phi35VisionWrapper
    from models.phi4_multimodal import Phi4MultimodalWrapper
    from models.paligemma2 import PaliGemma2Wrapper
    from models.minicpm_v import MiniCPMVWrapper
    from models.llava_next import LLaVANeXTWrapper
    # Models that may fail with older transformers versions
    try:
        from models.idefics3 import Idefics3Wrapper
        _idefics3_ok = True
    except ImportError:
        _idefics3_ok = False
    from models.internvl25 import InternVL25Wrapper
    from models.pixtral import PixtralWrapper
    from models.llama32_vision import Llama32VisionWrapper

    registry = {
        "blip2": BLIP2Wrapper,
        "instructblip": InstructBLIPWrapper,
        "llava": LLaVAWrapper,
        "internvl2": InternVL2Wrapper,
        "qwen2vl": Qwen2VLWrapper,
        "qwen3vl": Qwen3VLWrapper,
        "phi35-vision": Phi35VisionWrapper,
        "phi4-mm": Phi4MultimodalWrapper,
        "paligemma2": PaliGemma2Wrapper,
        "minicpm-v": MiniCPMVWrapper,
        "llava-next": LLaVANeXTWrapper,
        "internvl25": InternVL25Wrapper,
        "pixtral": PixtralWrapper,
        "llama32-vision": Llama32VisionWrapper,
    }
    if _idefics3_ok:
        registry["idefics3"] = Idefics3Wrapper

    # DeepSeek-VL2 requires external package — optional
    try:
        from models.deepseek_vl2 import DeepSeekVL2Wrapper
        registry["deepseek-vl2"] = DeepSeekVL2Wrapper
    except ImportError:
        pass

    # Optional vLLM backends — only load if vllm is installed
    try:
        from models.vllm_wrapper import Qwen2VLVLLMWrapper, LLaVAVLLMWrapper
        registry["qwen2vl-vllm"] = Qwen2VLVLLMWrapper
        registry["llava-vllm"] = LLaVAVLLMWrapper
    except ImportError:
        pass

    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(registry.keys())}")
    return registry[name]()


def run_inference(
    model_names: list,
    prompt_key: str,
    subsample: int = 0,
    device: str = "cuda:0",
):
    """Run inference for each model on the test set.

    Args:
        model_names: List of short model names (e.g. ["blip2", "llava"]).
        prompt_key: Which prompt to use ("A", "B", "C", "ZH").
        subsample: Number of images to use (0 = full test set).
        device: CUDA device string.
    """
    prompt_text = PROMPT_MAP[prompt_key]
    print(f"[Prompt] {prompt_key}: {prompt_text[:100]}...")

    # Load dataset
    ds = load_test_dataset(subsample=subsample, seed=RANDOM_SEED)
    print(f"[Data] Loaded {len(ds)} images")

    # Build image_path map
    image_paths = {img_id: ds.get_image_path(img_id) for img_id in ds.image_ids}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"[Model] {model_name}")
        print(f"{'='*60}")

        # Output path
        model_out_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        pred_file = os.path.join(
            model_out_dir, f"predictions_prompt_{prompt_key.lower()}.jsonl"
        )

        # Resume
        processed = load_checkpoint(pred_file)
        remaining = [i for i in ds.image_ids if i not in processed]
        print(f"[Resume] {len(processed)} done, {len(remaining)} remaining")

        if not remaining:
            print("[Skip] All images already processed.")
            continue

        # Load model
        wrapper = _get_wrapper(model_name)
        print(f"[Load] Loading {model_name} on {device} ...")
        t0 = time.time()
        wrapper.load(device=device)
        print(f"[Load] Done in {time.time() - t0:.1f}s")

        # Inference loop
        with open(pred_file, "a") as f_out:
            for i, img_id in enumerate(remaining):
                img_path = image_paths[img_id]
                try:
                    caption = wrapper.generate(
                        img_path, prompt_text, max_new_tokens=MAX_NEW_TOKENS
                    )
                except Exception as e:
                    print(f"  [ERROR] image_id={img_id}: {e}", file=sys.stderr)
                    caption = ""

                record = {
                    "image_id": img_id,
                    "file_name": os.path.basename(img_path),
                    "caption": caption,
                    "prompt": prompt_key,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if (i + 1) % 10 == 0:
                    f_out.flush()
                    print(f"  [{i+1}/{len(remaining)}] {caption[:80]}...", flush=True)

        # Clean up
        del wrapper
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

        print(f"[Done] {model_name} → {pred_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Zero-Shot Inference")
    parser.add_argument(
        "--models", nargs="+", default=["blip2", "llava"],
        help="Model short names to run."
    )
    parser.add_argument(
        "--prompt", type=str, default="A",
        choices=["A", "B", "C", "ZH"],
        help="Prompt template key."
    )
    parser.add_argument(
        "--subsample", type=int, default=0,
        help="Number of images to subsample (0 = full test set)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device."
    )
    args = parser.parse_args()

    print(f"[Config] models={args.models}, prompt={args.prompt}, "
          f"subsample={args.subsample or 'full'}, device={args.device}")

    run_inference(
        model_names=args.models,
        prompt_key=args.prompt,
        subsample=args.subsample,
        device=args.device,
    )
