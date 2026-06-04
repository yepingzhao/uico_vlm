"""vLLM-based VLM wrappers for Qwen2.5-VL and LLaVA-1.5.

Uses vLLM's offline LLM.chat() API for multimodal inference with
automatic chat template handling.
"""

from vllm import LLM, SamplingParams

from .base import VLMWrapper
from ..config import (
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_SEQS,
    VLLM_ENFORCE_EAGER,
    VLLM_LIMIT_MM_PER_PROMPT,
)


def _build_sampling_params(kwargs) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=kwargs.get("max_new_tokens", 128),
    )


class Qwen2VLVLLMWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "qwen2vl-vllm"

    def load(self, device: str = "cuda:0"):
        self._model = LLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            enforce_eager=VLLM_ENFORCE_EAGER,
            limit_mm_per_prompt=VLLM_LIMIT_MM_PER_PROMPT,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        sampling_params = _build_sampling_params(kwargs)
        outputs = self._model.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()


class LLaVAVLLMWrapper(VLMWrapper):

    @property
    def model_name(self) -> str:
        return "llava-vllm"

    def load(self, device: str = "cuda:0"):
        self._model = LLM(
            model="llava-hf/llava-1.5-7b-hf",
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            enforce_eager=VLLM_ENFORCE_EAGER,
            limit_mm_per_prompt=VLLM_LIMIT_MM_PER_PROMPT,
        )

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self._validate_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        sampling_params = _build_sampling_params(kwargs)
        outputs = self._model.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()
