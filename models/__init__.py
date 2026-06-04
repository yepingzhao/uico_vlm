"""Model wrapper registry — single source of truth for model instantiation."""


def get_wrapper(name: str):
    """Return an instantiated model wrapper by short name.

    Args:
        name: Short model name (e.g. "blip2", "llava", "qwen2vl").

    Returns:
        A VLMWrapper instance.

    Raises:
        ValueError: If the model name is unknown.
    """
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

    # Models with optional dependencies — import failures are non-fatal
    try:
        from models.idefics3 import Idefics3Wrapper
        registry["idefics3"] = Idefics3Wrapper
    except ImportError:
        pass

    try:
        from models.deepseek_vl2 import DeepSeekVL2Wrapper
        registry["deepseek-vl2"] = DeepSeekVL2Wrapper
    except ImportError:
        pass

    try:
        from models.vllm_wrapper import Qwen2VLVLLMWrapper, LLaVAVLLMWrapper
        registry["qwen2vl-vllm"] = Qwen2VLVLLMWrapper
        registry["llava-vllm"] = LLaVAVLLMWrapper
    except ImportError:
        pass

    if name not in registry:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(registry.keys())}"
        )
    return registry[name]()
