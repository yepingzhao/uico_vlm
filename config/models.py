"""Model registry and sensitivity analysis configuration."""

# Each entry: (short_name, hf_model_id, wrapper_class_name)
MODEL_REGISTRY = [
    ("blip2",           "Salesforce/blip2-flan-t5-xl",            "BLIP2Wrapper"),
    ("instructblip",    "Salesforce/instructblip-vicuna-7b",      "InstructBLIPWrapper"),
    ("llava",           "llava-hf/llava-1.5-7b-hf",              "LLaVAWrapper"),
    ("internvl2",       "OpenGVLab/InternVL2-8B",                 "InternVL2Wrapper"),
    ("qwen2vl",         "Qwen/Qwen2.5-VL-7B-Instruct",            "Qwen2VLWrapper"),
    # New VLM baselines
    ("phi35-vision",    "microsoft/Phi-3.5-vision-instruct",       "Phi35VisionWrapper"),
    ("phi4-mm",         "microsoft/Phi-4-multimodal-instruct",     "Phi4MultimodalWrapper"),
    ("paligemma2",      "google/paligemma2-3b-ft-docci-448",      "PaliGemma2Wrapper"),
    ("minicpm-v",       "openbmb/MiniCPM-V-2_6",                  "MiniCPMVWrapper"),
    ("deepseek-vl2",    "deepseek-ai/deepseek-vl2-small",          "DeepSeekVL2Wrapper"),
    ("llava-next",      "llava-hf/llava-v1.6-mistral-7b-hf",      "LLaVANeXTWrapper"),
    ("idefics3",        "HuggingFaceM4/Idefics3-8B-Llama3",       "Idefics3Wrapper"),
    ("internvl25",      "OpenGVLab/InternVL2_5-8B",               "InternVL25Wrapper"),
    ("internvl35",      "OpenGVLab/InternVL3_5-8B",              "InternVL35Wrapper"),
    ("pixtral",         "mistralai/Pixtral-12B-2409",             "PixtralWrapper"),
    ("llama32-vision",  "meta-llama/Llama-3.2-11B-Vision-Instruct","Llama32VisionWrapper"),
    ("qwen3vl",         "Qwen/Qwen3-VL-8B-Instruct",               "Qwen3VLWrapper"),
]

# Sensitivity analysis: only run B/C prompts on these models
SENSITIVITY_MODELS = ["llava", "qwen2vl"]
