"""Prompt templates for VLM zero-shot urban incivility captioning."""

# --- English prompts ---
# Prompt A: concise description (primary, used for all models)
PROMPT_A = (
    "Describe any urban incivility or civic norm violations visible in "
    "this image in one or two sentences."
)

# Prompt B: structured description (sensitivity analysis)
PROMPT_B = (
    "Analyze this urban scene and describe:\n"
    "(1) what type of civic norm violation is present,\n"
    "(2) where it is located,\n"
    "(3) why it constitutes an incivility."
)

# Prompt C: governance-oriented (sensitivity analysis)
PROMPT_C = (
    "You are an urban management inspector. Describe the urban incivility "
    "in this image, focusing on the specific violation, its spatial "
    "context, and its impact on public order."
)

# --- Chinese prompt ---
# Supplemental: for Qwen2.5-VL which is Chinese-optimized
PROMPT_ZH = (
    "请描述这张图片中存在的城市不文明现象或违反城市管理规范的行为。"
)

# Mapping for config
PROMPT_MAP = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
    "ZH": PROMPT_ZH,
}
