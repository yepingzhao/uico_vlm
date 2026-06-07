"""Prompt templates for VLM zero-shot urban incivility captioning.

Prompt design rationale (see docs/research-notes/2026-06-04-prompt-gt-alignment-analysis.md):
- GT captions are concise (median 9 words), factual, and follow a WHAT+WHERE structure.
- Prompt A is the primary prompt: concise what+where, domain priming, negative constraint.
- Prompt B isolates format sensitivity: same content as A, structured output format.
- Prompt C isolates content sensitivity: same format as A, adds "why" justification.
- Each pair (A→B, A→C) varies a single dimension, enabling clean attribution.
- PROMPT_FEWSHOT is identical to Prompt A, so zero-shot vs few-shot comparison
  isolates only the effect of examples (no textual difference in the prompt).
- IMPORTANT: B and C are evaluated with ref-free metrics ONLY (CLIPScore, RefCLIPScore).
  Their format/content differences mechanically deflate n-gram metrics; comparing
  A/B/C via ref-based metrics would conflate format noise with sensitivity signal.
"""

# --- Primary prompt ---
# Prompt A: concise what+where description (primary, used for all models)
PROMPT_A = (
    "Describe any violation of urban order visible in this image "
    "in one short sentence. State what is wrong "
    "(e.g., illegal parking, garbage, clutter, damaged infrastructure) "
    "and where it is located. Do not explain why or analyze the situation."
)

# --- Sensitivity analysis prompts ---
# Prompt B: format ablation — same content as A, structured output format
PROMPT_B = (
    "Describe any violation of urban order in this image using this format:\n"
    "Violation: [what the problem is]\n"
    "Location: [where it occurs]"
)

# Prompt C: content ablation — same format as A, adds normative justification
PROMPT_C = (
    "In one or two sentences, describe any violation of urban order visible "
    "in this image. Include what the problem is, where it is located, and "
    "why it violates urban norms."
)

# --- Few-shot prompt ---
# Identical to Prompt A — zero-shot vs few-shot comparison isolates only the
# effect of examples, with no textual difference in the prompt itself.
PROMPT_FEWSHOT = PROMPT_A  # noqa — alias for semantic clarity in code

# Mapping for config
PROMPT_MAP = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
    "FS": PROMPT_FEWSHOT,
}
