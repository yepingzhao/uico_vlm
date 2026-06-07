"""Few-shot in-context learning for VLM evaluation.

Modules:
  - sampler: example selection from the training set
  - content: shared content-block construction for few-shot prompts
"""

from .sampler import sample_examples
from .content import build_fewshot_images_and_content

__all__ = ["sample_examples", "build_fewshot_images_and_content"]
