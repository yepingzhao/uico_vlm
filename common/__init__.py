"""Shared infrastructure for VLM inference and evaluation pipelines.

Modules:
  - checkpoint: prediction file checkpoint/resume helpers
  - dataset_bundle: lightweight dataset wrapper decoupled from scripts
  - pipeline: InferenceRunner orchestration
  - strategies: GenerationStrategy ABC and concrete implementations
  - eval_core: shared evaluation functions
"""
