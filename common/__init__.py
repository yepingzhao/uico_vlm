"""Shared infrastructure for VLM inference and evaluation pipelines.

Modules:
  - checkpoint: prediction file checkpoint/resume helpers

  - pipeline: InferenceRunner orchestration
  - strategies: GenerationStrategy ABC and concrete implementations
  - training: TrainingRunner QLoRA fine-tuning loop
  - training_adapter: TrainingModelAdapter ABC + per-model-family adapters
  - eval_core: shared evaluation functions
"""
