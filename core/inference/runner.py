"""InferenceRunner — orchestrates the inference loop.

The Runner owns the "framework work": checkpoint/resume, iteration,
JSONL writing, progress logging. It delegates generation to a
GenerationStrategy and knows nothing about models, prompts, or
generation mechanisms.
"""

import json
import os
import sys
import time

from core.inference.checkpoint import load_checkpoint
from core.inference.strategies import GenerationStrategy


class InferenceRunner:
    """Orchestrates the inference loop.

    Responsibilities:
      - Checkpoint/resume (read processed image_ids from JSONL)
      - Iterate over remaining image_ids
      - Call strategy.generate() for each
      - Write JSONL records with progress flushing
      - Call strategy.prepare() / cleanup() at boundaries
    """

    def __init__(
        self,
        strategy: GenerationStrategy,
        output_dir: str,
        filename: str,
        bundle,  # DatasetBundle
        *,
        prompt_label: str = "a",
    ):
        self._strategy = strategy
        self._output_dir = output_dir
        self._filename = filename
        self._bundle = bundle
        self._prompt_label = prompt_label

    def run(self, *, overwrite: bool = False, device: str = "cuda:0"):
        """Execute the inference loop.

        Args:
            overwrite: Delete existing predictions file before starting.
            device: CUDA device string.
        """
        # Resolve output path
        os.makedirs(self._output_dir, exist_ok=True)
        pred_file = os.path.join(self._output_dir, self._filename)

        # Overwrite
        if overwrite and os.path.exists(pred_file):
            os.remove(pred_file)
            print(f"[Overwrite] Removed existing {pred_file}")

        # Resume
        processed = load_checkpoint(pred_file)
        remaining = [
            i for i in self._bundle.image_ids if i not in processed
        ]
        print(
            f"[Resume] {len(processed)} done, {len(remaining)} remaining"
        )

        if not remaining:
            print("[Skip] All images already processed.")
            return

        # Prepare strategy (load model, examples, etc.)
        print(f"[Load] {self._strategy.label} on {device} ...")
        t0 = time.time()
        self._strategy.prepare(device=device)
        print(f"[Load] Done in {time.time() - t0:.1f}s")

        # Inference loop
        image_paths = self._bundle.image_paths
        try:
            with open(pred_file, "a") as f_out:
                for i, img_id in enumerate(remaining):
                    img_path = image_paths[img_id]
                    try:
                        caption = self._strategy.generate(img_path)
                    except Exception as e:
                        print(
                            f"  [ERROR] image_id={img_id}: {e}",
                            file=sys.stderr,
                        )
                        caption = ""

                    record = {
                        "image_id": img_id,
                        "file_name": os.path.basename(img_path),
                        "caption": caption,
                        "prompt": self._prompt_label,
                    }
                    f_out.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    if (i + 1) % 10 == 0:
                        f_out.flush()
                        print(
                            f"  [{i+1}/{len(remaining)}] {caption[:80]}...",
                            flush=True,
                        )
        finally:
            self._strategy.cleanup()

        print(f"[Done] → {pred_file}")
