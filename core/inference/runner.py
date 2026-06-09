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

    def run(self, *, overwrite: bool = False, device: str = "cuda:0",
            partition: tuple | None = None, chunk: tuple | None = None):
        """Execute the inference loop.

        Args:
            overwrite: Delete existing predictions file before starting.
            device: CUDA device string.
            partition: Optional (k, n) tuple for strided inference (every n-th).
            chunk: Optional (k, n) tuple for contiguous chunk inference.
                Splits the dataset into n equal blocks; worker k gets block k.
                Writes to {filename}.chunk{k} for later merge.
        """
        # Resolve output path
        os.makedirs(self._output_dir, exist_ok=True)
        pred_file = os.path.join(self._output_dir, self._filename)

        # Partition/chunk mode: write to separate file, resume from same file
        mode_tag = None
        if partition is not None:
            k, n = partition
            mode_tag = f"part{k}"
            if not (0 <= k < n):
                raise ValueError(f"partition k={k} out of range [0, {n})")
        elif chunk is not None:
            k, n = chunk
            mode_tag = f"chunk{k}"
            if not (0 <= k < n):
                raise ValueError(f"chunk k={k} out of range [0, {n})")

        if mode_tag:
            part_file = os.path.join(self._output_dir,
                                     f"{self._filename}.{mode_tag}")
            if overwrite and os.path.exists(pred_file):
                os.remove(pred_file)
        else:
            part_file = pred_file

        # Overwrite (partition-aware)
        if overwrite and os.path.exists(part_file):
            os.remove(part_file)
            print(f"[Overwrite] Removed existing {part_file}")

        # Resume — also read main file in partition/chunk mode so workers
        # skip images already processed by the main process or other workers.
        processed = load_checkpoint(part_file)
        if mode_tag and os.path.exists(pred_file):
            processed |= load_checkpoint(pred_file)
        all_ids = self._bundle.image_ids
        if partition is not None:
            k, n = partition
            all_ids = [iid for idx, iid in enumerate(all_ids)
                       if idx % n == k]
        elif chunk is not None:
            k, n = chunk
            total = len(all_ids)
            chunk_size = total // n
            start = k * chunk_size
            end = start + chunk_size if k < n - 1 else total
            all_ids = all_ids[start:end]
        remaining = [i for i in all_ids if i not in processed]
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
            with open(part_file, "a") as f_out:
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

        print(f"[Done] → {part_file}")
        if mode_tag and (partition or chunk) and (
            (partition and partition[0] == 0) or (chunk and chunk[0] == 0)
        ):
            glob_pat = f"{self._filename}.{mode_tag.split('0')[0]}*"
            print(f"[Merge] After all pieces finish: "
                  f"cat {self._output_dir}/{glob_pat} "
                  f">> {self._output_dir}/{self._filename}")
