"""Reference-based evaluation metrics using the coco-caption library.

Computes BLEU-1/4, METEOR, ROUGE-L, CIDEr-D, and SPICE,
plus the composite S_m score defined in the paper Eq.(1).
"""

import json
import os
import sys
import tempfile
from typing import Dict, List


#: Maximum word count for prediction captions in ref-based metrics.
#: Captions exceeding this are truncated before evaluation.  Reference captions
#: in this dataset are ≤34 words (median 9), so words beyond ~50 cannot match
#: any reference n-gram.  Truncation also prevents OOM in the Stanford parser
#: used by SPICE when models generate very long captions (e.g. Qwen2.5-VL).
MAX_CAPTION_WORDS = 50


def _build_coco_result(predictions: Dict[int, str], max_words: int = MAX_CAPTION_WORDS) -> List[dict]:
    """Convert {image_id: caption} to COCO result format.

    Captions longer than *max_words* are truncated to keep SPICE's Stanford
    parser within memory limits.
    """
    result = []
    for img_id, caption in predictions.items():
        words = caption.split()
        if len(words) > max_words:
            caption = " ".join(words[:max_words])
        result.append({"image_id": img_id, "caption": caption})
    return result


def compute_ref_based_metrics(
    predictions: Dict[int, str],
    references: Dict[int, List[str]],
    cache_dir: str = None,
) -> Dict[str, float]:
    """Compute all reference-based metrics using pycocoevalcap.

    Args:
        predictions: {image_id: generated_caption}
        references: {image_id: [ref1, ref2, ...]}
        cache_dir: Optional directory for caching intermediate files.

    Returns:
        Dict with keys: BLEU-1, BLEU-4, METEOR, ROUGE_L, CIDEr, SPICE, S_m
    """
    # Write predictions and references to temporary COCO-format JSON
    tmpdir = tempfile.mkdtemp(prefix="uico_vlm_eval_")
    try:
        pred_file = os.path.join(tmpdir, "predictions.json")
        ref_file = os.path.join(tmpdir, "references.json")

        # Predictions as list of {image_id, caption}
        pred_list = _build_coco_result(predictions)
        with open(pred_file, "w") as f:
            json.dump(pred_list, f)

        # References as {image_id: [captions]}
        with open(ref_file, "w") as f:
            json.dump(references, f)

        # Use pycocoevalcap's COCOEvalCap
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap

        # Create a minimal COCO ground-truth structure
        anns = []
        img_ids_set = set()
        ann_id = 0
        for img_id, refs in references.items():
            img_ids_set.add(img_id)
            for ref in refs:
                anns.append({
                    "image_id": img_id,
                    "id": ann_id,
                    "caption": ref,
                })
                ann_id += 1

        imgs = [{"id": img_id} for img_id in sorted(img_ids_set)]

        gt_file = os.path.join(tmpdir, "gt.json")
        with open(gt_file, "w") as f:
            json.dump({"images": imgs, "annotations": anns, "info": {}, "licenses": []}, f)

        coco = COCO(gt_file)
        coco_res = coco.loadRes(pred_file)

        # Run evaluation
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params["image_id"] = coco_res.getImgIds()
        try:
            coco_eval.evaluate()
        except Exception as e:
            print(f"  [WARN] Ref-based eval partially failed: {e}", file=sys.stderr)
            # Continue with whatever metrics were computed before the crash

        # Collect results
        metrics = {}
        for metric, score in coco_eval.eval.items():
            key = metric.upper().replace(" ", "_")
            metrics[key] = score * 100

        # Compute S_m (composite) per paper Eq.(1)
        sm_keys = ["BLEU_4", "METEOR", "ROUGE_L", "CIDER", "SPICE"]
        metrics["S_m"] = sum(metrics.get(k, 0.0) for k in sm_keys) / len(sm_keys)

        return metrics

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
