"""Reference-based and reference-free metric computation.

Does NOT handle file I/O or orchestration — only "how to calculate".
"""

import json
import os
import sys
import tempfile
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import CLIP_MODEL_NAME

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum word count for prediction captions in ref-based metrics.
#: Captions exceeding this are truncated before evaluation.  Reference captions
#: in this dataset are ≤34 words (median 9), so words beyond ~50 cannot match
#: any reference n-gram.  Truncation also prevents OOM in the Stanford parser
#: used by SPICE when models generate very long captions (e.g. Qwen2.5-VL).
MAX_CAPTION_WORDS = 50


# ---------------------------------------------------------------------------
# Reference-based metrics
# ---------------------------------------------------------------------------

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
        Dict with keys: BLEU-1, BLEU-4, METEOR, ROUGE_L, CIDEr, SPICE, S_m, S*_m
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

        # Collect results
        metrics = {}
        for metric, score in coco_eval.eval.items():
            key = metric.upper().replace(" ", "_")
            metrics[key] = score * 100

        # Compute S_m (composite) per paper Eq.(1)
        sm_keys = ["BLEU_4", "METEOR", "ROUGE_L", "CIDER", "SPICE"]
        metrics["S_m"] = sum(metrics.get(k, 0.0) for k in sm_keys) / len(sm_keys)

        # Compute S*_m (SPICE-excluded composite) per zhang2021global
        sm_star_keys = ["BLEU_4", "METEOR", "ROUGE_L", "CIDER"]
        metrics["S*_m"] = sum(metrics.get(k, 0.0) for k in sm_star_keys) / len(sm_star_keys)

        return metrics

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reference-free metrics (CLIPScore)
# ---------------------------------------------------------------------------

class CLIPScorer:
    """Compute CLIPScore and RefCLIPScore for image-caption pairs."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda:0"):
        self.device = device
        self._model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self._model.eval()

    @torch.no_grad()
    def compute_refclipscore(
        self,
        image_paths: Dict[int, str],
        captions: Dict[int, str],
        references: Dict[int, List[str]],
    ) -> Dict[str, float]:
        """Compute RefCLIPScore: harmonic mean of CLIPScore and reference-anchored score.

        For each image, we compute CLIPScore(image, generated) and
        the max CLIPScore among references for that image.
        RefCLIPScore = 2 * CLIPScore * RefMax / (CLIPScore + RefMax).
        """
        if not image_paths:
            return {"RefCLIPScore": 0.0, "CLIPScore": 0.0, "RefMax": 0.0}

        clip_scores = []
        ref_max_scores = []
        refclip_scores = []
        img_ids = sorted(set(image_paths.keys()) & set(captions.keys()))

        for img_id in img_ids:
            img_path = image_paths[img_id]
            caption = captions[img_id]
            refs = references.get(img_id, [])

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # CLIPScore for generated caption
            inputs = self._processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self._model(**inputs)
            gen_sim = torch.nn.functional.cosine_similarity(
                outputs.image_embeds[0], outputs.text_embeds[0], dim=0
            ).item()
            clip_scores.append(gen_sim)

            # Max CLIPScore among references
            ref_max = 0.0
            if refs:
                ref_inputs = self._processor(
                    text=refs,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                ref_outputs = self._model(**ref_inputs)
                for i in range(len(refs)):
                    ref_sim = torch.nn.functional.cosine_similarity(
                        ref_outputs.image_embeds[0],
                        ref_outputs.text_embeds[i],
                        dim=0,
                    ).item()
                    ref_max = max(ref_max, ref_sim)
            ref_max_scores.append(ref_max)

            # Harmonic mean
            if gen_sim + ref_max > 0:
                rc = 2 * gen_sim * ref_max / (gen_sim + ref_max)
            else:
                rc = 0.0
            refclip_scores.append(rc)

        n = len(clip_scores)
        return {
            "CLIPScore": sum(clip_scores) / n if n else 0.0,
            "RefMax": sum(ref_max_scores) / n if n else 0.0,
            "RefCLIPScore": sum(refclip_scores) / n if n else 0.0,
        }
