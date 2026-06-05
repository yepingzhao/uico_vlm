"""Reference-free evaluation: CLIPScore and RefCLIPScore.

CLIPScore: cosine similarity between image and generated caption CLIP embeddings.
RefCLIPScore: CLIPScore anchored by reference captions (uses references for
calibration but doesn't directly compare n-grams).
"""

from typing import Dict, List
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


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
