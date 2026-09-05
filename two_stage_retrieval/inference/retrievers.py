"""CLIP scoring extracted from the English and Taiyi experiment notebooks."""

import gc

from .config import ENGLISH_MODEL, TAIYI_IMAGE_MODEL, TAIYI_TEXT_MODEL


def checked_device(name):
    import torch

    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a CUDA environment, or --device cpu --skip-verification.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {name}")
    return device


def positive_probabilities(image_features, text_features, logit_scale):
    """Two-prompt softmax: normalized target similarity versus Others similarity."""
    import torch.nn.functional as F

    images = F.normalize(image_features.float(), dim=-1)
    texts = F.normalize(text_features.float(), dim=-1)
    logits = logit_scale.float().exp() * images @ texts.t()
    return logits.softmax(dim=-1)[:, 0].detach().cpu().tolist()


class CLIPRetriever:
    """Load one encoder pair at a time and cache its two query text features."""

    def __init__(self, language, config):
        import torch

        self.language = language
        self.device = checked_device(config.device)
        if language == "en":
            import clip

            self.model, self.processor = clip.load(ENGLISH_MODEL, device=self.device)
            self.model.eval()
            tokens = clip.tokenize([config.query_en, "others"]).to(self.device)
            with torch.inference_mode():
                self.text_features = self.model.encode_text(tokens)
        elif language == "zh":
            from transformers import BertForSequenceClassification, BertTokenizer, CLIPModel, CLIPProcessor

            tokenizer = BertTokenizer.from_pretrained(TAIYI_TEXT_MODEL)
            text_encoder = BertForSequenceClassification.from_pretrained(TAIYI_TEXT_MODEL).to(self.device).eval()
            tokens = tokenizer(
                [f"这是一张{config.query_zh}的图片", "这是一张其他的图片"],
                return_tensors="pt", padding=True,
            ).to(self.device)
            with torch.inference_mode():
                # Pass the padding mask as well as input IDs for custom queries.
                self.text_features = text_encoder(**tokens).logits
            del text_encoder
            self.model = CLIPModel.from_pretrained(TAIYI_IMAGE_MODEL).to(self.device).eval()
            if self.device.type == "cuda":
                self.model.half()
            self.processor = CLIPProcessor.from_pretrained(TAIYI_IMAGE_MODEL)
        else:
            raise ValueError(f"Unknown scoring language: {language}")

    def score(self, images):
        import torch

        with torch.inference_mode():
            if self.language == "en":
                inputs = torch.stack([self.processor(image) for image in images]).to(self.device)
                features = self.model.encode_image(inputs)
            else:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)
                features = self.model.get_image_features(**inputs)
            return positive_probabilities(features, self.text_features, self.model.logit_scale)

    def close(self):
        import torch

        self.model = None
        self.text_features = None
        self.processor = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
