"""Explicit inference parameters; no labels or threshold fitting are required."""

from dataclasses import dataclass
import math
from pathlib import Path

ENGLISH_MODEL = "ViT-B/32"
TAIYI_TEXT_MODEL = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"
TAIYI_IMAGE_MODEL = "openai/clip-vit-large-patch14"


@dataclass(frozen=True)
class InferenceConfig:
    image_dir: Path
    query_en: str
    query_zh: str
    threshold_en: float
    threshold_zh: float
    batch_size: int = 16
    device: str = "cuda"
    llava_model: str = "liuhaotian/llava-v1.5-7b"
    skip_verification: bool = False

    def validate(self):
        if not self.image_dir.is_dir():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not self.query_en.strip() or not self.query_zh.strip():
            raise ValueError("Both English and Chinese queries must be non-empty.")
        for value in (self.threshold_en, self.threshold_zh):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Thresholds must be finite numbers between 0 and 1.")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive.")
        if self.device != "cpu" and self.device != "cuda" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("Device must be cpu, cuda, or cuda:N.")
        if not self.skip_verification and self.device == "cpu":
            raise ValueError("LLaVA verification requires CUDA; use --skip-verification for CLIP-only CPU inference.")
        if not self.llava_model.strip():
            raise ValueError("LLaVA model must be a model ID or local checkpoint directory.")
