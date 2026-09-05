"""Portable notebook paths. Importing this module does not create directories."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Two-stage retrieval: English folder/prompt name -> Chinese prompt name.
# Edit this mapping once to use your own categories, then restart all kernels.
# Keep this order for the original project categories.
TARGET_CLASSES = {
    "Dog": "狗",
    "Piano": "钢琴",
    "Erhu": "二胡",
    "Porcelain": "瓷器",
    "Duck": "鸭子",
}

# Reference-image experiments: target/prompt name -> ImageFolder subdirectory.
# Other gallery folders are background. Independent of two-stage class ordering.
REFERENCE_CLASS_FOLDERS = {
    "Dog": "0",
    "Duck": "1",
    "Erhu": "2",
    "Piano": "3",
    "Porcelain": "4",
}


def reference_class_indices(class_to_idx, class_folders=None):
    """Resolve semantic targets against ImageFolder's actual directory labels."""
    folders = REFERENCE_CLASS_FOLDERS if class_folders is None else class_folders
    if not folders:
        raise ValueError("Configure at least one reference target in REFERENCE_CLASS_FOLDERS.")
    missing = sorted(set(folders.values()) - set(class_to_idx))
    if missing:
        raise ValueError(f"Reference target folders are missing from the gallery: {missing}")
    return {name: class_to_idx[folder] for name, folder in folders.items()}


def configured_path(variable, default):
    """Resolve a configured path relative to the repository, not notebook CWD."""
    value = Path(os.environ.get(variable, str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


PICTURE_DIR = configured_path(
    "RETRIEVAL_PICTURE_DIR", "two_stage_retrieval/image_text_matching/Picture"
)
GALLERY_DIR = configured_path("RETRIEVAL_GALLERY_DIR", "image_to_image_retrieval/datasets")
SIGLIP_MODEL = os.environ.get("RETRIEVAL_SIGLIP_MODEL", "google/siglip-base-patch16-224")
OUTPUT_DIR = configured_path("RETRIEVAL_OUTPUT_DIR", ".local/two_stage")
# May be a Hugging Face model ID or an existing local checkpoint directory.
LLAVA_MODEL = os.environ.get("RETRIEVAL_LLAVA_MODEL", "liuhaotian/llava-v1.5-7b")
