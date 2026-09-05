# Setup and model acquisition

## Environment status

This repository is a readable research implementation. It does not ship a tested environment lockfile, and full model execution was not rerun during publication preparation. `requirements.txt` lists the packages imported by the notebooks; it is not a guarantee of compatibility with every future release.

Create an isolated Python environment and install the root requirements for CLIP/SigLIP notebooks. Install PyTorch and torchvision together for the intended device. CLIP notebooks select CUDA when available; some historical half-precision paths have not been validated on CPU. Tip-Adapter and LLaVA contain explicit CUDA calls and should be treated as CUDA-only entry points.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

For Tip-Adapter alone, use `python -m pip install -r tip_adapter_ft/requirements.txt` in a suitable environment. The adaptation script imports its bundled CLIP implementation.

## Model inventory

| Module | Model identifier | Acquisition |
| --- | --- | --- |
| English candidate scoring | OpenAI CLIP `ViT-B/32` | `clip.load` downloads the checkpoint into its cache |
| Chinese text scoring | `IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese` | Hugging Face `from_pretrained` |
| Image encoder paired with Taiyi text | `openai/clip-vit-large-patch14` | Hugging Face `from_pretrained` |
| Reference-image comparison | CLIP `ViT-B/32`, `openai/clip-vit-large-patch14`, `google/siglip-base-patch16-224` | Corresponding `clip.load` / `from_pretrained` call |
| Current adaptation configuration | CLIP `RN50` | Bundled `clip.load`; change `backbone` in the YAML explicitly |
| Candidate verification | `liuhaotian/llava-v1.5-7b` | Upstream LLaVA loader; override with `RETRIEVAL_LLAVA_MODEL` for a local checkpoint |

Links: [OpenAI CLIP](https://github.com/openai/CLIP), [Taiyi-CLIP](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese), [SigLIP](https://huggingface.co/google/siglip-base-patch16-224), [LLaVA model](https://huggingface.co/liuhaotian/llava-v1.5-7b).

Historical model revision hashes and GPU memory measurements were not recorded in this repository. No model weights are included, and no fine-tuned checkpoint is advertised as a validated downloadable artifact.

## LLaVA environment

Use a **separate environment** following the [upstream LLaVA installation instructions](https://github.com/haotian-liu/LLaVA). The verification notebook imports `llava.constants`, `llava.conversation`, `llava.mm_utils` and `llava.model.builder`; installing a generic unrelated package named `llava` is not a replacement for that source tree.

The original project's LLaVA dependency manifest recorded version `1.2.2.post1`, including `torch==2.1.2`, `torchvision==0.16.2`, `transformers==4.37.2`, `tokenizers==0.15.1`, `sentencepiece==0.1.99`, and `accelerate==0.21.0`. These are provenance information, not a newly tested installation recipe. The exact upstream commit was not preserved in the public setup instructions.

Install a Jupyter kernel for that environment and select it for `llava_candidate_verification.ipynb`. The checked-in notebook keeps the historical non-8-bit loading setting and temperature `0.5`; no new determinism or memory guarantee is implied.

## Path configuration

Run Jupyter from this repository or a directory inside it. Notebook setup cells find `project_paths.py` by walking upward from the working directory. Defaults match the existing local project layout, with generated results under `.local/`.

Optional environment variables:

- `RETRIEVAL_PICTURE_DIR`: nested gallery for bilingual scoring.
- `RETRIEVAL_GALLERY_DIR`: labeled ImageFolder gallery for reference-image experiments, with numbered or named subfolders.
- `RETRIEVAL_SIGLIP_MODEL`: SigLIP model ID or compatible local checkpoint directory for reference-image notebooks and the inference CLI default.
- `RETRIEVAL_OUTPUT_DIR`: trusted local generated predictions and VQA answers.
- `RETRIEVAL_LLAVA_MODEL`: Hugging Face model ID or local checkpoint directory.

For the two-stage pipeline, edit `TARGET_CLASSES` in `project_paths.py` to set English folder/query names and their Chinese translations. Follow the [custom dataset instructions](../two_stage_retrieval/README.md#use-your-own-dataset) for the required directory layout and configuration examples. New candidate manifests use paths relative to `RETRIEVAL_PICTURE_DIR`, so they can be used after moving the gallery without rewriting each image path. Previously generated absolute-path manifests must be regenerated for that portability.

For reference-image experiments, edit the independent `REFERENCE_CLASS_FOLDERS` mapping in `project_paths.py`; labels are resolved from `ImageFolder.class_to_idx`. See the [reference-image data instructions](../image_to_image_retrieval/README.md#use-your-own-dataset-in-the-notebooks). The [SigLIP inference CLI](../image_to_image_retrieval/inference/README.md) instead takes explicit reference/gallery directories and requires no class mapping or gallery labels. Its smaller inference-only requirements can be installed separately from the notebook environment.

Environment variables are read on import. Set them before launching Jupyter, and restart kernels after editing the shared category mapping. The original default category order and saved numerical outputs remain unchanged.

Neither importing `project_paths.py` nor reading the notebooks downloads models. Executing their model-loading cells can download weights.
