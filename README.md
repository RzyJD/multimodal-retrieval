# Multimodal Retrieval from Unlabeled Image Collections

A research code portfolio exploring how to retrieve images using text queries, a small labeled cache, or reference images. The target categories in the historical project are **Dog, Duck, Erhu, Piano, and Porcelain**, with visually similar distractors.

The project compares three complementary approaches:

- **Two-stage text-to-image retrieval:** combine English CLIP and Taiyi-CLIP candidates, then use LLaVA visual question answering to verify them.
- **Task adaptation:** adapt Tip-Adapter and Tip-Adapter-F to target-versus-background retrieval, with class-specific prompts and F1 evaluation.
- **Reference-image retrieval:** construct an image prototype and explore encoder choice, reference-set size, image–text fusion, and clustering.

## Framework

[![Two-stage text-to-image retrieval and reference-image retrieval](docs/assets/retrieval_framework.png)](docs/assets/retrieval_framework.png)

*Framework diagram from the project report's figure assets. (a) English CLIP and Taiyi-CLIP generate a candidate union, followed by LLaVA verification. (b) SigLIP encodes reference images and gallery images for prototype-based retrieval. Predictions, similarity bars, and retrieved examples in this diagram illustrate the workflow; they are not measured model outputs. Click the image to view it at full resolution.*

Tip-Adapter is a separate adaptation experiment: a labeled feature cache augments the frozen CLIP score, and Tip-Adapter-F learns the cache keys. It is evaluated without the LLaVA verifier.

## Project-specific work and upstream components

The project-specific implementation includes the bilingual candidate-union pipeline, the LLaVA verification loop, target-versus-background dataset preparation, class-specific prompt templates and F1 evaluation for Tip-Adapter, and reference-prototype comparison notebooks. These are the main places to inspect the project's research and engineering decisions.

CLIP, Taiyi-CLIP, SigLIP, LLaVA and the Tip-Adapter method originate from their respective authors. The adaptation code is derived from Tip-Adapter; the bundled `tip_adapter_ft/clip/` code derives from OpenAI CLIP. This project does not claim authorship of those models or methods. See [third-party sources and permissions](THIRD_PARTY_NOTICES.md).

## Start reading here

| Module | Entry point | What to inspect |
| --- | --- | --- |
| Research narrative | [Project report](Multimodal_Retrieval_Research_Project_Report.md) | Motivation, method, historical results and limitations |
| Candidate generation and verification | [Two-stage guide](two_stage_retrieval/README.md) | CLIP scoring, union, VQA and retained candidates |
| Cache adaptation | [Tip-Adapter guide](tip_adapter_ft/README.md) | Prompt construction, cache fusion and key optimization |
| Reference-image queries | [Image retrieval guide](image_to_image_retrieval/README.md) | Prototype formation, feature fusion and reference selection |

## Historical results

The figures below are transcribed from the original project document, **《项目文档.pdf》**. They are retained as historical project records and were not re-estimated or independently validated during repository preparation. The source PDF and datasets are not distributed here.

| Experiment | Reported macro F1 |
| --- | ---: |
| English CLIP | 0.8162 |
| Taiyi-CLIP | 0.8751 |
| Bilingual union | 0.8313 |
| Bilingual union + LLaVA | 0.8853 |
| Tip-Adapter-F, 500 shots | 0.9248 |
| SigLIP, 20 reference images | 0.7960 |
| Image–text feature fusion | 0.86062539 |

Saved notebook outputs are historical snapshots and may belong to different runs from the summary. The report remains the presentation record; the snapshots are not a newly validated benchmark. The historical evaluation tunes thresholds on the labeled evaluation gallery; the Tip-Adapter implementation also uses evaluation features for hyperparameter and epoch selection.

## Running the code

For **unlabeled images and a custom query**, use the [two-stage inference CLI](two_stage_retrieval/inference/README.md). It takes explicit English/Chinese queries and thresholds, then returns image-level decisions in JSON. The notebooks remain available for reading the historical experiments.

For **reference-image queries**, use the [SigLIP inference CLI](image_to_image_retrieval/inference/README.md). Supply reference images and an unlabeled gallery to obtain ranked results using Top-K or an explicit threshold, with optional image–text fusion.

Reading the repository does not require installing models or downloading data. To execute it, prepare your own images and follow [setup and model acquisition](docs/SETUP.md) and the module guides above.

```bash
# Run from the repository root, inside your chosen Python environment.
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open notebooks from the repository or one of its subdirectories. Their first setup cell resolves the repository root automatically. Local paths can be overridden using the environment variables in [path configuration](docs/SETUP.md#path-configuration).

The executable code includes research notebooks and lightweight inference CLIs, not a deployed application. Full training/inference with pretrained models was not rerun for this publication cleanup. Model downloads require internet access; the adaptation and LLaVA modules require a compatible CUDA environment.

## Repository contents

```text
retrieval/
├── README.md
├── Multimodal_Retrieval_Research_Project_Report.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── project_paths.py
├── docs/                         # Setup and the final homepage framework figure
├── two_stage_retrieval/           # English/Chinese scoring and LLaVA verification
├── tip_adapter_ft/                # Tip-Adapter adaptation code and configuration
└── image_to_image_retrieval/      # Reference-prototype notebooks and SigLIP inference
```

Datasets, weights, caches, local figure drafts and machine-specific files are excluded by `.gitignore`. The homepage includes the report’s framework figure with illustrative image thumbnails; the source datasets are not included. No dataset or model redistribution rights are granted by this repository; see [third-party notices](THIRD_PARTY_NOTICES.md).
