# Two-stage text-to-image retrieval

This pipeline uses complementary English and Chinese image–text scores to retrieve candidates, then asks LLaVA a binary visual question about each candidate.

For an unlabeled image directory and a custom query, use the new [command-line inference entry point](inference/README.md). The notebooks below retain the historical labeled evaluation workflow.

## Reading and execution order

Run each notebook from its first cell in a fresh kernel with the appropriate environment:

1. [English CLIP](image_text_matching/english_clip_retrieval.ipynb): ViT-B/32, target versus Others, per-target threshold search and prediction export.
2. [Taiyi-CLIP](image_text_matching/taiyi_clip_retrieval.ipynb): Chinese text features paired with CLIP ViT-L/14 image features, threshold search and prediction export.
3. [Candidate union](image_text_matching/merge_clip_candidates.ipynb): combine positive predictions from either model and write candidate manifests.
4. [LLaVA verification](llava_verification/llava_candidate_verification.ipynb): load candidate manifests, ask Yes/No questions, save answers and calculate retained-set metrics.

The first two notebooks can be executed in either order. Their outputs must come from the same unchanged gallery. Run the union only after both have completed. Run the fourth notebook with the upstream LLaVA environment described in [setup](../docs/SETUP.md).

## Inputs and outputs

Notebook setup uses [project_paths.py](../project_paths.py); private images are under `RETRIEVAL_PICTURE_DIR`, and generated pickle files use `RETRIEVAL_OUTPUT_DIR`. These files and model weights are not published. The custom-data contract is described below.

The CLIP prediction format still uses list positions. Keep enumeration and data unchanged while generating the two CLIP outputs and their union; a new ordering must not be combined with old predictions. Newly generated candidate manifests store paths relative to the image root, and LLaVA answers retain those same path identifiers for evaluation.

## Use your own dataset

### 1. Configure locations before starting Jupyter

From the repository root, in a Bash/Zsh shell:

```bash
export RETRIEVAL_PICTURE_DIR="/path/to/my_dataset"
export RETRIEVAL_OUTPUT_DIR="/path/to/my_results"
# Optional: use a local checkpoint instead of the default Hugging Face model ID.
export RETRIEVAL_LLAVA_MODEL="/path/to/llava-v1.5-7b"
python -m jupyterlab
```

Replace the example paths with your own. In PowerShell, use `$env:RETRIEVAL_PICTURE_DIR = "C:/path/to/my_dataset"` and the same syntax for the other variables. Absolute dataset/output paths are used directly; relative ones are resolved from the repository root, independently of the notebook's working directory.

All four notebooks must receive the same configuration, including when LLaVA runs in a separate environment. These values are read when `project_paths.py` is imported. After changing environment variables, restart Jupyter with those variables; after editing the category mapping, restart every notebook kernel and run from its first cell.

### 2. Configure categories in one place

Edit only `TARGET_CLASSES` in [project_paths.py](../project_paths.py), for example:

```python
TARGET_CLASSES = {
    "Cat": "猫",
    "Car": "汽车",
}
```

Keys are English category names used for folder lookup, English CLIP queries and LLaVA questions. Values provide Chinese category names for Taiyi prompts. Use non-empty folder names without path separators. The original five categories remain the default; this setting applies only to the two-stage notebooks, not the independent adaptation/reference-image experiments.

Arrange your images using the existing nested layout:

```text
my_dataset/
├── Cat/
│   ├── Cat/       # Cat positives
│   └── Fox/       # Confusable negatives
└── Car/
    ├── Car/       # Car positives
    └── Truck/     # Confusable negatives
```

For target `Cat`, positives are read from `my_dataset/Cat/Cat/`. **All other second-level folders** are treated as negatives, including other targets such as `Car/Car/`. Use directly contained, readable image files, with at least one positive and one negative example per target. Keep these image folders free of non-image files because the historical enumeration code has not been redesigned here.

Then run the four notebooks in the order above, using a fresh output directory for a different dataset. The union notebook checks that the loaded prediction dictionaries match the configured categories. Category checks do not detect changed data under the same category names, so regenerate both CLIP outputs whenever the dataset changes.

These notebooks search thresholds and evaluate against labels derived from the folder layout. For an arbitrary unlabeled folder, use the [inference command](inference/README.md) with supplied thresholds; changing the notebook root path alone does not add that mode.

### 3. Move data without rewriting new candidate manifests

The union notebook writes `{class} llava dataset` files containing parallel `Path`, `True` and `Pre` lists. New `Path` values look like `Cat/Cat/example.jpg`, rather than a machine-specific absolute path. The verifier opens that file under the current `RETRIEVAL_PICTURE_DIR` and keeps `Cat/Cat/example.jpg` as the answer's identifier, so its evaluation lookup still matches.

Once candidate manifests have been generated, the dataset and output directory may be moved to another machine while preserving the relative image layout. Configure their new roots before running LLaVA. Existing absolute-path manifests are still readable where their original paths exist, but are not automatically migrated: regenerate them before moving data. Do not mix old absolute-path manifests with new relative-path answer files, or mix answers from different runs.

Only paths and category configuration have changed. Historical notebook outputs, threshold formulas, prompts and reported results have not been recalculated.

## Verification behavior

The prompt is `Is this an image of a {class}? Please answer with Yes or No.` The historical implementation removes candidates when the generated answer contains the case-sensitive string `No`. Other answers remain accepted; failed inference is logged and the original candidate prediction remains in the evaluation. Temperature is `0.5`, with no recorded generation seed.

These choices are documented as existing behavior. Strict answer parsing, failure handling and deterministic generation would be separate algorithm changes and have not been used to regenerate the historical results.

## Results provenance

Saved outputs are historical snapshots. The presentation figures follow the original project document. The publication cleanup changes setup paths and removes private media, machine paths and progress noise, without recalculating metrics.
