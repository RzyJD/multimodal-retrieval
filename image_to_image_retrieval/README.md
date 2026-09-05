# Reference-image retrieval

Represent a retrieval target with a small set of reference images. A frozen encoder maps the references and gallery images into features; their mean reference feature becomes the query prototype. The historical experiments compare encoders, reference-set sizes, image–text fusion, and clustering for reference selection.

For your own **unlabeled gallery**, start with the [SigLIP inference entry point](inference/README.md). The notebooks below retain labeled evaluation and historical outputs.

## Notebook guide

| Notebook | Purpose | Status |
| --- | --- | --- |
| [reference_image_retrieval.ipynb](reference_image_retrieval.ipynb) | Encoder branches and 1/5/10/20-reference comparison | Main image-only experiment; default branch is SigLIP |
| [image_text_fusion.ipynb](image_text_fusion.ipynb) | Combine SigLIP image and class-text features | Main image–text fusion experiment |
| [cluster_reference_selection.ipynb](cluster_reference_selection.ipynb) | Select representatives from a larger reference pool | Exploratory; default target is Porcelain |

The public notebooks above were formerly named `图搜图.ipynb`, `图文对其.ipynb`, and `聚类对比实验.ipynb`.

Three local drafts are omitted from the public snapshot: `clip_encoder_exploration.ipynb` (historical encoder exploration), `reference_retrieval_debug.ipynb` (single-target scratchpad), and `cluster_text_fusion_exploration.ipynb` (incomplete combined exploration). The supported inference path is documented separately.

## Use your own dataset in the notebooks

`ImageFolder` is a directory-based loader, not a requirement to use ImageNet or its 1,000 classes. It assigns numeric labels by sorting immediate subfolder names. A labeled evaluation gallery is required for the notebooks' reference sampling and F1 calculations.

The original default layout is:

```text
datasets/
├── 0/    # Dog images
├── 1/    # Duck images
├── 2/    # Erhu images
├── 3/    # Piano images
├── 4/    # Porcelain images
└── 5/    # Other/background images
```

For a custom gallery, semantic folder names are also supported:

```text
my_gallery/
├── Background/
├── Car/
└── Cat/
```

Set the gallery location before starting Jupyter:

```bash
export RETRIEVAL_GALLERY_DIR="/path/to/my_gallery"
python -m jupyterlab
```

Edit **only the reference-image mapping** in [`project_paths.py`](../project_paths.py):

```python
REFERENCE_CLASS_FOLDERS = {
    "Cat": "Cat",  # Target/prompt name -> actual subfolder name
    "Car": "Car",
}
```

Every notebook resolves these names through `dataset.class_to_idx`, so extra background folders or a different sort order do not silently change the target labels. Folders not selected as the current target act as negatives during its evaluation. Do not modify the separate two-stage `TARGET_CLASSES` mapping for this module.

Start from the first cell and restart kernels after changing shared configuration. The setup cell exposes reference counts, encoder choice where applicable, and the single target/cluster budget for exploratory notebooks. The default single target is the last configured target in the clustering notebook. Set `TARGET_CLASS` explicitly to choose another configured target.

Supply enough images for the selected reference count and leave target images for evaluation. Clustering additionally needs enough reference samples for `ELBOW_MAX_K` and `N_CLUSTERS`, with `REFERENCE_BUDGET` no larger than the reference pool. These remain manual experimental choices; the notebooks do not select them automatically.

`RETRIEVAL_GALLERY_DIR` defaults to this module's `datasets/` directory. Relative environment paths resolve against the repository root. No private dataset is included.

## Models and execution

Use the [CLIP/SigLIP notebook environment](../docs/SETUP.md), or the smaller [inference environment](inference/README.md#installation) for the CLI alone. `RETRIEVAL_SIGLIP_MODEL` can identify the default SigLIP model or a compatible local checkpoint directory. The notebook setup cells expose the CLIP model choices. Executing model-loading cells can download weights; importing the shared configuration does not.

The inference entry point loads only SigLIP once. Some historical notebooks still contain exploratory or redundant model-loading cells; the CLI does not depend on those cells or their execution state.

## Scoring and historical limits

- Reference sampling in the notebooks uses seed `0`; the inference entry point uses all explicitly supplied references.
- Notebook references are removed from the gallery by index. The CLI excludes the same reference paths (including symlink aliases); neither performs content-based deduplication of copied files.
- The SigLIP branch normalizes each image feature, averages reference features, and takes a dot product with each normalized gallery feature. It does **not** renormalize the mean prototype.
- Image–text fusion uses `(mean_reference_feature + raw_text_feature) / 2`, followed by the same dot product. Text features and the fused prototype are not normalized again. These scores are not probabilities, and fusion changes their scale.
- Historical notebook thresholds maximize F1 on the labeled gallery. The CLI uses an explicit threshold or Top-K, with no label-based threshold search or reported F1.
- Clustering excludes a larger reference pool, so its gallery differs from the main experiment. It remains a notebook exploration and is not included in the inference entry point.

The report retains the original project document's results. Saved numerical outputs remain historical snapshots, not newly executed benchmark results. Configuration and filename changes preserve the original default categories, sampling settings, and scoring formulas.
