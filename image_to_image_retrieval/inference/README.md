# SigLIP reference-image inference

Retrieve images from an unlabeled directory using one or more reference images. This entry point reuses the main notebooks' prototype and optional image–text fusion formulas, without labeled evaluation, random reference sampling, or clustering.

## Installation

Run from the repository root in an isolated Python environment:

```bash
python -m pip install -r image_to_image_retrieval/inference/requirements.txt
python -m image_to_image_retrieval.inference.run --help
```

Install a PyTorch build appropriate for your CPU or CUDA device. The inference-only requirements exclude CLIP, LLaVA, torchvision, and Jupyter as direct dependencies. They constrain Transformers to its 4.x API; they are not a validated environment lockfile. The existing notebook environment may also be used if compatible.

Default weights and preprocessing come from `google/siglip-base-patch16-224`. On first use, Hugging Face loads/downloads them through `from_pretrained`. Set `--model /path/to/local/siglip` to use a compatible local model directory containing its configuration, weights, image processor, and tokenizer. `RETRIEVAL_SIGLIP_MODEL` supplies the default when `--model` is omitted. Other encoder families are not supported by this entry point.

## Input layout

Prepare references for **one target per run**, separate from the gallery where possible:

```text
my_images/
├── references/
│   ├── example_1.jpg
│   └── example_2.jpg
└── gallery/
    ├── image_001.jpg
    └── any_subfolder/
        └── image_002.png
```

Both directories are scanned recursively. Class labels and numbered folders are unnecessary. All reference images are used; no training is required. Supported suffixes are JPG/JPEG, PNG, WEBP, BMP, GIF, TIF/TIFF (case-insensitive). Hidden files/directories are skipped; animated/multipage inputs use their first frame. Images are converted to RGB before model preprocessing.

A gallery entry that resolves to a reference file is excluded, including symlink aliases. Copied files with different paths are not deduplicated by content; keep reference copies out of the gallery if needed.

## Retrieve Top-K

From the repository root:

```bash
python -m image_to_image_retrieval.inference.run \
  --reference-dir /path/to/my_images/references \
  --image-dir /path/to/my_images/gallery \
  --top-k 20 \
  --device auto \
  --output .local/image_to_image/top20.json
```

`auto` selects CUDA if available, otherwise CPU. Explicit `cpu`, `cuda`, and `cuda:N` are also accepted. `--batch-size` defaults to 16; lower it if device memory is insufficient. CPU execution may be slower. Top-K returns up to K valid gallery images after reference exclusions.

## Optional image–text fusion

Supply the complete text prompt; the entry point does not add a template:

```bash
python -m image_to_image_retrieval.inference.run \
  --reference-dir /path/to/porcelain_references \
  --image-dir /path/to/gallery \
  --query-text "a photo of a porcelain vase" \
  --top-k 20 \
  --output .local/image_to_image/fusion.json
```

Without `--query-text`, retrieval uses reference images alone. Use a short description suited to the selected model. Overlong prompts produce an error instead of silent truncation.

## Threshold selection and score meaning

Exactly one of `--top-k` and `--threshold` is required. To select by threshold, replace `--top-k 20` with `--threshold VALUE`, where `VALUE` is a finite number you choose for your own target/reference set. There is no default calibrated threshold and no automatic search using gallery labels.

For raw reference image features `f_i` and a raw gallery image feature `g`:

```text
r_i = f_i / ||f_i||
p   = mean(r_i)
v   = g / ||g||
image-only score = v · p

with optional raw text feature t:
p_fused = (p + t) / 2
fusion score = v · p_fused
```

The prototype is not renormalized after averaging, and the text feature remains unnormalized, matching the SigLIP notebooks. Image-only scores are the mean of reference/gallery cosine similarities. Fusion scores may lie outside [-1, 1]; neither score is a probability. Do not reuse a threshold across modes, checkpoints, or reference sets without checking its suitability. The historical CLIP branches use different feature handling and are not implemented here.

## Output and failures

The JSON contains the model, actual device, batch size, mode, prompt, selection rule, scoring convention, and relative reference paths. `results` lists successfully scored images in descending score order, with stable path ordering for ties. Each result has:

- `path`: relative to `--image-dir`, so nested duplicate filenames remain distinguishable.
- `score` and `rank`: similarity and one-based position among all successfully scored gallery images.
- `selected`: whether the entry passes Top-K or `score >= threshold`.
- `status`: `selected`, `excluded`, `reference_excluded`, or `image_error`.
- `error`: the image-reading failure message, if any.

Excluded reference entries and image errors follow the ranked records, with null scores/ranks. All gallery records are included, not only the selected ones. Reference paths are relative to `--reference-dir`; keep those input roots with your local run records when resolving paths later. No F1, precision, or recall is calculated.

An unreadable reference aborts the run so the requested prototype is not silently changed. An unreadable gallery image is reported and other gallery images continue. Model loading/encoding failures abort rather than generating an incomplete ranking. An empty gallery or a gallery containing only reference paths is rejected. If all gallery images are unreadable, the report is partial with zero scored images.

Exit status is `0` for a complete run, `2` for a saved partial report containing gallery image errors, and `1` for a runtime failure. Invalid CLI syntax uses argparse's exit status `2` without saving a report. Existing output files are protected unless `--overwrite` is provided. Relative CLI paths resolve against the current working directory. The default output, `.local/image_to_image/results.json`, is ignored by Git.

## Validation status

Automated tests use temporary synthetic images, real CPU tensors, and a fake encoder to check scoring formulas, ordering, thresholds, path portability, reference exclusions, failures, and JSON output. An adapter test checks the expected model/processor calls without downloading weights. These checks do not validate pretrained SigLIP outputs or CUDA execution; a real-model end-to-end run has not been performed in the preparation environment.

```bash
python -m unittest discover -s tests -p 'test_image_retrieval.py' -v
```

This inference entry point does not modify or recompute the report or the notebooks' historical numerical outputs.
