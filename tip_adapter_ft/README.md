# Tip-Adapter for target-versus-background retrieval

This module adapts the [upstream Tip-Adapter implementation](https://github.com/gaopengcuhk/Tip-Adapter) to the project's five target categories. It is not the official Tip-Adapter repository. The original method is due to Zhang et al., ECCV 2022; see [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution and the unresolved source-permission status.

## Project-specific changes

- Prepare one target class and an Others class for each experiment.
- Build text features from target-specific positive and negative prompt templates.
- Evaluate threshold-based binary retrieval using F1 rather than the upstream top-1 classification metric.
- Compare a fixed feature cache with learned cache keys while keeping CLIP frozen.

The fused score follows the upstream method: CLIP logits plus a weighted exponential cache-affinity score. `run_tip_adapter_F` optimizes only the cache-key adapter with cross-entropy.

## Reading order

1. `main_imagenet.py`: scoring, historical evaluation protocol, prompts and training loop.
2. `utils.py`: text classifier, feature cache, feature loading and alpha/beta search.
3. `imagenet.py`: binary dataset loading and per-class K-shot sampling.
4. `preprocessing.py`: explicit, separate-output dataset preparation.
5. `imagenet.yaml`: one target/shot/backbone configuration.

The `ImageNet` class and filenames are inherited names; this experiment uses the custom retrieval data described in [DATASET.md](DATASET.md), not the ImageNet benchmark.

## Execution

From the repository root, in a compatible CUDA environment:

```bash
python -m pip install -r tip_adapter_ft/requirements.txt

# Supply correctly structured source data first.
python tip_adapter_ft/preprocessing.py --source_dir data/tip_adapter --target_dir .local/tip_adapter/datasets/images --keep_classes Dog

python tip_adapter_ft/main_imagenet.py --config tip_adapter_ft/imagenet.yaml
```

Alternatively, `main_imagenet.py --prepare-data` prepares data before training. Replacing existing generated data additionally requires `--overwrite-prepared-data`. The tool refuses to overwrite an unrecognized directory or a directory overlapping its source.

The checked-in configuration is **Dog / 200 shots / RN50**. Each run evaluates one target; it does not aggregate five-class macro F1 or run the four shot settings from the report. When changing target, regenerate the prepared data for that target. Caches are separated by class/backbone/shot under `.local/tip_adapter/caches/`; leave cache loading disabled when changing data, prompts or other settings.

## Historical evaluation

The original algorithm searches thresholds and alpha/beta using evaluation features and selects the best training epoch on that evaluation set. `val` and `test` name the same split in the loader. These behaviors are retained as historical implementation details, not an independent-test benchmark claim.

The report's numerical results remain unchanged. Cleanup adds portable paths, explicit preparation and output-directory initialization; it does not provide a newly verified reproduction. See [setup](../docs/SETUP.md).
