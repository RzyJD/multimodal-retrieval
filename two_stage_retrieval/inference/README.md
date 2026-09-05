# Inference on your own image gallery

This entry point accepts an **unlabeled image directory**, an English/Chinese query pair, and two supplied thresholds. It scores the gallery with English CLIP and Taiyi-CLIP, takes the union by image path, verifies candidates with LLaVA, and writes JSON. It does not read class labels, fit thresholds or calculate F1.

## Setup

Run commands from the repository root. `--help` works without loading PyTorch, Transformers or model weights:

```bash
python -m two_stage_retrieval.inference.run --help
```

Use a Python 3.10 environment and a PyTorch build suitable for your CUDA hardware. The provided [requirements](requirements.txt) select the model-facing versions recorded in the original LLaVA dependency manifest; they are not a complete environment lockfile or a newly GPU-tested installation.

```bash
python -m pip install -r two_stage_retrieval/inference/requirements.txt
```

Full two-stage inference additionally requires the **upstream LLaVA v1.5 implementation** in the same environment as this command. Follow [the project's LLaVA setup notes](../../docs/SETUP.md#llava-environment) and upstream instructions. The verifier imports `llava.model.builder` and `llava.conversation`; another package sharing the name `llava` is not interchangeable. Keep its dependency versions compatible with the file above. No dependency installation or model download is performed by `--help` or by importing this package.

The implementation supports OpenAI CLIP ViT-B/32, Taiyi's Chinese text encoder with OpenAI CLIP ViT-L/14, and **LLaVA v1.5-7B (LLaMA/Vicuna family)**. `--llava-model` changes the location of that model; it is not a generic switch to Qwen, LLaVA-NeXT or other architectures.

## Run

```bash
python -m two_stage_retrieval.inference.run \
  --image-dir "/path/to/images" \
  --query-en "cat" \
  --query-zh "猫" \
  --threshold-en 0.95 \
  --threshold-zh 0.95 \
  --device cuda \
  --batch-size 16 \
  --output ".local/inference/cat_results.json"
```

**The thresholds above illustrate the command format; they are not recommended values.** Supply thresholds calibrated for your model/query/data, or explicitly choose operating thresholds. Unlabeled images cannot provide an F1-optimal threshold. Each score is the two-prompt target-versus-Others softmax value, not a calibrated probability of correctness.

Use English/Chinese names for the same visual target. Queries are independent of the notebook's `TARGET_CLASSES` mapping, so it does not need editing for this entry point. Images can appear directly in the input directory or in nested folders; folder names have no label meaning. Image records use paths relative to `--image-dir`, keeping identically named files in different folders distinct. Hidden paths and unsupported extensions are ignored; unreadable supported image files are reported as errors.

Optional controls:

- `--llava-model /path/to/llava-v1.5-7b`: a local checkpoint. Defaults to `RETRIEVAL_LLAVA_MODEL` if set, otherwise `liuhaotian/llava-v1.5-7b`.
- `--device cuda:1`: select another available GPU. Full verification requires CUDA.
- `--skip-verification --device cpu`: run CLIP candidate generation only, without LLaVA installed. Candidate decisions remain **unverified**, with `retained: null`.
- `--overwrite`: explicitly replace an existing output JSON file. Otherwise an existing output is rejected before models load.

CLI image/output paths resolve from the current working directory. Model weights may download when a run starts. English CLIP, Taiyi-CLIP and LLaVA are loaded sequentially and released between stages to reduce simultaneous memory use. Reduce batch size if necessary; no specific GPU-memory requirement has been measured here. LLaVA is not loaded when there are no candidates or verification is skipped.

## Output

The JSON contains the query, thresholds, model names, execution parameters, summary counts and **one record per supported image file**:

```json
{
  "path": "animals/example.jpg",
  "score_en": 0.98,
  "score_zh": 0.72,
  "candidate": true,
  "answer": "Yes.",
  "retained": true,
  "status": "retained",
  "error": null
}
```

This record is illustrative, not a measured result. Filter records with `retained == true` for verified matches. No input image is copied, changed or deleted.

| Status | Meaning | `retained` |
| --- | --- | --- |
| `excluded` | Both CLIP scores are below their thresholds | `false` |
| `retained` | At least one threshold passed; verifier answered Yes | `true` |
| `rejected` | Candidate; verifier answered No | `false` |
| `invalid_answer` | Answer could not be parsed as an unambiguous leading Yes/No | `null` |
| `verification_error` | Candidate verification failed | `null` |
| `image_error` | Image could not be read | `null` |
| `candidate_unverified` | Candidate generation only; verification intentionally skipped | `null` |

Exit code `0` means the requested mode completed without per-image errors; `2` means JSON was written but includes image errors, verification errors or invalid answers. Exit code `1` means a fatal configuration, dependency, model-loading/scoring or output failure; the run does not promise a result file in that case. `--skip-verification` is a distinct mode and can complete successfully without any verified matches.

## Relation to the historical experiments

Model choice, two-prompt scoring, normalization and positive-set union are extracted from the notebooks. This entry point adds path-based identities and label-free operation. It also uses float32 score normalization, supplies Taiyi's padding attention mask, performs greedy LLaVA decoding (`do_sample=False`, maximum 32 new tokens), and accepts only unambiguous leading Yes/No answers. Ambiguous answers and failures remain unresolved rather than being silently accepted.

These are explicit inference behaviors, not a re-execution of the original experiment protocol. The historical notebooks, report and saved experimental results remain unchanged. No equivalence to their reported scores is asserted.

## Code map and checks

- `config.py`: required queries/thresholds and model identifiers.
- `retrievers.py`: query features and batched English/Taiyi scoring.
- `verifier.py`: LLaVA loading, prompt construction, generation and answer parsing.
- `run.py`: gallery scan, candidate union, per-image statuses and JSON CLI.

The lightweight tests use real temporary images, synthetic scorer/verifier backends, and small CPU tensors; they do not download or execute pretrained models:

```bash
python -B -m unittest discover -s tests -p 'test_inference.py' -v
```

Real CUDA execution with CLIP, Taiyi and LLaVA has not been validated on the development machine.
