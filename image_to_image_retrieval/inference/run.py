"""Reference images -> mean SigLIP prototype -> ranked gallery results as JSON."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

from project_paths import SIGLIP_MODEL
from .encoder import SigLIPEncoder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def scan_images(root):
    """Return stable relative identities, including nested folders."""
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths = [p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
             and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    if not paths:
        raise ValueError(f"No supported images found in: {root}")
    return [p.relative_to(root).as_posix() for p in paths]


def checked_features(features, count, normalize):
    import torch

    features = features.detach().float().cpu()
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError("Encoder returned an invalid feature shape.")
    if not torch.isfinite(features).all():
        raise ValueError("Encoder returned non-finite features.")
    if normalize:
        norms = features.norm(dim=1, keepdim=True)
        if (norms == 0).any():
            raise ValueError("Encoder returned a zero image feature.")
        features = features / norms
    return features


def encode_batch(rows, root, encoder, strict=False):
    """Bad references abort; bad gallery images remain visible as error records."""
    from PIL import Image

    images, valid = [], []
    try:
        for row in rows:
            try:
                with Image.open(root / row["path"]) as source:
                    image = source.convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                if strict:
                    raise ValueError(f"Cannot read reference image {row['path']}: {exc}") from exc
                row.update(status="image_error", error=str(exc))
                continue
            images.append(image)
            valid.append(row)
        if not images:
            return valid, None
        return valid, checked_features(encoder.encode_images(images), len(images), normalize=True)
    finally:
        for image in images:
            image.close()


def run_inference(reference_dir, image_dir, *, top_k=None, threshold=None,
                  query_text=None, model=SIGLIP_MODEL, device="auto", batch_size=16,
                  encoder_factory=SigLIPEncoder):
    """Use every reference image; no class labels, sampling, or F1 tuning."""
    if (top_k is None) == (threshold is None):
        raise ValueError("Specify exactly one of top_k or threshold.")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer.")
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("threshold must be finite; similarity scores are not probabilities.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", device):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N.")
    if not model.strip():
        raise ValueError("Specify a model identifier or local checkpoint directory.")
    if query_text is not None:
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("query_text must be nonempty when provided.")

    reference_dir = Path(reference_dir).expanduser().resolve()
    image_dir = Path(image_dir).expanduser().resolve()
    references = [{"path": path} for path in scan_images(reference_dir)]
    reference_paths = {(reference_dir / row["path"]).resolve() for row in references}
    rows = []
    for path in scan_images(image_dir):
        excluded = (image_dir / path).resolve() in reference_paths
        rows.append(dict(path=path, score=None, rank=None, selected=False,
                         status="reference_excluded" if excluded else "pending", error=None))
    pending = [row for row in rows if row["status"] == "pending"]
    if not pending:
        raise ValueError("No gallery images remain after excluding the reference files.")

    print(f"Encoding {len(references)} references and scoring {len(pending)} gallery images...", file=sys.stderr)
    encoder = encoder_factory(model, device)
    try:
        prototype = None
        for start in range(0, len(references), batch_size):
            _, features = encode_batch(references[start:start + batch_size], reference_dir, encoder, strict=True)
            batch_sum = features.sum(dim=0, keepdim=True)
            prototype = batch_sum if prototype is None else prototype + batch_sum
        # Preserve the historical formula: normalize each image, then average.
        # Do not normalize this mean again.
        prototype = prototype / len(references)
        if query_text is not None:
            text_feature = checked_features(encoder.encode_text(query_text), 1, normalize=False)
            if text_feature.shape != prototype.shape:
                raise ValueError("Image and text feature dimensions differ.")
            prototype = (prototype + text_feature) / 2
        for start in range(0, len(pending), batch_size):
            valid, features = encode_batch(pending[start:start + batch_size], image_dir, encoder)
            if features is None:
                continue
            scores = (features @ prototype.T).flatten().tolist()
            for row, score in zip(valid, scores):
                if not math.isfinite(score):
                    raise ValueError("Model returned a non-finite similarity score.")
                row.update(score=score, status="scored")
        actual_device = encoder.device
    finally:
        encoder.close()

    ranked = sorted((row for row in rows if row["status"] == "scored"),
                    key=lambda row: (-row["score"], row["path"]))
    for rank, row in enumerate(ranked, 1):
        selected = rank <= top_k if top_k is not None else row["score"] >= threshold
        row.update(rank=rank, selected=selected, status="selected" if selected else "excluded")
    errors = sum(row["status"] == "image_error" for row in rows)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_status": "partial" if errors else "complete",
        "mode": "image_text_fusion" if query_text is not None else "reference_image",
        "model": model,
        "query_text": query_text,
        "scoring": "normalized_gallery_dot_mean_normalized_references" if query_text is None
                   else "normalized_gallery_dot_half_mean_image_plus_raw_text",
        "selection": {"top_k": top_k, "threshold": threshold},
        "parameters": {"device": actual_device, "batch_size": batch_size},
        "references": references,
        "summary": {"references": len(references), "gallery_images": len(rows),
                    "scored": len(ranked), "selected": sum(row["selected"] for row in rows),
                    "reference_excluded": sum(row["status"] == "reference_excluded" for row in rows),
                    "errors": errors},
        # Ranked records first; errors and reference exclusions follow in path order.
        "results": ranked + [row for row in rows if row["rank"] is None],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path, help="Reference images for one retrieval target")
    parser.add_argument("--image-dir", required=True, type=Path, help="Unlabeled gallery, scanned recursively")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--top-k", type=int, help="Select the highest scoring K images")
    selection.add_argument("--threshold", type=float, help="Select scores >= this value; not a probability")
    parser.add_argument("--query-text", help="Optional full text prompt for the notebook's image-text fusion")
    parser.add_argument("--model", default=SIGLIP_MODEL, help="SigLIP model ID or local checkpoint directory")
    parser.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--output", default=Path(".local/image_to_image/results.json"), type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        if output.suffix.lower() != ".json":
            raise ValueError("Output must be a .json file.")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it.")
        report = run_inference(args.reference_dir, args.image_dir, top_k=args.top_k,
                               threshold=args.threshold, query_text=args.query_text,
                               model=args.model, device=args.device, batch_size=args.batch_size)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved results to {output}")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 2 if report["run_status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
