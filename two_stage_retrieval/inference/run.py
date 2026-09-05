"""CLI: gallery -> bilingual CLIP union -> LLaVA decisions -> JSON."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

from .config import InferenceConfig, ENGLISH_MODEL, TAIYI_IMAGE_MODEL, TAIYI_TEXT_MODEL
from .retrievers import CLIPRetriever
from .verifier import LLaVAVerifier, parse_answer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def scan_gallery(root):
    """Use relative paths as identities; filenames need not be globally unique."""
    from PIL import Image

    results = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row = dict(path=relative.as_posix(), score_en=None, score_zh=None,
                   candidate=False, answer=None, retained=None, status="pending", error=None)
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            row.update(status="image_error", error=str(exc))
        results.append(row)
    if not results:
        raise ValueError("No supported image files found in the gallery.")
    return results


def score_gallery(rows, config, language, factory):
    from PIL import Image

    valid = [row for row in rows if row["status"] != "image_error"]
    if not valid:
        return
    print(f"Scoring {len(valid)} images with {language} CLIP...", file=sys.stderr)
    retriever = factory(language, config)
    try:
        for start in range(0, len(valid), config.batch_size):
            images, batch_rows = [], []
            try:
                for row in valid[start:start + config.batch_size]:
                    try:
                        with Image.open(config.image_dir / row["path"]) as source:
                            image = source.convert("RGB")
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        row.update(status="image_error", error=str(exc))
                        continue
                    images.append(image)
                    batch_rows.append(row)
                if not images:
                    continue
                scores = retriever.score(images)
                if len(scores) != len(batch_rows):
                    raise RuntimeError("Model returned a different number of scores than images.")
                for row, score in zip(batch_rows, scores):
                    score = float(score)
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise RuntimeError("Model returned an invalid probability.")
                    row[f"score_{language}"] = score
            finally:
                for image in images:
                    image.close()
    finally:
        retriever.close()


def run_inference(config, retriever_factory=CLIPRetriever, verifier_factory=LLaVAVerifier):
    config.validate()
    rows = scan_gallery(config.image_dir)
    # The model pair is released after each pass, before LLaVA is loaded.
    for language in ("en", "zh"):
        score_gallery(rows, config, language, retriever_factory)
    candidates = []
    for row in rows:
        if row["status"] == "image_error":
            continue
        row["candidate"] = row["score_en"] >= config.threshold_en or row["score_zh"] >= config.threshold_zh
        if row["candidate"]:
            row["status"] = "candidate_unverified"
            candidates.append(row)
        else:
            row.update(status="excluded", retained=False)
    if candidates and not config.skip_verification:
        print(f"Verifying {len(candidates)} candidates with LLaVA...", file=sys.stderr)
        verifier = verifier_factory(config)
        try:
            for row in candidates:
                try:
                    row["answer"] = verifier.verify(config.image_dir / row["path"], config.query_en)
                    decision = parse_answer(row["answer"])
                    row["retained"] = decision
                    row["status"] = "invalid_answer" if decision is None else ("retained" if decision else "rejected")
                except Exception as exc:
                    row.update(status="verification_error", retained=None, error=str(exc))
        finally:
            verifier.close()
    errors = sum(row["status"] in {"image_error", "verification_error", "invalid_answer"} for row in rows)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_status": "partial" if errors else "complete",
        "mode": "clip_only" if config.skip_verification else "two_stage",
        "query": {"en": config.query_en, "zh": config.query_zh},
        "thresholds": {"en": config.threshold_en, "zh": config.threshold_zh},
        "models": {"english_clip": ENGLISH_MODEL, "taiyi_text": TAIYI_TEXT_MODEL,
                   "taiyi_image": TAIYI_IMAGE_MODEL, "llava": None if config.skip_verification else config.llava_model},
        "parameters": {"device": config.device, "batch_size": config.batch_size,
                       "llava_do_sample": False, "llava_max_new_tokens": 32},
        "summary": {"images": len(rows), "candidates": len(candidates),
                    "retained": sum(row["retained"] is True for row in rows),
                    "unresolved": sum(row["retained"] is None for row in rows), "errors": errors},
        "results": rows,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path, help="Unlabeled image directory; scanned recursively")
    parser.add_argument("--query-en", required=True, help="English target name, e.g. cat")
    parser.add_argument("--query-zh", required=True, help="Chinese target name, e.g. 猫")
    parser.add_argument("--threshold-en", required=True, type=float, help="English target-versus-Others softmax threshold [0,1]")
    parser.add_argument("--threshold-zh", required=True, type=float, help="Taiyi target-versus-Others softmax threshold [0,1]")
    parser.add_argument("--output", type=Path, default=Path(".local/inference/results.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (CLIP-only)")
    parser.add_argument("--llava-model", default=os.environ.get("RETRIEVAL_LLAVA_MODEL", "liuhaotian/llava-v1.5-7b"))
    parser.add_argument("--skip-verification", action="store_true", help="Return unverified CLIP candidates without loading LLaVA")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of an existing JSON output")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        if output.suffix.lower() != ".json":
            raise ValueError("Output must be a .json file.")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it.")
        config = InferenceConfig(
            image_dir=args.image_dir.expanduser().resolve(), query_en=args.query_en.strip(),
            query_zh=args.query_zh.strip(), threshold_en=args.threshold_en, threshold_zh=args.threshold_zh,
            batch_size=args.batch_size, device=args.device, llava_model=args.llava_model,
            skip_verification=args.skip_verification,
        )
        report = run_inference(config)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {report['summary']['images']} image records to {output}")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 2 if report["run_status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
