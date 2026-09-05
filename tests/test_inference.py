"""Inference contracts with real tiny images and synthetic model backends."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from two_stage_retrieval.inference.config import InferenceConfig
from two_stage_retrieval.inference.retrievers import positive_probabilities
from two_stage_retrieval.inference import run
from two_stage_retrieval.inference.verifier import parse_answer


class InferenceTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.gallery = self.root / "gallery"
        self.gallery.mkdir()
        self.events = []
        self.verified = []
        self.config = InferenceConfig(self.gallery, "cat", "猫", 0.8, 0.8, batch_size=2)

    def image(self, name, color):
        path = self.gallery / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("RGB", (2, 2), color) as image:
            image.save(path)
        return path

    def retriever_factory(self, language, config):
        self.events.append(f"open_{language}")
        events = self.events

        class Retriever:
            def score(self, images):
                channel = 0 if language == "en" else 1
                return [image.getpixel((0, 0))[channel] / 255 for image in images]

            def close(self):
                events.append(f"close_{language}")

        return Retriever()

    def verifier_factory(self, config):
        self.events.append("open_verifier")
        events, verified = self.events, self.verified

        class Verifier:
            def verify(self, path, query):
                relative = path.relative_to(config.image_dir).as_posix()
                verified.append(relative)
                if relative == "failure.png":
                    raise RuntimeError("synthetic verification failure")
                if relative == "uncertain.png":
                    return "I cannot tell."
                return "No." if relative == "b/same.png" else "Yes."

            def close(self):
                events.append("close_verifier")

        return Verifier()

    def execute(self, config=None):
        with redirect_stderr(io.StringIO()):
            return run.run_inference(config or self.config, self.retriever_factory, self.verifier_factory)

    def test_union_by_relative_path_threshold_boundary_and_release_order(self):
        self.image("a/same.png", (204, 51, 0))
        self.image("b/same.png", (51, 204, 0))
        self.image("both.png", (204, 204, 0))
        self.image("neither.png", (51, 51, 0))
        report = self.execute()
        rows = {row["path"]: row for row in report["results"]}
        self.assertEqual(report["summary"], {"images": 4, "candidates": 3, "retained": 2, "unresolved": 0, "errors": 0})
        self.assertEqual(rows["a/same.png"]["status"], "retained")
        self.assertEqual(rows["b/same.png"]["status"], "rejected")
        self.assertEqual(rows["neither.png"]["status"], "excluded")
        self.assertEqual(self.verified.count("both.png"), 1)
        self.assertEqual(self.events, ["open_en", "close_en", "open_zh", "close_zh", "open_verifier", "close_verifier"])

    def test_bad_images_and_uncertain_or_failed_answers_remain_unresolved(self):
        self.image("good.png", (204, 51, 0))
        self.image("uncertain.png", (204, 51, 0))
        self.image("failure.png", (204, 51, 0))
        (self.gallery / "broken.jpg").write_bytes(b"not an image")
        self.image(".hidden.png", (204, 51, 0))
        (self.gallery / "notes.txt").write_text("not an input image")
        report = self.execute()
        rows = {row["path"]: row for row in report["results"]}
        self.assertEqual(report["run_status"], "partial")
        self.assertEqual(report["summary"]["errors"], 3)
        for name, status in [("broken.jpg", "image_error"), ("uncertain.png", "invalid_answer"), ("failure.png", "verification_error")]:
            self.assertEqual(rows[name]["status"], status)
            self.assertIsNone(rows[name]["retained"])
        self.assertEqual(set(rows), {"good.png", "uncertain.png", "failure.png", "broken.jpg"})
        json.dumps(report, allow_nan=False)

    def test_no_candidates_avoids_loading_llava(self):
        self.image("low.png", (51, 51, 0))
        report = self.execute()
        self.assertEqual(report["summary"]["candidates"], 0)
        self.assertNotIn("open_verifier", self.events)

    def test_skip_verification_does_not_claim_verified_matches(self):
        self.image("candidate.png", (204, 51, 0))
        report = self.execute(replace(self.config, skip_verification=True, device="cpu"))
        self.assertEqual(report["mode"], "clip_only")
        self.assertEqual(report["results"][0]["status"], "candidate_unverified")
        self.assertIsNone(report["results"][0]["retained"])
        self.assertNotIn("open_verifier", self.events)

    def test_all_unreadable_images_do_not_load_models(self):
        (self.gallery / "bad.jpg").write_bytes(b"bad")
        report = self.execute()
        self.assertEqual(report["run_status"], "partial")
        self.assertEqual(self.events, [])

    def test_cli_writes_json_and_refuses_unrequested_overwrite(self):
        self.image("candidate.png", (204, 51, 0))
        output = self.root / "output/results.json"
        arguments = ["--image-dir", str(self.gallery), "--query-en", "cat", "--query-zh", "猫",
                     "--threshold-en", "0.8", "--threshold-zh", "0.8", "--output", str(output)]
        actual_run = run.run_inference
        with patch.object(run, "run_inference", side_effect=lambda config: actual_run(config, self.retriever_factory, self.verifier_factory)) as backend:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(arguments), 0)
                before = output.read_bytes()
                self.assertEqual(run.main(arguments), 1)
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(backend.call_count, 1)
                (self.gallery / "bad.jpg").write_bytes(b"bad")
                self.assertEqual(run.main(arguments + ["--overwrite"]), 2)
        self.assertEqual(json.loads(output.read_text())["run_status"], "partial")

    def test_invalid_configuration_is_rejected_before_loading(self):
        for changes in ({"threshold_en": float("nan")}, {"threshold_zh": 1.1},
                        {"batch_size": 0}, {"query_en": " "}, {"device": "cpu"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.execute(replace(self.config, **changes))
        self.assertEqual(self.events, [])

    def test_model_failure_releases_retriever(self):
        self.image("valid.png", (204, 51, 0))
        events = []

        class BrokenRetriever:
            def score(self, images):
                raise RuntimeError("synthetic model failure")

            def close(self):
                events.append("closed")

        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            run.run_inference(self.config, lambda language, config: BrokenRetriever(), self.verifier_factory)
        self.assertEqual(events, ["closed"])

    def test_help_does_not_import_model_dependencies(self):
        code = (
            "import sys; from two_stage_retrieval.inference.run import build_parser; "
            "assert not any(m in sys.modules for m in ('torch', 'transformers', 'clip', 'llava')); "
            "build_parser().parse_args(['--help'])"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--threshold-en", result.stdout)


class ScoringTests(unittest.TestCase):
    def test_normalization_logit_scale_and_two_prompt_softmax(self):
        import torch

        scores = positive_probabilities(
            torch.tensor([[3.0, 0.0], [0.0, 9.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 4.0]]), torch.tensor(math.log(2.0)),
        )
        expected = 1 / (1 + math.exp(-2))
        self.assertAlmostEqual(scores[0], expected, places=6)
        self.assertAlmostEqual(scores[1], 1 - expected, places=6)

    def test_yes_no_parser_does_not_accept_ambiguous_text(self):
        for answer in ("Yes.", " YES, it is a cat.", "yes"):
            self.assertIs(parse_answer(answer), True)
        for answer in ("No.", "no, it is a dog."):
            self.assertIs(parse_answer(answer), False)
        for answer in ("", "Maybe", "I cannot tell", "Yes or No", "yesterday", "nobody"):
            self.assertIsNone(parse_answer(answer))


if __name__ == "__main__":
    unittest.main()
