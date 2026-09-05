"""Real tiny images/tensors, fake encoder: no downloads or private datasets."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from image_to_image_retrieval.inference import run
from image_to_image_retrieval.inference.encoder import SigLIPEncoder
from project_paths import REFERENCE_CLASS_FOLDERS, reference_class_indices

ROOT = Path(__file__).resolve().parents[1]


class ColorEncoder:
    """Raw RGB feature vectors let tests predict scores independently."""
    def __init__(self, model, device):
        self.device = "cpu"
        self.closed = False

    def encode_images(self, images):
        return torch.tensor([image.getpixel((0, 0)) for image in images], dtype=torch.float32)

    def encode_text(self, text):
        return torch.tensor([[0., 0., 4.]])

    def close(self):
        self.closed = True


class ReferenceConfigTests(unittest.TestCase):
    def test_default_mapping_and_custom_folder_order(self):
        self.assertEqual(reference_class_indices({str(i): i for i in range(6)}),
                         {"Dog": 0, "Duck": 1, "Erhu": 2, "Piano": 3, "Porcelain": 4})
        # Background sorts first, and semantic order differs from directory order.
        labels = {"Background": 0, "Car": 1, "Cat": 2}
        self.assertEqual(reference_class_indices(labels, {"Cat": "Cat", "Car": "Car"}),
                         {"Cat": 2, "Car": 1})
        with self.assertRaisesRegex(ValueError, "missing"):
            reference_class_indices(labels, {"Dog": "Dog"})
        with self.assertRaisesRegex(ValueError, "at least one"):
            reference_class_indices(labels, {})

    def test_portable_gallery_and_model_configuration(self):
        with patch.dict(os.environ, {"RETRIEVAL_GALLERY_DIR": "custom/gallery",
                                     "RETRIEVAL_SIGLIP_MODEL": "example/local-checkpoint"}):
            config = runpy.run_path(str(ROOT / "project_paths.py"))
        self.assertEqual(config["GALLERY_DIR"], ROOT / "custom/gallery")
        self.assertEqual(config["SIGLIP_MODEL"], "example/local-checkpoint")

    @unittest.skipUnless((ROOT / "image_to_image_retrieval/reference_retrieval_debug.ipynb").exists(),
                         "Local scratchpad is omitted from the public snapshot")
    def test_debug_evaluation_uses_selected_target(self):
        notebook = ROOT / "image_to_image_retrieval/reference_retrieval_debug.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        source = next("".join(c["source"]) for c in cells
                      if "best_f1,best_precision,best_recall=" in "".join(c["source"]))
        import numpy as np
        observed = []
        namespace = dict(gallery_dataset=[(None, 2)], np=np, similarity=[1.],
                         target_label=2, target_class="Cat",
                         find_threshold=lambda scores, labels, target: (observed.append(target) or 1., 1., 1.))
        with redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(observed, [2])


class ReferenceInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs = self.root / "references"
        self.gallery = self.root / "gallery"
        self.write_image(self.refs / "red.png", (255, 0, 0))
        self.write_image(self.refs / "green.png", (0, 255, 0))
        self.write_image(self.gallery / "nested/mixed.PNG", (255, 255, 0))
        self.write_image(self.gallery / "red.png", (255, 0, 0))
        self.write_image(self.gallery / "green.png", (0, 255, 0))
        self.write_image(self.gallery / "blue.png", (0, 0, 255))
        self.encoder = ColorEncoder("test", "cpu")
        self.factory = lambda model, device: self.encoder

    def write_image(self, path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("RGB", (2, 2), color) as image:
            image.save(path)

    def infer(self, **kwargs):
        with redirect_stderr(io.StringIO()):
            return run.run_inference(self.refs, self.gallery, encoder_factory=self.factory, **kwargs)

    def test_top_k_preserves_mean_norm_and_stable_ties(self):
        report = self.infer(top_k=2, batch_size=1)
        rows = report["results"]
        self.assertEqual([r["path"] for r in rows], ["nested/mixed.PNG", "green.png", "red.png", "blue.png"])
        self.assertAlmostEqual(rows[0]["score"], 2 ** -.5, places=6)
        self.assertAlmostEqual(rows[1]["score"], .5)
        self.assertEqual([r["selected"] for r in rows], [True, True, False, False])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3, 4])
        self.assertTrue(self.encoder.closed)
        self.assertEqual(report["run_status"], "complete")

    def test_threshold_is_inclusive_and_large_top_k_is_clamped(self):
        report = self.infer(threshold=.5, batch_size=3)
        self.assertEqual(report["summary"]["selected"], 3)
        self.assertEqual(self.infer(top_k=50)["summary"]["selected"], 4)
        self.assertEqual(self.infer(threshold=2)["summary"]["selected"], 0)

    def test_text_fusion_keeps_raw_text_scale(self):
        report = self.infer(threshold=1.5, query_text="a blue object")
        self.assertEqual(report["mode"], "image_text_fusion")
        self.assertEqual(report["results"][0]["path"], "blue.png")
        self.assertAlmostEqual(report["results"][0]["score"], 2.)
        self.assertEqual(report["summary"]["selected"], 1)

    def test_corrupt_gallery_is_reported_and_hidden_files_are_ignored(self):
        (self.gallery / "broken.jpg").write_bytes(b"not an image")
        self.write_image(self.gallery / ".cache/hidden.png", (255, 0, 0))
        report = self.infer(top_k=2)
        self.assertEqual(report["run_status"], "partial")
        self.assertEqual(report["summary"]["gallery_images"], 5)
        self.assertEqual(report["summary"]["errors"], 1)
        error = report["results"][-1]
        self.assertEqual(error["status"], "image_error")
        self.assertIsNone(error["score"])
        self.assertFalse(error["selected"])

    def test_corrupt_reference_aborts_and_releases_encoder(self):
        (self.refs / "broken.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "reference image"):
            self.infer(top_k=1)
        self.assertTrue(self.encoder.closed)

    def test_overlapping_references_and_symlinks_are_excluded(self):
        self.refs = self.gallery / "refs"
        self.write_image(self.refs / "reference.png", (255, 0, 0))
        (self.gallery / "alias.png").symlink_to(self.refs / "reference.png")
        report = self.infer(top_k=99)
        self.assertEqual(report["summary"]["reference_excluded"], 2)
        self.assertEqual(report["summary"]["scored"], 4)
        self.assertTrue(all(row["score"] is None for row in report["results"][-2:]))

    def test_gallery_move_keeps_relative_identifiers(self):
        before = self.infer(top_k=2)
        moved = self.root / "moved gallery"
        self.gallery.rename(moved)
        self.gallery = moved
        after = self.infer(top_k=2)
        self.assertEqual(before["results"], after["results"])
        self.assertTrue(all(not Path(r["path"]).is_absolute() for r in after["results"]))

    def test_bad_options_and_empty_gallery_fail_before_loading(self):
        for options in [{}, {"top_k": 0}, {"top_k": 1, "threshold": .5},
                        {"threshold": float("nan")}, {"top_k": 1, "batch_size": 0},
                        {"top_k": 1, "query_text": " "}, {"top_k": 1, "device": "tpu"}]:
            with self.subTest(options=options), patch.object(run, "SigLIPEncoder"):
                with self.assertRaises(ValueError):
                    self.infer(**options)
        self.gallery = self.root / "empty"
        self.gallery.mkdir()
        with self.assertRaisesRegex(ValueError, "No supported images"):
            self.infer(top_k=1)

    def test_invalid_model_features_abort(self):
        for feature in [torch.zeros((1, 3)), torch.full((1, 3), float("nan"))]:
            with self.subTest(feature=feature):
                with self.assertRaises(ValueError):
                    run.checked_features(feature, 1, normalize=True)

    def test_cli_json_exit_status_and_overwrite_protection(self):
        output = self.root / "out/results.json"
        argv = ["--reference-dir", str(self.refs), "--image-dir", str(self.gallery),
                "--top-k", "2", "--output", str(output)]
        real_run = run.run_inference
        def fake_run(*args, **kwargs):
            return real_run(*args, **kwargs, encoder_factory=self.factory)
        with patch.object(run, "run_inference", side_effect=fake_run), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(argv), 0)
            saved = output.read_bytes()
            self.assertEqual(run.main(argv), 1)
            self.assertEqual(output.read_bytes(), saved)
            (self.gallery / "broken.jpg").write_bytes(b"broken")
            self.assertEqual(run.main(argv + ["--overwrite"]), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["run_status"], "partial")
        self.assertEqual(report["summary"]["errors"], 1)


class EncoderAdapterTests(unittest.TestCase):
    def test_adapter_uses_frozen_raw_features_and_matching_processor(self):
        class Batch(dict):
            def to(self, device):
                return self
        class Processor:
            def __call__(self, images, return_tensors):
                return Batch(pixel_values=torch.ones(len(images), 3))
            def tokenizer(self, texts, **kwargs):
                self.text_options = kwargs
                return Batch(input_ids=torch.ones(1, 8, dtype=torch.int64))
        class Model:
            config = SimpleNamespace(model_type="siglip", text_config=SimpleNamespace(max_position_embeddings=8))
            def to(self, device):
                return self
            def eval(self):
                self.evaluating = True
                return self
            def get_image_features(self, pixel_values):
                self.grad_enabled = torch.is_grad_enabled()
                return pixel_values * 3
            def get_text_features(self, **inputs):
                return torch.tensor([[0., 0., 4.]])
        model, processor = Model(), Processor()
        loaded = []
        fake_transformers = SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda name: (loaded.append(name) or model)),
            AutoProcessor=SimpleNamespace(from_pretrained=lambda name: (loaded.append(name) or processor)))
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            encoder = SigLIPEncoder("local/model", "cpu")
            try:
                self.assertEqual(encoder.encode_images([object()]).tolist(), [[3., 3., 3.]])
                self.assertEqual(encoder.encode_text("blue").tolist(), [[0., 0., 4.]])
                self.assertEqual(processor.text_options["padding"], "max_length")
                self.assertTrue(model.evaluating)
                self.assertFalse(model.grad_enabled)
                self.assertEqual(loaded, ["local/model", "local/model"])
            finally:
                encoder.close()


if __name__ == "__main__":
    unittest.main()
