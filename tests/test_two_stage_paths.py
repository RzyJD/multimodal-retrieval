"""Exercise notebook path orchestration without importing or loading models."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
MATCHING = ROOT / "two_stage_retrieval/image_text_matching"
UNION = MATCHING / "merge_clip_candidates.ipynb"
VERIFIER = ROOT / "two_stage_retrieval/llava_verification/llava_candidate_verification.ipynb"


def cell_source(notebook, fragment):
    for cell in json.loads(notebook.read_text())["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and fragment in source:
            return source
    raise AssertionError(f"Cell not found: {fragment}")


def load_function(notebook, name, namespace):
    tree = ast.parse(cell_source(notebook, f"def {name}("))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)


class TwoStagePathTests(unittest.TestCase):
    def test_environment_paths_are_independent_of_notebook_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "not-created"
            with patch.dict(os.environ, {
                "RETRIEVAL_PICTURE_DIR": temp,
                "RETRIEVAL_OUTPUT_DIR": str(output),
                "RETRIEVAL_LLAVA_MODEL": "example/model",
            }):
                config = runpy.run_path(str(ROOT / "project_paths.py"))
            self.assertEqual(config["PICTURE_DIR"], Path(temp).resolve())
            self.assertEqual(config["OUTPUT_DIR"], output.resolve())
            self.assertEqual(config["LLAVA_MODEL"], "example/model")
            self.assertFalse(output.exists())
            with patch.dict(os.environ, {"RETRIEVAL_PICTURE_DIR": "custom/gallery"}):
                config = runpy.run_path(str(ROOT / "project_paths.py"))
            self.assertEqual(config["PICTURE_DIR"], ROOT / "custom/gallery")

    def test_custom_categories_and_stale_prediction_rejection(self):
        classes = {"Cat": "猫", "Car": "汽车"}
        namespace = {"TARGET_CLASSES": classes}
        for name in ("ePclass", "eNclass", "cPclass", "cNclass"):
            namespace[name] = {key: [] for key in classes}
        source = cell_source(UNION, "Prediction categories differ")
        with redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(namespace["picture"], ["Cat", "Car"])
        namespace["cPclass"] = {"Dog": []}
        with self.assertRaisesRegex(ValueError, "Regenerate both CLIP outputs"):
            exec(source, namespace)

    def test_relative_manifests_survive_gallery_move_and_match_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            gallery = base / "original gallery"
            output = base / "results"
            output.mkdir()
            for directory in ("Cat/Cat", "Cat/Fox", "Car/Car", "Car/Truck"):
                folder = gallery / directory
                folder.mkdir(parents=True)
                # File contents are irrelevant to this path/identifier test.
                (folder / "sample image.jpg").write_bytes(b"synthetic fixture")
            namespace = {
                "Path": Path, "os": os, "pickle": pickle,
                "PICTURE_DIR": gallery, "OUTPUT_DIR": output,
                "TARGET_CLASSES": {"Cat": "猫", "Car": "汽车"},
                "picture": ["Cat", "Car"],
                "Pclass": {"Cat": [1], "Car": [1]},
                "Nclass": {"Cat": [1, 1, 1], "Car": [1, 1, 1]},
            }
            for name in ("subfolder", "get_figure_path", "create_path"):
                load_function(UNION, name, namespace)
            exec(cell_source(UNION, "dataset={'Path':path"), namespace)

            moved = base / "moved gallery"
            shutil.move(str(gallery), moved)
            namespace["PICTURE_DIR"] = moved
            namespace["Image"] = Mock()
            load_function(VERIFIER, "load_image", namespace)
            for category in namespace["picture"]:
                with (output / f"{category} llava dataset").open("rb") as handle:
                    manifest = pickle.load(handle)
                self.assertEqual(len(manifest["Path"]), 4)
                for relative in manifest["Path"]:
                    self.assertFalse(Path(relative).is_absolute())
                    self.assertNotIn("\\", relative)
                    self.assertTrue((moved / relative).is_file())
                    namespace["load_image"](relative)
                    namespace["Image"].open.assert_called_with(moved / relative)
                answers = {
                    "Path": manifest["Path"],
                    "Pre": ["Yes" if truth else "No" for truth in manifest["True"]],
                }
                with (output / f"{category}推理结果").open("wb") as handle:
                    pickle.dump(answers, handle)

            # Execute the real result-matching cell with synthetic answers.
            with redirect_stdout(io.StringIO()):
                exec(cell_source(VERIFIER, "evaluation={x:"), namespace)
            for metrics in namespace["evaluation"].values():
                self.assertEqual(metrics, {"Precision": 1.0, "Recall": 1.0, "F1": 1.0})

            # Existing absolute paths on the same machine still work.
            absolute = moved / "Cat/Cat/sample image.jpg"
            namespace["load_image"](str(absolute))
            namespace["Image"].open.assert_called_with(absolute)


if __name__ == "__main__":
    unittest.main()
