"""Check preparation safety using synthetic files in temporary directories only."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tip_adapter_ft" / "preprocessing.py"
spec = importlib.util.spec_from_file_location("preparation", SCRIPT)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        for split in ("train", "val"):
            for category in ("Dog", "Duck", "Piano"):
                folder = self.source / split / category
                folder.mkdir(parents=True)
                # Preparation copies files; image decoding belongs to the loader.
                (folder / "same-name.jpg").write_bytes(f"{split}/{category}".encode())

    def prepare(self, **kwargs):
        preparation.process_dataset(self.source, self.output, "Dog", **kwargs)

    def test_binary_labels_and_duplicate_filenames_preserved(self):
        self.prepare()
        for split in ("train", "val"):
            for label, category in (("0", "Dog"), ("1", "Duck"), ("1", "Piano")):
                original = self.source / split / category / "same-name.jpg"
                copied = self.output / split / label / category / "same-name.jpg"
                self.assertEqual(copied.read_bytes(), original.read_bytes())

    def test_existing_output_requires_explicit_overwrite(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.prepare(overwrite=True)
        self.assertTrue((self.source / "train/Dog/same-name.jpg").exists())

    def test_foreign_directory_cannot_be_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "personal.txt"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            self.prepare(overwrite=True)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_source_validation_precedes_output_replacement(self):
        self.prepare()
        saved = self.output / "val/0/Dog/same-name.jpg"
        original = saved.read_bytes()
        (self.source / "val/Dog/same-name.jpg").unlink()
        with self.assertRaises(ValueError):
            self.prepare(overwrite=True)
        self.assertEqual(saved.read_bytes(), original)

    def test_overlapping_directories_are_rejected(self):
        for destination in (self.source, self.source / "generated", self.root):
            with self.assertRaises(ValueError):
                preparation.process_dataset(self.source, destination, "Dog", overwrite=True)
        self.assertTrue((self.source / "train/Dog/same-name.jpg").exists())

    def test_output_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        self.output.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.prepare(overwrite=True)
        self.assertEqual((outside / "keep.txt").read_text(), "keep")

    def test_hidden_and_nonimage_files_are_excluded(self):
        folder = self.source / "train/Dog"
        (folder / "._same-name.jpg").write_text("metadata")
        (folder / "notes.txt").write_text("private note")
        self.prepare()
        self.assertEqual([p.name for p in (self.output / "train/0/Dog").iterdir()], ["same-name.jpg"])


if __name__ == "__main__":
    unittest.main()
