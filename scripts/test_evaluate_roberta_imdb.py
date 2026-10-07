"""Integrity and selection-policy tests; no CUDA/model/network required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluate_roberta_imdb as ev


class EvaluatorIntegrityTests(unittest.TestCase):
    def test_threshold_matches_precision_recall_curve_argmax_ties(self):
        labels = np.array([1, 0, 0, 1])
        scores = np.array([.9, .8, .8, .7])
        # Thresholds .7 and .9 both give F1 2/3; choose the lower sorted tie.
        self.assertEqual(ev.max_f1_threshold(labels, scores), .7)
        self.assertEqual(ev.max_f1_threshold([0, 1], [0., 0.]), 0.)

    def test_validation_threshold_rejects_bad_labels_and_lengths(self):
        with self.assertRaises(ValueError):
            ev.max_f1_threshold([0, 0], [.1, .2])
        with self.assertRaises(ValueError):
            ev.max_f1_threshold([0, 1], [.2])

    def test_fingerprint_binds_revision_input_precision_and_batch(self):
        base, payload = ev.fingerprint(ev.MODEL_REVISION, "abc", 16)
        self.assertEqual(base, ev.fingerprint(ev.MODEL_REVISION, "abc", 16)[0])
        self.assertNotEqual(base, ev.fingerprint(ev.MODEL_REVISION, "abc", 8)[0])
        self.assertNotEqual(base, ev.fingerprint(ev.MODEL_REVISION, "def", 16)[0])
        self.assertEqual(payload["dtype"], "float32")

    def test_manifest_hashes_and_validation_test_split_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            splits = {}
            for split in ("validation", "test"):
                p = root / f"{split}.jsonl"
                p.write_text("{}\n")
                splits[split] = {"file": p.name, "sha256": ev.sha256_file(p), "rows": 1}
            manifest = {"version": 1, "splits": splits}
            self.assertEqual(set(ev.validate_manifest(manifest, root)), {"validation", "test"})
            manifest["splits"]["test"]["sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ev.validate_manifest(manifest, root)
            manifest["splits"]["test"].update(file="validation.jsonl", sha256=splits["validation"]["sha256"])
            with self.assertRaisesRegex(ValueError, "distinct split"):
                ev.validate_manifest(manifest, root)

    def test_row_contract_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rows.jsonl"
            row = dict(source_line=3, movie_id="movie", text="private fixture", label=1)
            p.write_text(json.dumps(row) + "\n")
            self.assertEqual(list(ev.iter_rows(p))[0]["label"], 1)
            row["extra"] = "x"
            p.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly"):
                list(ev.iter_rows(p))


if __name__ == "__main__":
    unittest.main()
