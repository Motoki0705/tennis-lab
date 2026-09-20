"""Small CPU fixtures for the experiment's fail-closed comparison policy."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "one_clip_probe", Path(__file__).with_name("probe.py")
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_supported_identical_and_changed_teacher_or_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a", Path(temporary) / "b"
            a.mkdir()
            b.mkdir()
            teacher = np.ones((3, 2, 3), dtype=np.float32)
            mask = np.ones((3, 2), dtype=np.bool_)
            for root in (a, b):
                np.savez(
                    root / "scene.npz", player_position=teacher, supported_mask=mask
                )
                (root / "quality.json").write_text(
                    json.dumps({"weights": [1, 0], "sources": [2, 0]})
                )
            self.assertTrue(probe.compare_trees(a, b, {})["equal"])
            changed = teacher.copy()
            changed[1, 1, 0] += 0.125
            np.savez(b / "scene.npz", player_position=changed, supported_mask=mask)
            comparison = probe.compare_trees(a, b, {})
            self.assertFalse(comparison["equal"])
            self.assertEqual(
                comparison["files"]["scene.npz"]["arrays"]["player_position"][
                    "max_abs_diff_finite"
                ],
                0.125,
            )
            changed_mask = mask.copy()
            changed_mask[1, 1] = False
            np.savez(
                b / "scene.npz", player_position=teacher, supported_mask=changed_mask
            )
            self.assertFalse(probe.compare_trees(a, b, {})["equal"])
            np.savez(b / "scene.npz", player_position=teacher, supported_mask=mask)
            (b / "quality.json").write_text(
                json.dumps({"weights": [0, 0], "sources": [2, 0]})
            )
            self.assertFalse(probe.compare_trees(a, b, {})["equal"])

    def test_existing_output_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary)
            with self.assertRaises(FileExistsError):
                probe.require_fresh([existing])
            probe.require_fresh([existing / "a", existing / "b"])
            with self.assertRaises(ValueError):
                probe.require_fresh([existing / "a", existing / "a" / "b"])

    def test_only_declared_location_fields_normalize(self) -> None:
        value = {"output_dir": "new/run/clip", "prediction": "new/run/clip"}
        self.assertEqual(
            probe.normalize_document(value, {"new/run": "old/run"}),
            {"output_dir": "old/run/clip", "prediction": "new/run/clip"},
        )


if __name__ == "__main__":
    unittest.main()
