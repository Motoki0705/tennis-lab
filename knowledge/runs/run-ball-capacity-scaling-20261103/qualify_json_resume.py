"""Reuse the historical no-update guard suite at the verified 5500 checkpoint."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "historical_capacity_resume_tests", DIRECTORY / "test_resume_json_plan.py"
)
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)


class CurrentFrozenJsonResume(historical.TestFrozenJsonResume):
    """Keep all historical guards; update the explicit restored test boundary."""

    def test_adapter_reaches_real_training_boundary_with_recovered_checkpoint(self):
        output = historical.ROOT / "query-d512-l4-T64"
        latest = json.loads((output / "latest.json").read_text())
        self.assertEqual(latest["update"], 5500)
        self.assertEqual(latest["checkpoint"], "checkpoint-update-00005500.pt")
        with self.boundary(True) as sentinel:
            with self.assertRaises(historical.TrainingBoundaryReached):
                self.controller.train(self.args)
            sentinel.assert_called_once()
        kwargs = sentinel.call_args.kwargs
        self.assertEqual(kwargs["output"], output)
        self.assertEqual(kwargs["resume"], output / latest["checkpoint"])
        self.assertIsInstance(kwargs["model_config"].stem_channels, tuple)
        self.assertEqual(kwargs["model_config"].dim, 512)
        self.assertEqual(kwargs["budget"].updates, 6000)
        self.assertEqual(kwargs["budget"].schedule_updates, 12000)


if __name__ == "__main__":
    import torch

    assert not torch.cuda.is_initialized()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CurrentFrozenJsonResume)
    )
    assert not torch.cuda.is_initialized(), "Qualification must not initialize CUDA"
    print(
        json.dumps(
            dict(
                schema="ball_capacity_current_json_resume_guard_suite.v1",
                successful=result.wasSuccessful(),
                tests=result.testsRun,
                restored_update=5500,
                gpu_updates=0,
                controller_changed=False,
                historical_suite="test_resume_json_plan.py",
            )
        ),
        flush=True,
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
