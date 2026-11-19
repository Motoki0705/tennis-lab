"""Reuse historical no-update guards at an explicitly restored checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "historical_capacity_resume_tests", DIRECTORY / "test_resume_json_plan.py"
)
historical = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = historical
spec.loader.exec_module(historical)


def verified_target(plan, stage, update, latest):
    entries = [entry for entry in plan["stages"] if entry["name"] == stage]
    if len(entries) != 1:
        raise ValueError("Expected exactly one planned recovery stage")
    entry = entries[0]
    if not 0 < update < entry["budget"]["updates"]:
        raise ValueError("Recovery target must be an incomplete saved update")
    if (latest["update"] != update
            or latest["checkpoint"] != f"checkpoint-update-{update:08d}.pt"):
        raise ValueError("Restored checkpoint pointer/update differs from explicit target")
    return entry


class CurrentFrozenJsonResume(historical.TestFrozenJsonResume):
    """Keep all historical guards; update the explicit restored test boundary."""

    resume_stage = "query-d512-l4-T64"
    resume_update = 5500

    def test_adapter_reaches_real_training_boundary_with_recovered_checkpoint(self):
        output = historical.ROOT / self.resume_stage
        latest = json.loads((output / "latest.json").read_text())
        entry = verified_target(self.plan, self.resume_stage, self.resume_update, latest)
        with self.boundary(True) as sentinel:
            with self.assertRaises(historical.TrainingBoundaryReached):
                self.controller.train(self.args)
            sentinel.assert_called_once()
        kwargs = sentinel.call_args.kwargs
        self.assertEqual(kwargs["output"], output)
        self.assertEqual(kwargs["resume"], output / latest["checkpoint"])
        self.assertIsInstance(kwargs["model_config"].stem_channels, tuple)
        self.assertEqual(kwargs["model_config"].dim, entry["model"]["dim"])
        self.assertEqual(kwargs["model_config"].frames, entry["model"]["frames"])
        self.assertEqual(kwargs["budget"].payload(), entry["budget"])


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="query-d512-l4-T64")
    parser.add_argument("--update", type=int, default=5500)
    args = parser.parse_args()
    CurrentFrozenJsonResume.resume_stage = args.stage
    CurrentFrozenJsonResume.resume_update = args.update
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
                restored_stage=args.stage,
                restored_update=args.update,
                gpu_updates=0,
                controller_changed=False,
                historical_suite="test_resume_json_plan.py",
            )
        ),
        flush=True,
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
