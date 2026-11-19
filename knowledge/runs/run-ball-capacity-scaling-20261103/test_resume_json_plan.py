"""CPU checks using the restored campaign; never execute a training update."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = Path("/content/tennis-lab")
ROOT = WORKSPACE / "outputs/ball_detection/train/capacity_scaling/s42-20261103-main"
ARGV = [
    "train",
    "--workspace-root",
    str(WORKSPACE),
    "--run-id",
    "s42-20261103-main",
    "--rclone-config",
    "/tmp/ball-drive-mount.conf",
    "--benchmark",
    str(ROOT / "benchmark.json"),
    "--resume",
]


class TrainingBoundaryReached(BaseException):
    """Bypass the controller's failure persistence without touching live state."""


class TestFrozenJsonResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "resume_adapter_test", DIRECTORY / "resume_json_plan.py"
        )
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)
        cls.args = cls.adapter.parse_args(ARGV)
        cls.plan = json.loads((ROOT / "plan.json").read_text())
        cls.benchmark = json.loads((ROOT / "benchmark.json").read_text())

    def setUp(self):
        self.controller = self.adapter.load_controller()

    def boundary(self, adapted):
        if adapted:
            self.adapter.provenance(self.controller, self.args)
            self.adapter.install_adapter(self.controller)
        return patch.object(
            self.controller, "train_temporal_model", side_effect=TrainingBoundaryReached
        )

    def test_original_resume_guard_rejects_restored_json_arrays(self):
        with self.boundary(False) as sentinel:
            with self.assertRaisesRegex(ValueError, "unchanged original campaign plan"):
                self.controller.train(self.args)
            sentinel.assert_not_called()

    def test_adapter_reaches_real_training_boundary_with_recovered_checkpoint(self):
        with self.boundary(True) as sentinel:
            with self.assertRaises(TrainingBoundaryReached):
                self.controller.train(self.args)
            sentinel.assert_called_once()
        kwargs = sentinel.call_args.kwargs
        self.assertEqual(kwargs["output"], ROOT / "query-d512-l4-T64")
        self.assertEqual(
            kwargs["resume"], ROOT / "query-d512-l4-T64/checkpoint-update-00001500.pt"
        )
        self.assertIsInstance(kwargs["model_config"].stem_channels, tuple)
        self.assertEqual(kwargs["model_config"].dim, 512)
        self.assertEqual(kwargs["budget"].updates, 6000)
        self.assertEqual(kwargs["budget"].schedule_updates, 12000)

    def test_changed_budget_is_rejected_before_training(self):
        original = self.controller.budget

        def changed(*args, **kwargs):
            budget = original(*args, **kwargs)
            return dataclasses.replace(budget, updates=budget.updates + 1)

        with (
            self.boundary(True) as sentinel,
            patch.object(self.controller, "budget", changed),
        ):
            with self.assertRaisesRegex(ValueError, "unchanged original campaign plan"):
                self.controller.train(self.args)
            sentinel.assert_not_called()

    def test_changed_model_is_rejected_before_training(self):
        original = self.controller.configuration

        def changed(*args, **kwargs):
            config = original(*args, **kwargs)
            return dataclasses.replace(config, frames=config.frames * 2)

        with (
            self.boundary(True) as sentinel,
            patch.object(self.controller, "configuration", changed),
        ):
            with self.assertRaisesRegex(ValueError, "model/runtime differs"):
                self.controller.train(self.args)
            sentinel.assert_not_called()

    def test_changed_source_is_rejected_before_training(self):
        changed = copy.deepcopy(self.controller.source_identity())
        name = next(iter(changed["source_sha256"]))
        changed["source_sha256"][name] = "0" * 64
        with (
            self.boundary(True) as sentinel,
            patch.object(self.controller, "source_identity", return_value=changed),
        ):
            with self.assertRaisesRegex(ValueError, "Model source changed"):
                self.controller.train(self.args)
            sentinel.assert_not_called()

    def test_unrelated_controller_byte_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "run.py"
            changed.write_bytes(
                self.adapter.CONTROLLER.read_bytes() + b"\n# unrelated change\n"
            )
            with self.assertRaisesRegex(ValueError, "controller SHA differs"):
                self.adapter.verify_frozen_controller(
                    self.controller, self.plan, self.benchmark, changed
                )

    def test_real_controller_hash_must_match_both_saved_records(self):
        for changed_record in ("plan", "benchmark"):
            plan, benchmark = copy.deepcopy(self.plan), copy.deepcopy(self.benchmark)
            (plan if changed_record == "plan" else benchmark)["controller_sha256"] = (
                "0" * 64
            )
            with self.assertRaisesRegex(ValueError, "controller SHA differs"):
                self.adapter.verify_frozen_controller(self.controller, plan, benchmark)

    def test_serialization_binding_is_local_and_model_objects_are_intact(self):
        from src.tasks.ball_detection.training.temporal import runner

        config = self.controller.configuration("query", 512, 4)
        original = copy.deepcopy(config)
        bindings = (dataclasses.asdict, json.loads, json.dumps, runner.asdict)
        expected = dataclasses.asdict(config)
        self.adapter.provenance(self.controller, self.args)
        self.adapter.install_adapter(self.controller)
        converted = self.controller.asdict(config)
        self.assertEqual(config, original)
        self.assertIsInstance(config.stem_channels, tuple)
        self.assertIsInstance(converted["stem_channels"], list)
        self.assertEqual(converted, json.loads(json.dumps(expected, allow_nan=False)))
        self.assertEqual(
            bindings, (dataclasses.asdict, json.loads, json.dumps, runner.asdict)
        )
        self.assertIs(runner.asdict, dataclasses.asdict)
        self.assertEqual(runner.asdict(config), expected)

    def test_nonfinite_metadata_fails_explicitly(self):
        @dataclasses.dataclass
        class Metadata:
            measurement: float = float("nan")

        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            self.adapter.json_asdict(Metadata())

    def test_entrypoint_requires_existing_resume_and_explicit_restored_benchmark(self):
        invalid = [
            ["benchmark", *ARGV[1:]],
            [part for part in ARGV if part != "--resume"],
            [*ARGV[: ARGV.index("--benchmark")], "--resume"],
            [
                *ARGV[: ARGV.index("--run-id") + 1],
                "new-run",
                *ARGV[ARGV.index("--run-id") + 2 :],
            ],
            [
                *ARGV[: ARGV.index("--benchmark") + 1],
                "/tmp/other-benchmark.json",
                "--resume",
            ],
        ]
        for argv in invalid:
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.adapter.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
