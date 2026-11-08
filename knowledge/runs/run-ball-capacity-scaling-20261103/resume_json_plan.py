"""Explicit JSON-metadata adapter for the frozen #1675 campaign resume."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
from typing import Any

CONTROLLER = Path(__file__).resolve().with_name("run.py")
RUN_ID = "s42-20261103-main"
EFFECT = (
    "Only the imported controller.asdict binding uses dataclasses.asdict followed "
    "by a JSON roundtrip with allow_nan=False. Model objects, the training module, "
    "stdlib bindings, true hashes and all original guards remain in effect."
)


def json_asdict(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(value), allow_nan=False))


def load_controller():
    if Path.cwd().resolve() != CONTROLLER.parents[3]:
        raise ValueError("Run from the restored ball-capacity worktree")
    spec = importlib.util.spec_from_file_location(
        "capacity_frozen_json_resume", CONTROLLER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the original controller")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    return controller


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train"])
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--run-id", choices=[RUN_ID], required=True)
    parser.add_argument("--rclone-config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", required=True)
    args = parser.parse_args(argv)
    root = (
        args.workspace_root / "outputs/ball_detection/train/capacity_scaling" / RUN_ID
    )
    if args.benchmark.resolve() != (root / "benchmark.json").resolve():
        parser.error("Use the explicit restored campaign benchmark.json")
    if not (root / "plan.json").is_file():
        parser.error("An existing restored campaign plan is required")
    return args


def verify_frozen_controller(
    controller, plan, benchmark, path: Path = CONTROLLER
) -> str:
    actual = controller.dual_sha256(path)
    if actual != plan["controller_sha256"]:
        raise ValueError("Frozen controller SHA differs from the saved plan")
    if actual != benchmark["controller_sha256"]:
        raise ValueError("Frozen controller SHA differs from the saved benchmark")
    return actual


def provenance(controller, args: argparse.Namespace) -> dict[str, Any]:
    root = args.benchmark.parent
    plan = json.loads((root / "plan.json").read_text())
    benchmark = json.loads(args.benchmark.read_text())
    original_sha = verify_frozen_controller(controller, plan, benchmark)
    if controller.dual_sha256(args.benchmark) != plan["benchmark_sha256"]:
        raise ValueError("Restored benchmark SHA differs from the saved plan")
    return dict(
        schema="ball_capacity_json_representation_adapter.v1",
        adapter="knowledge/runs/run-ball-capacity-scaling-20261103/resume_json_plan.py",
        adapter_sha256=controller.dual_sha256(Path(__file__).resolve()),
        original_controller="knowledge/runs/run-ball-capacity-scaling-20261103/run.py",
        original_controller_sha256=original_sha,
        plan_sha256=controller.dual_sha256(root / "plan.json"),
        benchmark_sha256=controller.dual_sha256(args.benchmark),
        effect=EFFECT,
        action="train",
        resume=True,
        run_id=RUN_ID,
    )


def install_adapter(controller) -> None:
    if controller.asdict is not dataclasses.asdict:
        raise ValueError(
            "Unexpected controller asdict binding before adapter installation"
        )
    controller.asdict = json_asdict


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    controller = load_controller()
    record = provenance(controller, args)  # Verify both real hashes before binding.
    path = args.benchmark.parent / "recovery/attempt-3-runtime-provenance.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Existing adapter provenance differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    controller.save_json(path, record)
    install_adapter(controller)
    print(json.dumps(dict(event="json_metadata_adapter_active", **record)), flush=True)
    controller.train(args)


if __name__ == "__main__":
    main()
