"""Extract, prepare, annotate or publish partial player association labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.player_association.pseudo_annotation.calibration import (
    prepare_calibration,
)
from src.tasks.player_association.pseudo_annotation.extract import extract, load_plan
from src.tasks.player_association.pseudo_annotation.frozen_run import FrozenRun
from src.tasks.player_association.pseudo_annotation.prepare import prepare_baseline
from src.tasks.player_association.pseudo_annotation.runner import (
    publish,
    run_annotation,
)
from src.tasks.player_association.pseudo_annotation.sampling import STRATA
from src.utils.io import write_json_atomic

_CONFIG = Path(__file__).resolve().parents[1] / "configs/pseudo_annotation_meiji.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="action", required=True)
    plan = command.add_parser("extract")
    plan.add_argument("snapshot", type=Path)
    plan.add_argument("destination", type=Path)
    plan.add_argument("--config", type=Path, default=_CONFIG)
    prepare = command.add_parser("prepare")
    prepare.add_argument("snapshot", type=Path)
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("clip_id")
    prepare.add_argument("packet", type=Path)
    subset = prepare.add_mutually_exclusive_group()
    subset.add_argument("--frames", type=int, nargs="+")
    subset.add_argument("--stratum", choices=STRATA)
    prepare.add_argument("--show-prior-association", action="store_true")
    calibration = command.add_parser("prepare-calibration")
    calibration.add_argument("dataset", type=Path)
    calibration.add_argument("clip_id")
    calibration.add_argument("packet", type=Path)
    calibration.add_argument("--frames", type=int, nargs="+", required=True)
    run = command.add_parser("run")
    run.add_argument("packet", type=Path)
    run.add_argument("run_dir", type=Path)
    store = command.add_parser("publish")
    store.add_argument("packet", type=Path)
    store.add_argument("run_dir", type=Path)
    store.add_argument("dataset", type=Path)
    args = parser.parse_args()
    if args.action == "extract":
        result = extract(FrozenRun(args.snapshot), args.config)
        write_json_atomic(args.destination, result)
        print(json.dumps(result["summary"], indent=2))
    elif args.action == "prepare":
        frozen = FrozenRun(args.snapshot)
        plan_data = load_plan(args.plan, frozen)
        scope = plan_data["clips"][args.clip_id]
        frames = (
            args.frames
            if args.frames is not None
            else (
                scope["sampled"][args.stratum]
                if args.stratum
                else scope["sampled_frames"]
            )
        )
        prepare_baseline(
            frozen,
            plan_data,
            args.clip_id,
            frames,
            args.packet,
            show_prior_association=args.show_prior_association,
        )
    elif args.action == "prepare-calibration":
        prepare_calibration(args.dataset, args.clip_id, args.frames, args.packet)
    elif args.action == "run":
        receipt = run_annotation(args.packet, args.run_dir)
        print(
            json.dumps(
                {key: value for key, value in receipt.items() if key != "command"},
                indent=2,
            )
        )
    else:
        print(publish(args.packet, args.run_dir, args.dataset))


if __name__ == "__main__":
    main()
