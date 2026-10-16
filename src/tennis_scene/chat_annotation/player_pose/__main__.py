from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ..artifacts.configuration import artifact_path_resolver
from .selection import initialize
from .storage import read_json

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.player_pose",
    fields=(
        BoundaryPathField(
            "campaign",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "config",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
    ),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Court-free ball-store player pose campaign"
    )
    parser.add_argument(
        "command",
        choices=[
            "init",
            "orchestrate",
            "generate-clip",
            "pose-clip",
            "publish-clip",
            "review",
            "review-clip",
            "status",
        ],
    )
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    paths = {"campaign": args.campaign}
    if args.config is not None:
        paths["config"] = args.config
    checked = PATH_BOUNDARY.validate(
        paths,
        resolver=artifact_path_resolver(args.campaign.resolve()),
        independent_artifact_inputs=True,
    )
    args.campaign = checked.declared("campaign").path
    if args.config is not None:
        args.config = checked.declared("config").path
    if args.command != "init" and not args.campaign.is_dir():
        parser.error("--campaign must be an existing campaign directory")
    if args.command != "init" and args.config is not None:
        parser.error("--config is accepted only by init")
    if args.index is not None and args.index < 0:
        parser.error("--index must be nonnegative")
    if args.index is not None and args.command not in {
        "generate-clip",
        "pose-clip",
        "publish-clip",
        "review-clip",
    }:
        parser.error("--index is accepted only by per-clip commands")
    if args.command == "init":
        if args.config is None:
            parser.error("init requires --config")
        plan = initialize(args.campaign, read_json(args.config))
        print(
            json.dumps(
                {
                    "clips": len(plan["clips"]),
                    "selected_clips": plan["selected_clips"],
                    "selected_frames": plan["selected_frames"],
                    "threshold": plan["threshold"],
                },
                indent=2,
            )
        )
    elif args.command == "orchestrate":
        from .orchestrator import orchestrate

        orchestrate(args.campaign)
    elif args.command == "generate-clip":
        from .generation import generate_clip, record_failure

        if args.index is None:
            parser.error("generate-clip requires --index")
        try:
            generate_clip(args.campaign, args.index)
        except Exception:
            record_failure(args.campaign, args.index)
            raise
    elif args.command in ("pose-clip", "publish-clip"):
        from .publication import publish
        from .selected_pose import generate_selected_pose

        if args.index is None:
            parser.error(f"{args.command} requires --index")
        if args.command == "pose-clip":
            generate_selected_pose(args.campaign, args.index)
        publish(args.campaign, args.index)
    else:
        from .runner import review_clip, review_worker, status

        if args.command == "review":
            review_worker(args.campaign)
        elif args.command == "review-clip":
            if args.index is None:
                parser.error("review-clip requires --index")
            print(
                json.dumps(review_clip(args.campaign, args.index), ensure_ascii=False)
            )
        else:
            print(json.dumps(status(args.campaign), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
