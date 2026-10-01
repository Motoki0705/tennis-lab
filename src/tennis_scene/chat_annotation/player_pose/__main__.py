from __future__ import annotations

import argparse
import json
from pathlib import Path

from .selection import initialize
from .storage import read_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Court-free ball-store player pose campaign"
    )
    parser.add_argument(
        "command",
        choices=[
            "init",
            "orchestrate",
            "generate-clip",
            "review",
            "review-clip",
            "status",
        ],
    )
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if not args.campaign.is_absolute():
        parser.error("--campaign must be absolute")
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
