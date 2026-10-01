"""Worker CLI; implementation is split into context, edits, images, candidates and session tools."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from .configuration import paths
from .path_contracts import campaign_resolver, validate_command_paths
from .worker_candidates import BallModel as BallModel
from .worker_candidates import ball_identity as ball_identity
from .worker_candidates import blob_center as blob_center
from .worker_candidates import cmd_cands_ball as cmd_cands_ball
from .worker_candidates import cmd_refine_ball as cmd_refine_ball
from .worker_candidates import compute_ball_candidates as compute_ball_candidates
from .worker_candidates import read_cache as read_cache
from .worker_candidates import refine_candidates as refine_candidates
from .worker_context import Ctx as Ctx
from .worker_context import parse_frames as parse_frames
from .worker_edits import cmd_accept as cmd_accept
from .worker_edits import cmd_apply as cmd_apply
from .worker_edits import cmd_check as cmd_check
from .worker_edits import cmd_info as cmd_info
from .worker_edits import cmd_init as cmd_init
from .worker_edits import cmd_interpolate as cmd_interpolate
from .worker_edits import cmd_status as cmd_status
from .worker_images import cmd_crops as cmd_crops
from .worker_images import cmd_frames as cmd_frames
from .worker_session import cmd_context as cmd_context
from .worker_session import cmd_finish as cmd_finish

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.local_agent",
    fields=(BoundaryPathField("campaign", PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                              must_exist=True, allow_role_root=True),),
)



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("attempt_dir", type=Path)
        return p

    add("info", "task, clip and timing facts")
    add("init", "create the annotation (template or previous attempt copy); idempotent")
    add("status", "validate and summarize the current annotation")
    p = add("apply", "merge frame rows from an edits JSON (see WORKER.md)")
    p.add_argument("--edits", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = add("accept", "write verified candidate centers as reviewed visible ball rows")
    p.add_argument(
        "--frames",
        required=True,
        help="half-open ranges 'S:E' and single indices, comma separated",
    )
    p.add_argument("--track", required=True)
    p.add_argument(
        "--rank",
        default="",
        help="per-frame candidate rank overrides 'F=R,...' (default rank 0)",
    )
    p.add_argument(
        "--use",
        default="blob",
        choices=["blob", "raw"],
        help="blob = streak-middle center from refine-ball (default); raw = model peak",
    )
    p.add_argument("--dry-run", action="store_true")
    p = add(
        "interpolate", "fill reviewed null ball frames between two visible endpoints"
    )
    p.add_argument("--track-id", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p = add("frames", "contact sheet of consecutive frames (full frame or fixed crop)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--scale", type=float)
    p.add_argument("--cols", type=int)
    p.add_argument(
        "--ruler", type=int, default=0, help="tick spacing in original px (0=off)"
    )
    p.add_argument(
        "--draw", default="none", choices=["none", "annotation", "cands-ball"]
    )
    p.add_argument("--name")
    p = add(
        "crops", "one tile per point: annotation objects, candidates or given points"
    )
    p.add_argument(
        "--source", default="annotation", choices=["annotation", "cands-ball"]
    )
    points = p.add_mutually_exclusive_group()
    points.add_argument(
        "--points",
        help="Inline JSON: {frame: [x,y]} or [{frame,x,y,label}]",
    )
    points.add_argument('--points-file', type=Path, help='Absolute point JSON file inside the campaign')
    p.add_argument("--track")
    p.add_argument(
        "--top", type=int, default=1, help="candidates per frame for cands-ball"
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--size", help="crop size in original px, N or WxH")
    p.add_argument("--scale", type=float)
    p.add_argument("--cols", type=int)
    p.add_argument("--ruler", type=int, default=0)
    p.add_argument(
        "--draw",
        default=None,
        choices=["none", "annotation", "cands-ball"],
        help="default: annotation for --source annotation, otherwise none",
    )
    p.add_argument(
        "--mark",
        action="store_true",
        help="edge ticks pointing at each point (draw=none)",
    )
    p.add_argument(
        "--frames",
        help="only these frames: 'S:E' ranges and single indices, comma separated",
    )
    p.add_argument("--name")
    p = add("cands-ball", "CPU ball-model proposals (search guides)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--max-peaks", type=int, default=3)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=50,
        help="return after about this long (long calls get killed by the exec tool); rerun while rerun_to_continue",
    )
    p = add("refine-ball", "add streak-middle blob centers to cached ball candidates")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--top", type=int, default=2)
    p = add(
        "check",
        "frames worth a targeted visual re-check: jumps, kinks, isolated points, events",
    )
    p.add_argument(
        "--max-speed-frac",
        type=float,
        default=2.0,
        help="flag a 1-frame jump faster than this x image diagonal per second",
    )
    p.add_argument("--max-second-diff", type=float, default=25.0)
    p.add_argument("--limit", type=int, default=80)
    add("context", "latest-request context occupancy of this Codex session")
    p = add("finish", "final checks; writes result.json and prints the final message")
    p.add_argument(
        "--outcome",
        required=True,
        choices=["completed", "partial", "needs_continuation"],
    )
    p.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    PATH_BOUNDARY.validate({"campaign": paths().campaign_dir}, resolver=campaign_resolver(paths()))
    args.attempt_dir = validate_command_paths(attempt=args.attempt_dir)['attempt']
    if args.command == 'apply':
        args.edits = str(validate_command_paths(edits=args.edits)['edits'])
    if args.command in {'frames', 'crops'} and (
        args.step < 1 or args.ruler < 0 or (args.cols is not None and args.cols < 1)
        or (args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0))
    ):
        raise ValueError('Image step/columns/scale must be positive and ruler nonnegative')
    if args.command == 'crops' and args.points_file is not None:
        args.points_file = validate_command_paths(edits=args.points_file)['edits']
    cv2.setNumThreads(2)
    handler = {
        "info": cmd_info,
        "init": cmd_init,
        "status": cmd_status,
        "apply": cmd_apply,
        "interpolate": cmd_interpolate,
        "frames": cmd_frames,
        "crops": cmd_crops,
        "accept": cmd_accept,
        "cands-ball": cmd_cands_ball,
        "refine-ball": cmd_refine_ball,
        "context": cmd_context,
        "finish": cmd_finish,
        "check": cmd_check,
    }[args.command]
    try:
        ctx = Ctx(args.attempt_dir)
        result: int = handler(ctx, args)
        return result
    except Exception as error:  # report compactly; the worker decides what to do
        message = str(error)
        if len(message) > 3000:
            message = message[:3000] + " ..."
        print(f"failed: {type(error).__name__}: {message}")
        return 1
