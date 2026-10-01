"""Worker CLI; implementation is split into context, edits, images, candidates and session tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

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
    p.add_argument(
        "--points",
        help="JSON file or inline JSON: {frame: [x,y]} or [{frame,x,y,label}]",
    )
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
        return handler(ctx, args)
    except Exception as error:  # report compactly; the worker decides what to do
        message = str(error)
        if len(message) > 3000:
            message = message[:3000] + " ..."
        print(f"failed: {type(error).__name__}: {message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
