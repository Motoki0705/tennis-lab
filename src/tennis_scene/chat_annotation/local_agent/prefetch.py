"""Prefetch shared per-clip caches ahead of the workers (one process, one model load).

For clips that are running or next in the dispatcher order it writes:
  cache/timeline/<video_sha256>.json   verified timeline (workers skip the full-clip decode)
  cache/cands_ball/<clip_id>.json      ball-model candidates + blob (streak-middle) centers
Workers copy these with `ct cands-ball` / reuse them in every frame read. Search guides only.
Single instance via flock (.prefetch.lock). Use the package's ``prefetch`` command;
CUDA runs require the shared training queue.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from .campaign_state import next_candidates, read_control, read_state
from .common import load_manifest, locate_video, utc_now, verified_timeline
from .configuration import paths
from .ct import BallModel, compute_ball_candidates, read_cache, refine_candidates
from .path_contracts import campaign_resolver, validate_command_paths

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.local_agent",
    fields=(BoundaryPathField("campaign", PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                              must_exist=True, allow_role_root=True),),
)



def log(message: str) -> None:
    with (paths().logs / "prefetch.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")


def upcoming(lookahead: int) -> list[tuple[str, Path]]:
    state = read_state()
    control = read_control()
    # Running tasks only if their worker has not produced candidates itself yet.
    running = [
        t
        for t in state["tasks"].values()
        if t["status"] == "running"
        and not (Path(t["attempts"][-1]["dir"]) / "work" / "cands_ball.json").exists()
    ]
    queued = [
        state["tasks"][tid]
        for tid in next_candidates(state, {**control, "mode": "run"})
    ]
    seen: set[str] = set()
    result: list[tuple[str, Path]] = []
    for task in running + queued:
        if task["clip_id"] not in seen:
            seen.add(task["clip_id"])
            result.append((task["clip_id"], Path(task["manifest"])))
    return result[:lookahead] if lookahead > 0 else result


def complete(clip_id: str, frames: int) -> bool:
    path = (paths().campaign_dir / "cache" / "cands_ball") / f"{clip_id}.json"
    if not path.exists():
        return False
    cache = read_cache(path)
    if len(cache["frames"]) < frames:
        return False
    return all(
        "blob" in c for f in cache["frames"].values() for c in f["candidates"][:2]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("PREFETCH_THREADS", "4"))
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=40,
        help="clips ahead of the dispatcher; 0 = all",
    )
    parser.add_argument(
        "--exit-when-idle",
        action="store_true",
        help="exit once every upcoming clip is cached",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0,
        help="exit after the current clip past this; 0 = no limit",
    )
    args = parser.parse_args(argv)
    PATH_BOUNDARY.validate({"campaign": paths().campaign_dir}, resolver=campaign_resolver(paths()))
    validate_command_paths()
    if args.device == "cuda" and (
        not os.environ.get("TENNIS_RUN_ID")
        or os.environ.get("TENNIS_GPU_RESOURCE") not in ("half", "all")
    ):
        raise ValueError(
            "CUDA prefetch must run through the shared training queue (TENNIS_RUN_ID/TENNIS_GPU_RESOURCE required)"
        )
    paths().logs.mkdir(parents=True, exist_ok=True)
    (paths().campaign_dir / "cache" / "cands_ball").mkdir(parents=True, exist_ok=True)
    lock = (paths().campaign_dir / ".prefetch.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("prefetch already running", file=sys.stderr)
        return 1
    (paths().logs / "prefetch.pid").write_text(f"{os.getpid()}\n")
    model = BallModel(threads=args.threads, device=args.device)
    log(
        f"START pid={os.getpid()} device={args.device} threads={args.threads} lookahead={args.lookahead} "
        f"exit_when_idle={args.exit_when_idle} max_minutes={args.max_minutes}"
    )
    began_run = time.monotonic()
    done: set[str] = set()  # cached during this run; their cache files are not re-read
    failed: set[str] = (
        set()
    )  # logged once and skipped by this run (workers then compute locally)
    while True:
        if read_control().get("mode") == "stop":
            log("EXIT mode=stop")
            return 0
        if args.max_minutes and time.monotonic() - began_run > args.max_minutes * 60:
            log(f"EXIT max_minutes reached done={len(done)} failed={len(failed)}")
            return 0
        worked = False
        for clip_id, manifest_path in upcoming(args.lookahead):
            if clip_id in done or clip_id in failed:
                continue
            try:
                manifest = load_manifest(manifest_path)
                n = len(manifest.frames)
                if complete(clip_id, n):
                    done.add(clip_id)
                    continue
                video = locate_video(manifest)
                began = time.monotonic()
                verified_timeline(video, manifest)
                path = (
                    paths().campaign_dir / "cache" / "cands_ball"
                ) / f"{clip_id}.json"
                compute_ball_candidates(
                    model, video, manifest, None, path, 0, n, 0.05, 3, max_seconds=1e9
                )
                refine_candidates(video, manifest, None, path, 0, n, 2)
                log(
                    f"DONE {clip_id} frames={n} seconds={time.monotonic() - began:.0f} device={args.device}"
                )
                done.add(clip_id)
            except Exception:
                failed.add(clip_id)
                log(f"ERROR {clip_id} {traceback.format_exc()[-800:]!r}")
            worked = True
            break  # re-read the queue after every clip so the order stays current
        if not worked:
            if args.exit_when_idle:
                log(
                    f"EXIT idle: every upcoming clip is cached done={len(done)} failed={len(failed)}"
                )
                return 1 if failed else 0
            time.sleep(60)
