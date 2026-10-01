"""Explicit annotation edits and validated interpolation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from src.tennis_scene.chat_annotation.runtime.contracts import (
    BallAnnotation,
    PlayerAnnotation,
    loads_json,
    make_template,
    parse_annotation,
)
from src.tennis_scene.chat_annotation.runtime.geometry import interpolate_ball

from .common import (
    atomic_write_json,
    compact_ranges,
    load_annotation,
    utc_now,
    verified_timeline,
)
from .worker_candidates import read_cache
from .worker_context import (
    BALL_ROW_KEYS,
    PLAYER_ROW_KEYS,
    Ctx,
    brief,
    dump,
    parse_frames,
    summarize,
    write_annotation,
)


def cmd_info(ctx: Ctx, args: argparse.Namespace) -> int:
    frames = ctx.manifest.frames
    targets = [f.frame_index for f in frames if f.is_target]
    from fractions import Fraction

    duration = float(
        (frames[-1].clip_pts + frames[-1].duration_pts)
        * Fraction(ctx.manifest.time_base)
    )
    dump(
        {
            "task_id": ctx.task["task_id"],
            "attempt": ctx.task["attempt"],
            "target": ctx.target,
            "clip_id": ctx.task["clip_id"],
            "video": str(ctx.video),
            "manifest": str(ctx.manifest_path),
            "annotation": str(ctx.annotation_path),
            "frames": ctx.n,
            "size": [ctx.w, ctx.h],
            "nominal_fps": ctx.manifest.nominal_fps,
            "duration_seconds": round(duration, 3),
            "owned_range_is_target": compact_ranges(targets),
            "note": "All frames (including context frames) must be annotated.",
            "ball_max_gap_seconds": ctx.manifest.policies.ball_max_gap_seconds,
            "source_title": ctx.manifest.source.title,
            "previous_annotation": ctx.task.get("previous_annotation"),
        }
    )
    return 0


def cmd_init(ctx: Ctx, args: argparse.Namespace) -> int:
    created = False
    if not ctx.annotation_path.exists():
        previous = ctx.task.get("previous_annotation")
        if previous:
            annotation = load_annotation(Path(previous))
            expected = BallAnnotation if ctx.target == "ball" else PlayerAnnotation
            if not isinstance(annotation, expected):
                raise ValueError("previous annotation has a different target schema")
            if (
                annotation.clip_id != ctx.task["clip_id"]
                or annotation.frame_count != ctx.n
            ):
                raise ValueError("previous annotation belongs to another clip")
            unreview = ctx.task.get("unreview_frames", [])
            if unreview:
                data = annotation.model_dump(mode="json")
                for index in unreview:
                    if not 0 <= index < ctx.n:
                        raise ValueError("revision frame is outside this clip")
                    data["frames"][index]["reviewed"] = False
                    data["frames"][index]["notes"] = "親の指摘による再確認待ち"
                data["status"] = "partial"
                annotation = parse_annotation(data)
            write_annotation(ctx, annotation)
        else:
            verified_timeline(ctx.video, ctx.manifest, ctx.work)
            template = make_template(ctx.manifest, ctx.target)
            write_annotation(ctx, template)
        created = True
    summary = summarize(ctx.annotation(), ctx.manifest)
    dump({"annotation": str(ctx.annotation_path), "created": created, **summary})
    return 0


def cmd_status(ctx: Ctx, args: argparse.Namespace) -> int:
    summary = summarize(ctx.annotation(), ctx.manifest)
    dump(summary)
    return 1 if summary["error_count"] else 0


def cmd_apply(ctx: Ctx, args: argparse.Namespace) -> int:
    edits = loads_json(Path(args.edits).read_text(encoding="utf-8"))
    if not isinstance(edits, dict):
        raise ValueError("edits file must be a JSON object")
    unknown = set(edits) - {"frames", "ranges", "status", "issues"}
    if unknown:
        raise ValueError(f"unknown top-level edit keys: {sorted(unknown)}")
    data = loads_json(ctx.annotation_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = data["frames"]
    allowed = BALL_ROW_KEYS if ctx.target == "ball" else PLAYER_ROW_KEYS
    touched: list[int] = []

    def update(index: int, row: dict[str, Any]) -> None:
        if not 0 <= index < ctx.n:
            raise ValueError(f"frame {index} outside clip [0,{ctx.n})")
        extra = set(row) - allowed
        if extra:
            raise ValueError(
                f"frame {index}: keys {sorted(extra)} not allowed for {ctx.target}"
            )
        if rows[index]["frame_index"] != index:
            raise ValueError("annotation frames are not in display order")
        rows[index].update(copy.deepcopy(row))
        touched.append(index)

    for key, row in (edits.get("frames") or {}).items():
        update(int(key), row)
    for item in edits.get("ranges") or []:
        start, stop = int(item["start"]), int(item["stop"])
        if not 0 <= start < stop <= ctx.n:
            raise ValueError(f"bad range [{start},{stop})")
        for index in range(start, stop):
            update(index, item["set"])
    if "status" in edits:
        data["status"] = edits["status"]
    if "issues" in edits:
        data["issues"] = edits["issues"]
    annotation = parse_annotation(data)
    summary = summarize(annotation, ctx.manifest)
    if args.dry_run:
        dump({"written": False, "touched": brief(touched), **summary})
        return 0
    shutil.copy2(ctx.annotation_path, ctx.work / "annotation.prev.json")
    write_annotation(ctx, annotation)
    with (ctx.work / "edits.log.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "at": utc_now(),
                    "edits": str(args.edits),
                    "touched": compact_ranges(touched),
                }
            )
            + "\n"
        )
    dump(
        {
            "written": True,
            "touched": brief(touched),
            "validation_status": summary["validation_status"],
            "error_count": summary["error_count"],
            "errors": summary["errors"],
            "reviewed": summary["reviewed"],
            "frames": summary["frames"],
        }
    )
    return 0


def cmd_accept(ctx: Ctx, args: argparse.Namespace) -> int:
    """Write visually verified candidate centers as reviewed visible rows (ball only)."""
    if ctx.target != "ball":
        raise ValueError("accept is for ball tasks")
    cands = read_cache(ctx.work / "cands_ball.json")["frames"]
    frames = parse_frames(args.frames, ctx.n)
    ranks = (
        {
            int(k): int(v)
            for k, v in (item.split("=") for item in args.rank.split(",") if item)
        }
        if args.rank
        else {}
    )
    missing = []
    rows: dict[str, Any] = {}
    for index in frames:
        entry = cands.get(str(index), {}).get("candidates", [])
        rank = ranks.get(index, 0)
        if rank >= len(entry):
            missing.append(index)
            continue
        if args.use == "blob":
            blob = entry[rank].get("blob")
            if not blob:
                missing.append(index)
                continue
            x, y = blob["center_px"]
        else:
            x, y = entry[rank]["center_px"]
        rows[str(index)] = {
            "reviewed": True,
            "balls": [
                {
                    "track_id": args.track,
                    "center_px": [round(x, 1), round(y, 1)],
                    "status": "visible",
                    "interpolation_frames": None,
                }
            ],
            "notes": "",
        }
    if missing:
        raise ValueError(
            f"no {'blob center' if args.use == 'blob' else 'candidate'} of the requested rank for frames "
            f"{compact_ranges(missing)}; nothing written (handle those frames by eye)"
        )
    edits_path = ctx.work / f"accept_{int(time.time() * 1000)}.json"
    atomic_write_json(edits_path, {"frames": rows})
    args.edits = str(edits_path)
    return cmd_apply(ctx, args)


def cmd_check(ctx: Ctx, args: argparse.Namespace) -> int:
    """List frames worth a targeted visual re-check (instead of re-viewing every frame)."""
    annotation = ctx.annotation()
    if not isinstance(annotation, BallAnnotation):
        raise ValueError("check is for ball annotations")
    from fractions import Fraction

    def t(i: int) -> float:
        return float(ctx.manifest.frames[i].clip_pts * Fraction(ctx.manifest.time_base))

    rows = annotation.frames
    breaks = {f.frame_index for f in rows if f.interpolation_break}
    reasons: dict[int, list[str]] = {}

    def flag(i: int, why: str) -> None:
        reasons.setdefault(i, []).append(why)

    points: dict[str, dict[int, tuple[float, float]]] = {}
    for f in rows:
        for b in f.balls:
            if b.center_px is not None:
                points.setdefault(b.track_id, {})[f.frame_index] = (
                    b.center_px[0],
                    b.center_px[1],
                )
    diag = math.hypot(ctx.w, ctx.h)
    for _track, pts in points.items():
        keys = sorted(pts)
        for a, b in zip(keys, keys[1:], strict=False):
            if b - a == 1 and a not in breaks and b not in breaks:
                speed = math.dist(pts[a], pts[b]) / max(t(b) - t(a), 1e-6)
                if speed > args.max_speed_frac * diag:
                    flag(b, f"jump {math.dist(pts[a], pts[b]):.0f}px from f{a}")
        for a, b, c in zip(keys, keys[1:], keys[2:], strict=False):
            if c - a == 2 and not ({a, b, c} & breaks):
                second = math.hypot(
                    pts[a][0] - 2 * pts[b][0] + pts[c][0],
                    pts[a][1] - 2 * pts[b][1] + pts[c][1],
                )
                if second > args.max_second_diff:
                    flag(b, f"kink {second:.0f}px")
        for k in keys:
            if k - 1 not in pts and k + 1 not in pts:
                flag(k, "isolated point")
    for f in rows:
        if f.interpolation_break and any(b.center_px is not None for b in f.balls):
            flag(f.frame_index, "event frame")
        if len({b.track_id for b in f.balls}) > 1:
            flag(f.frame_index, "several balls")
    unreviewed = [f.frame_index for f in rows if not f.reviewed]
    listed = sorted(reasons)[: args.limit]
    dump(
        {
            "suspicious": len(reasons),
            "listed": [{"f": i, "why": ", ".join(reasons[i])} for i in listed],
            "frames_arg": ",".join(str(i) for i in listed),
            "unreviewed": compact_ranges(unreviewed)[:20],
            "next": "view them: ct crops <dir> --source annotation --frames <frames_arg> --size 96 --scale 2",
        }
    )
    return 0


def cmd_interpolate(ctx: Ctx, args: argparse.Namespace) -> int:
    annotation = ctx.annotation()
    if not isinstance(annotation, BallAnnotation):
        raise ValueError("interpolation is only for ball annotations")
    result = interpolate_ball(
        annotation, ctx.manifest, args.track_id, args.start, args.stop
    )
    shutil.copy2(ctx.annotation_path, ctx.work / "annotation.prev.json")
    write_annotation(ctx, result)
    summary = summarize(result, ctx.manifest)
    dump(
        {
            "interpolated": [args.start + 1, args.stop],
            "error_count": summary["error_count"],
            "errors": summary["errors"],
        }
    )
    return 0
