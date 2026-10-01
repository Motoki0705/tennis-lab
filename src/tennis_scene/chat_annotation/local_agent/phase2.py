"""Phase 2: re-annotate the weakest processed ball annotations from scratch (user decisions 2026-09-30).

  phase2.py rank [--show 30]          # metrics for every processed ball annotation -> logs/phase2_rank.json
  phase2.py enqueue --top N           # add "<clip>__ball__p2" tasks (from scratch) in rank order
  phase2.py compare [<p2_task_id> ...] [--video]   # old (processed) vs new (held): metrics, centre agreement, overlay video
  phase2.py disagreements <p2_task_id> ...         # rule condition 4: frames where only one side has a ball, by run
  phase2.py triage                                 # rule over all held p2 tasks -> replace / keep_old / eyeball lists

Ranking (worst first) uses automatic metrics only, as decided:
  1. unreviewed frames (should be 0 after validation, kept for older sources)
  2. unresolved share of ball-bearing frames
  3. trajectory anomalies (speed + isolated flags, each capped at 30 by qa.ball_metrics) per 100 ball frames
New annotations never replace processed ones automatically: intake holds them and the user approves
each replacement from the comparison (PROCESS_RAW step 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .campaign_state import channel_map, locked_state, log_event, read_state
from .common import load_annotation, load_manifest, manifest_index
from .configuration import file_sha256, paths
from .qa import ball_metrics


def source_of(clip_id: str, campaign_clips: set[str]) -> str:
    return "cli_campaign" if clip_id in campaign_clips else "earlier_route"


def cmd_rank(args: argparse.Namespace) -> int:
    manifests = manifest_index()
    channels = channel_map()
    state = read_state()
    campaign_clips = {
        t["clip_id"] for t in state["tasks"].values() if t.get("phase", 1) == 1
    }
    rows: list[dict[str, Any]] = []
    for path in sorted((paths().annotated / "processed" / "ball").glob("*.json")):
        clip_id = path.stem
        if clip_id not in manifests:
            rows.append({"clip_id": clip_id, "error": "no manifest"})
            continue
        m = ball_metrics(load_annotation(path), load_manifest(manifests[clip_id]))
        ball = max(1, m["frames_with_ball"])
        rows.append(
            {
                "clip_id": clip_id,
                "channel": channels.get(clip_id.split("__")[0], "?"),
                "source": source_of(clip_id, campaign_clips),
                "frames": m["frames"],
                "ball_frames": m["frames_with_ball"],
                "unreviewed": m["frames"] - m["reviewed"],
                "unresolved_share": round(m["unresolved_ratio_of_ball_frames"], 4),
                "anomalies_per100": round(
                    (len(m["speed_flags"]) + len(m["isolated_points"])) / ball * 100, 2
                ),
            }
        )
    ok = [r for r in rows if "error" not in r]
    ok.sort(
        key=lambda r: (-r["unreviewed"], -r["unresolved_share"], -r["anomalies_per100"])
    )
    for rank, row in enumerate(ok, 1):
        row["rank"] = rank
    (paths().logs / "phase2_rank.json").write_text(
        json.dumps(
            {"rows": ok, "errors": [r for r in rows if "error" in r]},
            indent=1,
            ensure_ascii=False,
        )
    )
    print(
        f"{'rank':>4} {'clip':44s} {'channel':14s} {'source':13s} {'ballF':>5} {'unrev':>5} {'unres':>6} {'anom':>5}"
    )
    for row in ok[: args.show]:
        print(
            f"{row['rank']:4d} {row['clip_id'][:44]:44s} {row['channel'][:14]:14s} {row['source']:13s} "
            f"{row['ball_frames']:5d} {row['unreviewed']:5d} {row['unresolved_share']:6.3f} {row['anomalies_per100']:5.1f}"
        )
    print("errors:", len(rows) - len(ok), "| ranked:", len(ok))
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    ranked = json.loads(
        (paths().logs / "phase2_rank.json").read_text(encoding="utf-8")
    )["rows"][: args.top]
    manifests = manifest_index()
    added = []
    with locked_state() as state:
        for row in ranked:
            task_id = f"{row['clip_id']}__ball__p2"
            if task_id in state["tasks"]:
                continue
            state["tasks"][task_id] = {
                "clip_id": row["clip_id"],
                "target": "ball",
                "manifest": str(manifests[row["clip_id"]]),
                "status": "pending",
                "attempts": [],
                "phase": 2,
                "rank": row["rank"],
                "phase2_reason": {
                    k: row[k]
                    for k in (
                        "unresolved_share",
                        "anomalies_per100",
                        "unreviewed",
                        "source",
                    )
                },
            }
            added.append(task_id)
    if added:
        log_event(
            "PHASE2_ENQUEUE",
            f"{len(added)} tasks, ranks {ranked[0]['rank']}..{ranked[-1]['rank']}",
        )
    print(json.dumps({"added": len(added)}))
    return 0


def centres(
    annotation: Any,
) -> dict[int, list[tuple[str, str, tuple[float, float] | None]]]:
    return {
        f.frame_index: [
            (
                b.track_id,
                b.status,
                tuple(b.center_px) if b.center_px is not None else None,
            )
            for b in f.balls
        ]
        for f in annotation.frames
    }


def compare_pair(old: Any, new: Any, manifest: Any) -> dict[str, Any]:
    """Old vs new on the same clip: per-status counts, centre agreement, ball presence disagreement."""
    import math
    import statistics

    mo, mn = ball_metrics(old, manifest), ball_metrics(new, manifest)
    co, cn = centres(old), centres(new)
    dists, only_old, only_new = [], [], []
    for index in co:
        a = [c for _, _, c in co[index] if c is not None]
        b = [c for _, _, c in cn.get(index, []) if c is not None]
        if a and b:
            if len(a) == len(b) == 1:
                dists.append(math.dist(a[0], b[0]))
            else:
                from scipy.optimize import linear_sum_assignment

                costs = [[math.dist(x, y) for y in b] for x in a]
                rows, columns = linear_sum_assignment(costs)
                dists.extend(
                    costs[int(row)][int(column)]
                    for row, column in zip(rows, columns, strict=True)
                )
        if co[index] and not cn.get(index):
            only_old.append(index)
        elif cn.get(index) and not co[index]:
            only_new.append(index)

    def summary(m: dict[str, Any]) -> dict[str, Any]:
        return {
            "unreviewed": m["frames"] - m["reviewed"],
            "ball_frames": m["frames_with_ball"],
            "status": m["status_count"],
            "unresolved_share": m["unresolved_ratio_of_ball_frames"],
            "speed_flags": len(m["speed_flags"]),
            "isolated": len(m["isolated_points"]),
        }

    return {
        "old": summary(mo),
        "new": summary(mn),
        "both_centred_frames": sum(
            bool([c for _, _, c in co[i] if c is not None])
            and bool([c for _, _, c in cn.get(i, []) if c is not None])
            for i in co
        ),
        "matched_centres": len(dists),
        "count_disagreement_frames": [
            i for i in co if co[i] and cn.get(i) and len(co[i]) != len(cn[i])
        ],
        "centre_distance_px": {
            "median": statistics.median(dists) if dists else None,
            "p90": round(sorted(dists)[int(0.9 * (len(dists) - 1))], 1)
            if dists
            else None,
            "over_5px": sum(d > 5 for d in dists),
        },
        "ball_only_in_old": len(only_old),
        "ball_only_in_new": len(only_new),
        "only_old_frames": only_old[:40],
        "only_new_frames": only_new[:40],
    }


def compare_video(
    old: Any, new: Any, manifest: Any, out: Path, scale: float = 0.5
) -> None:
    """One video, both annotations: old = cyan ring (left label), new = yellow ring (right label)."""
    from fractions import Fraction

    import av
    import cv2

    from .common import iter_frames, locate_video

    co, cn = centres(old), centres(new)
    width = int(round(manifest.width * scale / 2) * 2)
    height = int(round(manifest.height * scale / 2) * 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(manifest.nominal_fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for index, image in iter_frames(
            locate_video(manifest), manifest, 0, len(manifest.frames)
        ):
            canvas = image.copy()
            for items, color, dx in (
                (co.get(index, []), (255, 255, 0), -1),
                (cn.get(index, []), (0, 255, 255), 1),
            ):
                for _track, _status, c in items:
                    if c is None:
                        continue
                    p = (int(round(c[0])), int(round(c[1])))
                    cv2.circle(canvas, p, 16 if dx < 0 else 11, color, 2, cv2.LINE_AA)
            old_s = ",".join(f"{t}:{s[:3]}" for t, s, _ in co.get(index, [])) or "-"
            new_s = ",".join(f"{t}:{s[:3]}" for t, s, _ in cn.get(index, [])) or "-"
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 76), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                f"f{index}  OLD(cyan,large) {old_s}",
                (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"      NEW(yellow,small) {new_s}",
                (8, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frame = av.VideoFrame.from_ndarray(
                cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA),
                format="bgr24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def cmd_compare(args: argparse.Namespace) -> int:
    state = read_state()
    out_rows = []
    task_ids = args.task_ids or [
        tid
        for tid, t in state["tasks"].items()
        if t.get("phase") == 2 and t["status"] in ("held", "review")
    ]
    for task_id in task_ids:
        task = state["tasks"][task_id]
        attempt = Path(task["attempts"][-1]["dir"])
        manifest = load_manifest(Path(task["manifest"]))
        old = load_annotation(
            paths().annotated / "processed" / "ball" / f"{task['clip_id']}.json"
        )
        new = load_annotation(attempt / f"annotation_{task['clip_id']}.json")
        report = {
            "task_id": task_id,
            "clip_id": task["clip_id"],
            "rank": task.get("rank"),
            "old_sha256": file_sha256(
                paths().annotated / "processed" / "ball" / f"{task['clip_id']}.json"
            ),
            "new_sha256": file_sha256(attempt / f"annotation_{task['clip_id']}.json"),
            "manifest_sha256": file_sha256(Path(task["manifest"])),
            **compare_pair(old, new, manifest),
        }
        target = paths().campaign_dir / "qa" / "phase2" / task["clip_id"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "compare.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False)
        )
        if args.video:
            compare_video(old, new, manifest, target / "compare.mp4")
        out_rows.append(report)
        o, n = report["old"], report["new"]
        print(
            f"{task_id[:44]:44s} unrev {o['unreviewed']:4d}->{n['unreviewed']:<4d} ballF {o['ball_frames']:4d}->{n['ball_frames']:<4d} "
            f"unres {o['unresolved_share']:.2f}->{n['unresolved_share']:.2f} both {report['both_centred_frames']:4d} "
            f"med {report['centre_distance_px']['median']} >5px {report['centre_distance_px']['over_5px']:3d} "
            f"onlyOld {report['ball_only_in_old']:3d} onlyNew {report['ball_only_in_new']:3d}"
        )
    (paths().logs / "phase2_compare_last.json").write_text(
        json.dumps(out_rows, indent=1, ensure_ascii=False)
    )
    return 0


def cmd_disagreements(args: argparse.Namespace) -> int:
    """Rule condition 4 aid: every contiguous run where only one annotation has a centred ball; for
    the longest runs, 5 frames (start..end) + a 2x crop of the middle frame. cyan = old, yellow = new.
    Short runs (<= 4 frames) at a track start/end are boundary differences, within tolerance."""
    import cv2
    import numpy as np

    from .common import iter_frames, locate_video

    state = read_state()
    out_dir = paths().campaign_dir / "qa" / "phase2" / "_disagreements"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_id in args.task_ids:
        task = state["tasks"][task_id]
        manifest = load_manifest(Path(task["manifest"]))
        old = centres(
            load_annotation(
                paths().annotated / "processed" / "ball" / f"{task['clip_id']}.json"
            )
        )
        new = centres(
            load_annotation(
                Path(task["attempts"][-1]["dir"]) / f"annotation_{task['clip_id']}.json"
            )
        )
        side = {}
        for i in old:
            if old[i] and not new.get(i):
                side[i] = "OLD"
            elif new.get(i) and not old[i]:
                side[i] = "NEW"
            elif old[i] and new.get(i) and len(old[i]) != len(new[i]):
                side[i] = "COUNT"
        runs: list[list[Any]] = []
        for i in sorted(side):
            if runs and runs[-1][1] == i - 1 and runs[-1][2] == side[i]:
                runs[-1][1] = i
            else:
                runs.append([i, i, side[i]])
        if not runs:
            continue
        runs = sorted(sorted(runs, key=lambda r: -(r[1] - r[0]))[: args.max_runs])
        wanted = sorted({a + k * (b - a) // 4 for a, b, _ in runs for k in range(5)})
        frames = {
            i: f
            for i, f in iter_frames(
                locate_video(manifest), manifest, wanted[0], wanted[-1] + 1
            )
            if i in wanted
        }
        rows = []
        for a, b, sd in runs:
            picks = [a + k * (b - a) // 4 for k in range(5)]
            tiles = []
            for i in picks:
                frame = frames[i]
                s_ = 400 / frame.shape[1]
                t = cv2.resize(
                    frame, (400, int(frame.shape[0] * s_)), interpolation=cv2.INTER_AREA
                )
                for src, col, rad in (
                    (old, (255, 255, 0), 10),
                    (new, (0, 255, 255), 6),
                ):
                    for *_, c in src.get(i, []):
                        if c:
                            cv2.circle(
                                t,
                                (int(c[0] * s_), int(c[1] * s_)),
                                rad,
                                col,
                                2,
                                cv2.LINE_AA,
                            )
                cv2.putText(
                    t,
                    f"f{i} {sd}-only",
                    (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(t)
            mid = (a + b) // 2
            c = next(
                (c for *_, c in (old if sd in ("OLD", "COUNT") else new)[mid] if c),
                None,
            )
            frame = frames[picks[2]]
            if c is not None:
                size = min(100, frame.shape[0], frame.shape[1])
                x0 = int(min(max(c[0] - size / 2, 0), frame.shape[1] - size))
                y0 = int(min(max(c[1] - size / 2, 0), frame.shape[0] - size))
                crop = frame[y0 : y0 + size, x0 : x0 + size]
                tiles.append(
                    cv2.resize(
                        crop,
                        (tiles[0].shape[0], tiles[0].shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            else:
                tiles.append(
                    np.zeros((tiles[0].shape[0], tiles[0].shape[0], 3), np.uint8)
                )
            row = np.hstack(tiles)
            lab = np.zeros((row.shape[0] + 16, row.shape[1], 3), np.uint8)
            lab[16:] = row
            cv2.putText(
                lab,
                f"{task['clip_id'][:11]} run f{a}-f{b} ({b - a + 1} frames) {sd} only",
                (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            rows.append(lab)
        width = max(r.shape[1] for r in rows)
        image = np.vstack(
            [
                np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
                for r in rows
            ]
        )
        cv2.imwrite(
            str(out_dir / f"{task['clip_id']}.jpg"),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        print(task["clip_id"][:40], "runs:", [(a, b, sd) for a, b, sd in runs])
    return 0


def presence_runs(
    old: dict[int, Any], new: dict[int, Any]
) -> list[tuple[int, int, str]]:
    side = {}
    for i in old:
        if old[i] and not new.get(i):
            side[i] = "OLD"
        elif new.get(i) and not old[i]:
            side[i] = "NEW"
        elif old[i] and new.get(i) and len(old[i]) != len(new[i]):
            side[i] = "COUNT"
    runs: list[list[Any]] = []
    for i in sorted(side):
        if runs and runs[-1][1] == i - 1 and runs[-1][2] == side[i]:
            runs[-1][1] = i
        else:
            runs.append([i, i, side[i]])
    return [tuple(r) for r in runs]


def cmd_triage(args: argparse.Namespace) -> int:
    """Numerical screening only; every presence disagreement still needs a visual decision."""
    state = read_state()
    out: dict[str, Any] = {"numeric_pass": [], "keep_old": [], "eyeball": {}}
    for task_id, task in state["tasks"].items():
        if task.get("phase") != 2 or task["status"] != "held":
            continue
        manifest = load_manifest(Path(task["manifest"]))
        old_a = load_annotation(
            paths().annotated / "processed" / "ball" / f"{task['clip_id']}.json"
        )
        new_a = load_annotation(
            Path(task["attempts"][-1]["dir"]) / f"annotation_{task['clip_id']}.json"
        )
        r = compare_pair(old_a, new_a, manifest)
        old, new, distance = r["old"], r["new"], r["centre_distance_px"]
        runs = presence_runs(centres(old_a), centres(new_a))
        passes = (
            new["unreviewed"] == 0
            and new["unresolved_share"] <= old["unresolved_share"]
            and distance["median"] is not None
            and distance["median"] <= 2
        )
        improved = (
            new["unresolved_share"] < old["unresolved_share"]
            or old["unreviewed"] > 0
            or bool(runs)
        )
        if not passes or not improved:
            out["keep_old"].append(task_id)
        elif runs:
            out["eyeball"][task_id] = runs
        else:
            out["numeric_pass"].append(task_id)
    (paths().logs / "phase2_triage.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False)
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rank")
    p.add_argument("--show", type=int, default=30)
    p = sub.add_parser("enqueue")
    p.add_argument("--top", type=int, required=True)
    p = sub.add_parser("compare")
    p.add_argument("task_ids", nargs="*")
    p.add_argument("--video", action="store_true")
    p = sub.add_parser("disagreements")
    p.add_argument("task_ids", nargs="+")
    p.add_argument("--max-runs", type=int, default=10000)
    p = sub.add_parser("triage")
    args = parser.parse_args(argv)
    return {
        "rank": cmd_rank,
        "enqueue": cmd_enqueue,
        "compare": cmd_compare,
        "disagreements": cmd_disagreements,
        "triage": cmd_triage,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
