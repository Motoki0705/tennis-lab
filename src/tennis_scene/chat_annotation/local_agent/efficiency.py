"""Worker efficiency per prompt version (WORKER.md + ct.py), from Codex rollouts.

  efficiency.py            # table per version label (versions.json) and A/B arm; writes logs/efficiency.json
Normalised per 100 annotated frames. Quality guards next to cost: unresolved share of
ball frames and trajectory flags, so a cheaper version is not silently a sloppier one.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from datetime import datetime
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from .common import load_annotation, load_manifest
from .configuration import paths
from .path_contracts import campaign_resolver, validate_command_paths
from .qa import ball_metrics

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.local_agent",
    fields=(BoundaryPathField("campaign", PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                              must_exist=True, allow_role_root=True),),
)



def sessions_directory() -> Path:
    return paths().codex_home / "sessions"


def sessions_by_cwd(cache: dict[str, str]) -> dict[str, list[Path]]:
    """All rollouts of the last 7 days keyed by their session cwd (= attempt dir), oldest first.
    A launcher bug on 2026-09-30 re-ran codex in 38 attempt dirs after the first run had finished;
    the attempt's cost is its FIRST session (the real worker run), re-runs are reported separately."""
    index: dict[str, list[Path]] = {}
    for day in sorted(sessions_directory().glob("*/*/*"), reverse=True)[:7]:
        for path in sorted(day.glob("rollout-*.jsonl")):
            key = str(path)
            if key not in cache:
                with path.open(encoding="utf-8", errors="ignore") as handle:
                    first = handle.readline()
                try:
                    cache[key] = (json.loads(first).get("payload") or {}).get(
                        "cwd"
                    ) or ""
                except json.JSONDecodeError:
                    continue
            index.setdefault(cache[key], []).append(path)
    return {cwd: sorted(paths) for cwd, paths in index.items()}


def scan_cached(path: Path, cache: dict[str, dict[str, float]]) -> dict[str, float]:
    """Rollouts of finished attempts do not change; key on path + size + mtime to be safe."""
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    if key not in cache:
        cache[key] = scan(path)
    return cache[key]


def scan(path: Path) -> dict[str, float]:
    requests = images = 0
    last = total = None
    first_ts = last_ts = None
    for line in path.open(encoding="utf-8", errors="ignore"):
        stamp = (
            line[14 : line.index('"', 14)]
            if line.startswith('{"timestamp":"')
            else None
        )
        if stamp:
            first_ts = first_ts or stamp
            last_ts = stamp
        if '"token_count"' in line:
            payload = json.loads(line).get("payload") or {}
            info = payload.get("info") or {}
            if info.get("last_token_usage"):
                requests += 1
                last, total = info["last_token_usage"], info["total_token_usage"]
        elif '"input_image"' in line:
            images += line.count('"input_image"')
    seconds = (
        (
            datetime.fromisoformat(last_ts.rstrip("Z"))
            - datetime.fromisoformat(first_ts.rstrip("Z"))
        ).total_seconds()
        if first_ts and last_ts
        else 0.0
    )
    return {
        "requests": requests,
        "images": images,
        "session_minutes": round(seconds / 60, 1),
        "ctx_end": (last or {}).get("total_tokens", 0),
        "input": (total or {}).get("input_tokens", 0),
        "cached": (total or {}).get("cached_input_tokens", 0),
        "output": (total or {}).get("output_tokens", 0),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    PATH_BOUNDARY.validate({"campaign": paths().campaign_dir}, resolver=campaign_resolver(paths()))
    validate_command_paths()
    state = json.loads(
        (paths().campaign_dir / "state.json").read_text(encoding="utf-8")
    )
    cache = (
        json.loads(
            (paths().logs / "efficiency_scan_cache.json").read_text(encoding="utf-8")
        )
        if (paths().logs / "efficiency_scan_cache.json").exists()
        else {}
    )
    cwd_cache = cache.setdefault("__cwd__", {})
    by_cwd = sessions_by_cwd(cwd_cache)
    rows = []
    for task in state["tasks"].values():
        for record in task["attempts"]:
            if record.get("kind") != "done" or not record.get("ended_at"):
                continue
            session_files = by_cwd.get(record["dir"], [])
            if not session_files:
                continue
            stats = scan_cached(session_files[0], cache)
            attempt_dir = Path(record["dir"])
            annotation = load_annotation(
                attempt_dir / f"annotation_{task['clip_id']}.json"
            )
            metrics = ball_metrics(annotation, load_manifest(Path(task["manifest"])))
            frames = metrics["frames"]
            minutes = stats[
                "session_minutes"
            ]  # first session only (wall time of the real worker run)
            version = record.get("prompt_version")
            label = version_labels().get(version, version) if version else "v1-baseline"
            if record.get("variant") not in (None, "base"):
                label += "+" + record["variant"]
            elif record.get("variant") == "base":
                label += "+base"
            if record["n"] > 1:
                label += " (continuation)"  # starts from a previous annotation: a different workload
            rows.append(
                {
                    "version": label,
                    "variant": record.get("variant"),
                    "prompt_version": version,
                    "frames": frames,
                    "ball_frames": metrics["frames_with_ball"],
                    "minutes": minutes,
                    "unresolved_share": metrics["unresolved_ratio_of_ball_frames"],
                    "flags": len(metrics["speed_flags"])
                    + len(metrics["isolated_points"]),
                    "rerun_sessions": len(session_files) - 1,
                    **stats,
                }
            )
    (paths().logs / "efficiency_scan_cache.json").write_text(json.dumps(cache))
    by_version: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for row in rows:
        by_version[row["version"]].append(row)
    summary = {}
    for version, items in by_version.items():

        def per100(key: str, entries: list[dict[str, float]] = items) -> float:
            return round(st.median(r[key] / r["frames"] * 100 for r in entries), 2)

        summary[version] = {
            "tasks": len(items),
            "with_accidental_rerun": sum(1 for r in items if r["rerun_sessions"]),
            "median_minutes": round(st.median(r["minutes"] for r in items), 1),
            "per100_requests": per100("requests"),
            "per100_images": per100("images"),
            # one number for quota: uncached input 1, cached input 0.1, output 8 (API price ratios)
            "per100_cost_kunits": round(
                st.median(
                    ((r["input"] - r["cached"]) + 0.1 * r["cached"] + 8 * r["output"])
                    / r["frames"]
                    * 100
                    for r in items
                )
                / 1e3,
                1,
            ),
            "per100_input_Mtok": round(
                st.median(r["input"] / r["frames"] * 100 for r in items) / 1e6, 3
            ),
            "per100_output_ktok": round(
                st.median(r["output"] / r["frames"] * 100 for r in items) / 1e3, 2
            ),
            "median_ctx_end_k": round(st.median(r["ctx_end"] for r in items) / 1e3, 1),
            "median_unresolved_share": round(
                st.median(r["unresolved_share"] for r in items), 3
            ),
            "median_flags": st.median(r["flags"] for r in items),
        }
    (paths().logs / "efficiency.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1)
    )
    print(json.dumps(summary, indent=1))
    return 0




def version_labels() -> dict[str, str]:
    from .configuration import json_object

    return dict(json_object(paths().campaign_dir / "versions.json")["labels"])
