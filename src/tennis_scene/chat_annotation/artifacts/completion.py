"""Move clips only after both AI-published target annotations pass validation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from ..layout import done_video_path, published_video_path, video_path
from ..runtime.contracts import (
    BallAnnotation,
    ClipManifest,
    PlayerAnnotation,
    annotation_clip_id,
    parse_annotation,
    read_json,
    sha256_file,
)
from ..runtime.validation import validate_annotation


def _regular_path(path: Path, root: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError("path outside output root")
    if any(
        part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)
    ):
        raise ValueError(f"symlink is forbidden: {path}")


def sync_done(root: Path, *, dry_run: bool = False) -> dict[str, list[str]]:
    root = root.resolve()
    if not (root / "_preparation").is_dir():
        raise ValueError("output root must contain _preparation manifests")
    report: dict[str, list[str]] = {
        "moved": [],
        "ready": [],
        "pending": [],
        "already_done": [],
        "errors": [],
    }
    lock = root / ".completion.lock"
    _regular_path(lock, root)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        manifests: dict[str, ClipManifest] = {}
        for path in sorted(
            (root / "_preparation").glob("*/*/clips/*/clip_manifest.json")
        ):
            if path.parent.name.startswith(".building-"):
                continue
            _regular_path(path, root)
            manifest = ClipManifest.model_validate(read_json(path))
            ready_path = path.parents[2] / "ready" / f"{path.parent.name}.json"
            _regular_path(ready_path, root)
            ready = read_json(ready_path)
            if ready["files"] != {
                "clip_manifest.json": sha256_file(path),
                manifest.filename: manifest.sha256,
            }:
                raise ValueError(f"preparation receipt mismatch: {path}")
            clip_id = annotation_clip_id(manifest)
            if clip_id in manifests:
                raise ValueError(f"duplicate clip identity: {clip_id}")
            manifests[clip_id] = manifest
        processed = root / "annotated" / "processed"
        candidates = {
            path.stem
            for target in ("ball", "player")
            for path in (processed / target).glob("*.json")
        }
        for clip_id in sorted(candidates):
            try:
                if clip_id not in manifests:
                    raise ValueError("no matching prepared clip")
                manifest = manifests[clip_id]
                complete = True
                for target, model in (
                    ("ball", BallAnnotation),
                    ("player", PlayerAnnotation),
                ):
                    path = processed / target / f"{clip_id}.json"
                    _regular_path(path, root)
                    if not path.exists():
                        complete = False
                        continue
                    annotation = parse_annotation(read_json(path))
                    if not isinstance(annotation, model):
                        raise ValueError(
                            f"{target} file has the wrong annotation schema"
                        )
                    result = validate_annotation(annotation, manifest)
                    if result.errors:
                        raise ValueError(f"{target}: {'; '.join(result.errors)}")
                    if result.status != "completed":
                        complete = False
                if not complete:
                    report["pending"].append(clip_id)
                    continue
                source = video_path(root, manifest)
                destination = done_video_path(root, manifest)
                _regular_path(source, root)
                _regular_path(destination, root)
                # Recover interruption between link and unlink; a different copy is a conflict.
                linked = (
                    source.exists()
                    and destination.exists()
                    and source.samefile(destination)
                )
                current = (
                    destination if linked else published_video_path(root, manifest)
                )
                if not current.is_file() or sha256_file(current) != manifest.sha256:
                    raise ValueError("clip bytes do not match the preparation manifest")
                if current == destination and not linked:
                    report["already_done"].append(clip_id)
                elif dry_run:
                    report["ready"].append(clip_id)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not linked:
                        os.link(
                            source, destination
                        )  # Atomic, never overwrites; same filesystem required.
                    source.unlink()
                    report["moved"].append(clip_id)
            except (ValueError, OSError) as error:
                report["errors"].append(f"{clip_id}: {error}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/chat_annotation"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0,
        help="0: one pass; positive: poll interval",
    )
    args = parser.parse_args()
    if args.watch_seconds < 0 or (args.watch_seconds and args.dry_run):
        parser.error(
            "watch requires a positive interval and cannot be combined with dry-run"
        )
    while True:
        report = sync_done(args.root, dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not args.watch_seconds:
            raise SystemExit(1 if report["errors"] else 0)
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
