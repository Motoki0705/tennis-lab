"""Move clips only after both AI-published target annotations pass validation."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from ..layout import done_video_path, published_video_path, video_path
from ..manifests import load_prepared_manifests, require_regular_path
from ..runtime.contracts import (
    BallAnnotation,
    PlayerAnnotation,
    parse_annotation,
    read_json,
    sha256_file,
)
from ..runtime.validation import validate_annotation


def sync_done(root: Path, *, dry_run: bool = False) -> dict[str, list[str]]:
    root = root.resolve()
    if not (root / "_preparation").is_dir():
        # Checked before the lock file is created inside an invalid root.
        raise ValueError("output root must contain _preparation manifests")
    report: dict[str, list[str]] = {
        "moved": [],
        "ready": [],
        "pending": [],
        "already_done": [],
        "errors": [],
    }
    lock = root / ".completion.lock"
    require_regular_path(lock, root)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        manifests = load_prepared_manifests(root)
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
                    require_regular_path(path, root)
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
                require_regular_path(source, root)
                require_regular_path(destination, root)
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
