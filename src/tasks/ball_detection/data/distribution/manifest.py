"""Content manifests for transferred files and zero-copy source verification."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256
from src.utils.data.verification import SharedFileVerification

DISTRIBUTION_SCHEMA = "ball_frame_store_distribution.v1"
DISTRIBUTION_FILE = "distribution.json"


def relative_file(value: str) -> Path:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or path.as_posix() != value or value == "."):
        raise ValueError(f"Expected a canonical relative file path: {value!r}")
    return Path(value)


def local_file(root: Path, value: str) -> Path:
    path = root / relative_file(value)
    if (not path.resolve().is_relative_to(root.resolve())
            or any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent)):
        raise ValueError(f"Distribution file escapes its root or uses a symlink: {value}")
    if not path.is_file():
        raise ValueError(f"Distribution file is missing: {value}")
    return path.resolve()


def load_distribution(directory: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(local_file(directory, DISTRIBUTION_FILE).read_text())
    if manifest.get("schema_version") != DISTRIBUTION_SCHEMA:
        raise ValueError("Unknown ball frame store distribution schema")
    files = manifest["files"]
    paths = [entry["path"] for entry in files]
    if not paths or len(set(paths)) != len(paths) or DISTRIBUTION_FILE in paths:
        raise ValueError("Distribution file paths must be nonempty, unique, and exclude the manifest")
    for entry in files:
        relative_file(entry["path"])
        if (type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0
                or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError("Invalid distribution file size or SHA-256")
    clips = manifest["clips"]
    if ([c["index"] for c in clips] != list(range(len(clips)))
            or len({c["clip_id"] for c in clips}) != len(clips)
            or len({c["source_index"] for c in clips}) != len(clips)):
        raise ValueError("Distribution requires an explicit one-to-one clip index mapping")
    for clip in clips:
        relative_file(clip["payload"])
        relative_file(clip["source_payload"])
    if len({c["payload"] for c in clips}) != len(clips) or not {c["payload"] for c in clips} <= set(paths):
        raise ValueError("Every distribution clip needs one distinct manifested payload")
    if not {"metadata.json", "index.npz"} <= set(paths):
        raise ValueError("Distribution must include its metadata and index")
    return manifest


def source_payloads(directory: Path, source_directory: Path) -> dict[str, Path]:
    """Resolve all payloads against the pinned source, never search for substitutes."""
    manifest = load_distribution(directory)
    for name, digest in manifest["source"]["hashes"].items():
        if name not in {"metadata.json", "index.npz"} or dual_sha256(local_file(source_directory, name)) != digest:
            raise ValueError("Distribution source store differs from its pinned identity")
    if set(manifest["source"]["hashes"]) != {"metadata.json", "index.npz"}:
        raise ValueError("Distribution source must pin metadata and index")
    source_directory = source_directory.resolve()
    store = BallFrameStore(source_directory)
    result = {}
    for entry in manifest["clips"]:
        clip = store.clip_by_id(entry["clip_id"])
        path = local_file(source_directory, entry["source_payload"])
        if clip.index != entry["source_index"] or path != store.image_path(clip):
            raise ValueError("Distribution source clip mapping differs from its metadata")
        result[clip.clip_id] = path
    return result


def distribution_files(
    directory: Path, *, source_directory: Path | None = None,
) -> dict[str, Path]:
    """Get exact upload sources, without copying or linking image payloads."""
    manifest = load_distribution(directory)
    external = source_payloads(directory, source_directory) if source_directory is not None else None
    payloads = {c["payload"]: c["clip_id"] for c in manifest["clips"]}
    return {
        entry["path"]: (external[payloads[entry["path"]]]
                        if external is not None and entry["path"] in payloads
                        else local_file(directory, entry["path"]))
        for entry in manifest["files"]
    }


def verify_distribution(
    directory: Path, *, source_directory: Path | None = None, workers: int = 1,
) -> dict[str, Any]:
    """Check every byte and the complete declaration; refuse partial transfers."""
    manifest = load_distribution(directory)
    directory = directory.resolve()
    files = distribution_files(directory, source_directory=source_directory)
    for entry in manifest["files"]:
        path = files[entry["path"]]
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Distribution size mismatch: {entry['path']}")
    expected = set(files) | {DISTRIBUTION_FILE}
    if source_directory is not None:
        expected -= {c["payload"] for c in manifest["clips"]}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError(f"Distribution file set mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    verifier = SharedFileVerification({files[e["path"]]: e["sha256"] for e in manifest["files"]})
    verifier.verify_all(workers)
    external = source_payloads(directory, source_directory) if source_directory is not None else None
    store = BallFrameStore(directory, external_payloads=external)
    if [(c.index, c.clip_id) for c in store.clips] != [(c["index"], c["clip_id"]) for c in manifest["clips"]]:
        raise ValueError("Distribution clip declaration differs from its store")
    if store.metadata.get("version") == manifest["source"]["version"]:
        raise ValueError("A distribution must declare a new dataset version")
    for clip in store.clips:
        if clip.source == "chat_annotation" and dual_sha256(store.annotation_path(clip)) != clip.annotation_sha256:
            raise ValueError("Distribution original annotation checksum differs")
    return dict(
        schema_version=1, status="verified", files=len(files) + 1,
        size_bytes=sum(e["size_bytes"] for e in manifest["files"]) + (directory / DISTRIBUTION_FILE).stat().st_size,
        clips=len(store.clips), frames=len(store), version=store.metadata["version"],
        distribution_sha256=dual_sha256(directory / DISTRIBUTION_FILE),
        mode="source_payloads" if source_directory is not None else "self_contained",
    )
