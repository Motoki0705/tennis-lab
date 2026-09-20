"""Integrity checks; geometry acceptance is an explicit recorded visual review."""

from __future__ import annotations

import html
from pathlib import Path

from PIL import Image

from .contracts import AttemptRecord, Manifest
from .workspace import sha256


def load_attempt(root: Path, relative: str) -> AttemptRecord:
    attempt: AttemptRecord = AttemptRecord.model_validate_json(
        (root / relative).read_text()
    )
    return attempt


def validate_integrity(
    root: Path, manifest: Manifest, *, verify_source: bool = False
) -> None:
    if sha256(root / "variant.yaml") != manifest.config_sha256:
        raise ValueError("Saved variant configuration changed")
    for name, expected in manifest.prompt_sha256.items():
        if sha256(root / "prompts" / f"{name}.txt") != expected:
            raise ValueError(f"Fixed prompt changed: {name}")
    if sha256(root / "inputs/reference-source.jpg") != manifest.reference_source_sha256:
        raise ValueError("Reference source changed")
    if (
        manifest.reference_import is not None
        and sha256(root / "reference/clay.png") != manifest.reference_import["sha256"]
    ):
        raise ValueError("Imported fixed reference changed")
    accepted = [manifest.reference_accepted_attempt]
    for frame in manifest.frames:
        if sha256(root / "inputs/targets" / frame.name) != frame.source_sha256:
            raise ValueError(f"Input changed: {frame.name}")
        accepted.append(frame.accepted_attempt)
    for relative in accepted:
        if relative is None:
            continue
        attempt = load_attempt(root, relative)
        if not attempt.accepted or attempt.normalized_path is None:
            raise ValueError(f"Invalid accepted attempt: {relative}")
        if sha256(root / attempt.raw_path) != attempt.raw_sha256:
            raise ValueError(f"Generated original changed: {relative}")
        if sha256(root / attempt.normalized_path) != attempt.normalized_sha256:
            raise ValueError(f"Accepted image changed: {relative}")
    if verify_source:
        for relative, expected in manifest.source_files.items():
            if sha256(manifest.config.source_workspace / relative) != expected:
                raise ValueError(f"Original source artifact changed: {relative}")


def validate_ready(root: Path, manifest: Manifest) -> None:
    validate_integrity(root, manifest, verify_source=True)
    if manifest.pending or (
        manifest.reference_accepted_attempt is None
        and manifest.reference_import is None
    ):
        raise ValueError("Reference/pending request is incomplete")
    missing = [
        frame.name for frame in manifest.frames if frame.accepted_attempt is None
    ]
    if missing:
        raise ValueError(f"Unaccepted transformed frames: {missing}")
    expected = {frame.name for frame in manifest.frames}
    actual = {path.name for path in (root / "generation/accepted").iterdir()}
    if actual != expected:
        raise ValueError("Accepted images do not exactly match selected frame names")
    for name in expected:
        with Image.open(root / "generation/accepted" / name) as im:
            if im.size != manifest.image_size:
                raise ValueError(f"Wrong normalized image dimensions: {name}")


def write_comparison(root: Path, name: str, normalized_path: Path) -> None:
    """Save a diagnostic overlay; never use this visualization as training input."""
    with (
        Image.open(root / "inputs/targets" / name) as original,
        Image.open(normalized_path) as generated,
    ):
        overlay = Image.blend(original.convert("RGB"), generated.convert("RGB"), 0.5)
        overlay.save(root / "review" / f"{Path(name).stem}-overlay.jpg", quality=95)


def write_gallery(root: Path, manifest: Manifest) -> None:
    cards = []
    for frame in manifest.frames:
        if frame.accepted_attempt is None:
            continue
        name = html.escape(frame.name)
        stem = html.escape(Path(frame.name).stem)
        cards.append(
            f'<section><h2>{name} ({frame.split})</h2><div><img src="../inputs/targets/{name}" alt="Original"><img src="../generation/accepted/{name}" alt="Clay"><img src="{stem}-overlay.jpg" alt="50% overlay"></div></section>'
        )
    document = (
        '<!doctype html><meta charset="utf-8"><title>B00 clay comparison</title><style>body{font:14px sans-serif;background:#222;color:white}div{display:flex}img{width:33.33%;object-fit:contain}h2{font-size:16px}</style><h1>Original / generated / 50% overlay</h1>'
        + "\n".join(cards)
    )
    (root / "review/index.html").write_text(document, encoding="utf-8")
