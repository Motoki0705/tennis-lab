"""Publish hash-verified production NHT previews into the tracked report assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    REPOSITORY_ROOT
    / ".codex-loop"
    / "3dgs-synthetic-data"
    / "artifacts"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "assets" / "production-previews"


@dataclass(frozen=True)
class Preview:
    public_id: str
    artifact: Path
    category: str
    assessment: str
    overview_frame: int | None = None


PREVIEWS = (
    Preview(
        "alignment",
        ARTIFACT_ROOT / "cycle-17" / "production-alignment-frame-000080-v1",
        "alignment",
        "accepted",
        0,
    ),
    Preview(
        "blcs-seed-20260728",
        ARTIFACT_ROOT / "cycle-17" / "production-blcs-visible-video-seed-20260728-v1",
        "blcs",
        "accepted-visible-camera",
        10,
    ),
    Preview(
        "blcs-seed-20260730",
        ARTIFACT_ROOT / "cycle-17" / "production-blcs-visible-video-seed-20260730-v1",
        "blcs",
        "accepted-visible-camera",
    ),
    Preview(
        "blcs-seed-20260732",
        ARTIFACT_ROOT / "cycle-17" / "production-blcs-visible-video-seed-20260732-v1",
        "blcs",
        "accepted-visible-camera",
    ),
    Preview(
        "plcs-seed-20260728",
        ARTIFACT_ROOT / "cycle-17" / "production-plcs-video-seed-20260728-v1",
        "plcs",
        "accepted",
        6,
    ),
    Preview(
        "plcs-seed-20260729",
        ARTIFACT_ROOT / "cycle-17" / "production-plcs-video-seed-20260729-v1",
        "plcs",
        "accepted",
    ),
    Preview(
        "plcs-seed-20260731",
        ARTIFACT_ROOT / "cycle-17" / "production-plcs-video-seed-20260731-v1",
        "plcs",
        "accepted",
    ),
    Preview(
        "court-circle-075-complex",
        ARTIFACT_ROOT
        / "cycle-17"
        / "production-court-video-circle-scale-0.75-target-complex-v1",
        "court",
        "accepted-stable",
    ),
    Preview(
        "court-circle-100-court0",
        ARTIFACT_ROOT
        / "cycle-17"
        / "production-court-video-circle-scale-1.00-target-court_0-v1",
        "court",
        "accepted-with-local-artifacts",
    ),
    Preview(
        "court-circle-130-complex",
        ARTIFACT_ROOT
        / "cycle-17"
        / "production-court-video-circle-scale-1.30-target-complex-v1",
        "court",
        "rejected-outside-sfm-support",
        11,
    ),
    Preview(
        "court-ellipse-075-court1",
        ARTIFACT_ROOT
        / "cycle-17"
        / "production-court-video-ellipse-scale-0.75-target-court_1-v1",
        "court",
        "accepted-stable",
    ),
    Preview(
        "court-ellipse-100-complex",
        ARTIFACT_ROOT
        / "cycle-17"
        / "production-court-video-ellipse-scale-1.00-target-complex-v1",
        "court",
        "accepted-stable",
        11,
    ),
    Preview(
        "court-ellipse-130-court1",
        ARTIFACT_ROOT
        / "cycle-17"
        / "production-court-video-ellipse-scale-1.30-target-court_1-v1",
        "court",
        "rejected-outside-sfm-support",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _copy_verified(
    *,
    source_root: Path,
    source_record: dict[str, Any],
    destination: Path,
) -> dict[str, object]:
    source = source_root / source_record["relative_path"]
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha256 = _sha256(source)
    if actual_sha256 != source_record["sha256"]:
        raise RuntimeError(f"Source hash changed: {source}")
    if source.stat().st_size != source_record["size_bytes"]:
        raise RuntimeError(f"Source size changed: {source}")
    shutil.copy2(source, destination)
    published_sha256 = _sha256(destination)
    if published_sha256 != actual_sha256:
        raise RuntimeError(f"Published hash differs: {destination}")
    return {
        "relative_path": destination.relative_to(destination.parents[1]).as_posix(),
        "sha256": published_sha256,
        "size_bytes": destination.stat().st_size,
    }


def _load_preview(preview: Preview, temporary: Path) -> dict[str, object]:
    manifest_path = preview.artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "passed":
        raise RuntimeError(f"Preview did not pass rendering: {preview.artifact}")
    if manifest["renderer"]["rgb_overlay_used_in_native_render"]:
        raise RuntimeError(f"RGB overlay contaminated native render: {preview.artifact}")
    if not manifest["overlay_policy"]["raw_rgb_unchanged"]:
        raise RuntimeError(f"Raw RGB was not preserved: {preview.artifact}")

    target = temporary / preview.public_id
    target.mkdir()
    video = manifest["video"]
    files = {
        "raw_rgb": _copy_verified(
            source_root=preview.artifact,
            source_record=video["raw_rgb"],
            destination=target / "rgb.mp4",
        ),
        "diagnostic_overlay": _copy_verified(
            source_root=preview.artifact,
            source_record=video["diagnostic_overlay"],
            destination=target / "rgb-with-diagnostic-overlay.mp4",
        ),
        "contact_sheet": _copy_verified(
            source_root=preview.artifact,
            source_record=video["contact_sheet"],
            destination=target / "contact-sheet.jpg",
        ),
    }
    return {
        "public_id": preview.public_id,
        "category": preview.category,
        "assessment": preview.assessment,
        "source_artifact": str(preview.artifact),
        "source_content_fingerprint": manifest["content_fingerprint"],
        "task": manifest["task"],
        "frame_count": video["frame_count"],
        "fps": video["fps"],
        "duration_seconds": video["duration_seconds"],
        "source": manifest["source"],
        "metrics": manifest["metrics"],
        "files": files,
    }


def _overview_pair(preview: Preview) -> tuple[Image.Image, Image.Image]:
    if preview.overview_frame is None:
        raise ValueError(f"No overview frame assigned to {preview.public_id}.")
    filename = f"{preview.overview_frame:06d}.png"
    raw = Image.open(preview.artifact / "frames" / filename).convert("RGB")
    overlay = Image.open(preview.artifact / "overlays" / filename).convert("RGB")
    return raw, overlay


def _publish_overview(temporary: Path) -> dict[str, object]:
    overview = [preview for preview in PREVIEWS if preview.overview_frame is not None]
    cell_size = (960, 540)
    header_height = 52
    canvas = Image.new(
        "RGB",
        (cell_size[0] * 2, (cell_size[1] + header_height) * len(overview)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for row, preview in enumerate(overview):
        top = row * (cell_size[1] + header_height)
        label = (
            f"{preview.public_id} | {preview.assessment} | "
            "left=raw RGB, right=diagnostic overlay"
        )
        draw.text((16, top + 12), label, fill="black", font=font)
        raw, overlay = _overview_pair(preview)
        canvas.paste(ImageOps.fit(raw, cell_size), (0, top + header_height))
        canvas.paste(
            ImageOps.fit(overlay, cell_size),
            (cell_size[0], top + header_height),
        )
    destination = temporary / "production-rgb-overview.jpg"
    canvas.save(destination, quality=90, optimize=True)
    return {
        "relative_path": destination.relative_to(temporary).as_posix(),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "rows": [preview.public_id for preview in overview],
        "layout": "left raw RGB; right diagnostic overlay",
    }


def _copy_trajectory_plot(temporary: Path) -> dict[str, object]:
    source = (
        ARTIFACT_ROOT
        / "cycle-14"
        / "multicourt-orbit-render-v1"
        / "diagnostics"
        / "orbit-trajectories.png"
    )
    destination = temporary / "court-orbit-trajectories.png"
    shutil.copy2(source, destination)
    return {
        "relative_path": destination.relative_to(temporary).as_posix(),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _verify_record(root: Path, record: dict[str, Any]) -> None:
    path = root / record["relative_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["size_bytes"]:
        raise RuntimeError(f"Published size changed: {path}")
    if _sha256(path) != record["sha256"]:
        raise RuntimeError(f"Published hash changed: {path}")


def verify_publication() -> None:
    manifest_path = OUTPUT_ROOT / "manifest.json"
    publication = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = {preview.public_id for preview in PREVIEWS}
    actual_ids = {preview["public_id"] for preview in publication["previews"]}
    if publication["status"] != "passed" or actual_ids != expected_ids:
        raise RuntimeError("Published preview inventory changed.")
    if publication["preview_count"] != len(PREVIEWS):
        raise RuntimeError("Published preview count changed.")
    for preview in publication["previews"]:
        files = preview["files"]
        for record in files.values():
            _verify_record(OUTPUT_ROOT, record)
        if files["raw_rgb"]["sha256"] == files["diagnostic_overlay"]["sha256"]:
            raise RuntimeError(f"Raw and overlay videos are identical: {preview['public_id']}")
    _verify_record(OUTPUT_ROOT, publication["overview"])
    _verify_record(OUTPUT_ROOT, publication["trajectory_plot"])
    expected_fingerprint = publication.pop("content_fingerprint")
    if _canonical_sha256(publication) != expected_fingerprint:
        raise RuntimeError("Publication content fingerprint changed.")
    print(
        json.dumps(
            {
                "status": "passed",
                "preview_count": len(PREVIEWS),
                "content_fingerprint": expected_fingerprint,
            },
            indent=2,
        )
    )


def publish() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite published previews: {OUTPUT_ROOT}")
    temporary = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale publication directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        records = [_load_preview(preview, temporary) for preview in PREVIEWS]
        overview = _publish_overview(temporary)
        trajectory_plot = _copy_trajectory_plot(temporary)
        publication = {
            "schema": "tennis_production_nht_preview_publication_v1",
            "status": "passed",
            "preview_count": len(records),
            "category_counts": {
                category: sum(record["category"] == category for record in records)
                for category in ("alignment", "blcs", "plcs", "court")
            },
            "raw_rgb_policy": {
                "native_rgb_has_no_2d_overlay": True,
                "diagnostic_overlay_is_separate": True,
            },
            "previews": records,
            "overview": overview,
            "trajectory_plot": trajectory_plot,
        }
        publication["content_fingerprint"] = _canonical_sha256(publication)
        (temporary / "manifest.json").write_text(
            json.dumps(publication, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(OUTPUT_ROOT)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(OUTPUT_ROOT),
                "preview_count": len(PREVIEWS),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-hash an existing publication without changing it.",
    )
    args = parser.parse_args()
    if args.verify_only:
        verify_publication()
    else:
        publish()


if __name__ == "__main__":
    main()
