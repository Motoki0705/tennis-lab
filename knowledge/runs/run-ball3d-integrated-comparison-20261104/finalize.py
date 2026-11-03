"""Verify immutable sources, package relative HTML assets, and seal final checksums."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from common import SOURCE_PINS, code_fingerprint, git_head, source_inventory, write_json

from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.visualization.ball3d.rendering import validate_video
from src.tennis_scene.visualization.ball3d.report import (
    _References,
    broken_html_references,
)
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    parser.add_argument("--videos", type=Path)
    args = parser.parse_args()
    output = args.output
    if (output / "final-artifacts.json").exists():
        raise ValueError("Final comparison already sealed")
    plan = json.loads((output / "execution-plan.json").read_text())
    if code_fingerprint() != plan["code_fingerprint"]:
        raise ValueError("Executable bytes differ from the executed plan")
    if broken_html_references(output / "index.html"):
        raise ValueError("Broken comparison HTML references")
    verification = {
        "packaging": {
            "code_commit": git_head(),
            "script_sha256": dual_sha256(Path(__file__)),
        }
    }
    if args.videos is not None:
        bundle = load_bundle(
            Path(plan["inputs"]) / "fixed_input_bundle_v1",
            expected_sha256=plan["bundle_sha256"],
        )
        videos = {
            camera: args.videos / f"{camera}.mp4" for camera in bundle.inputs.camera_ids
        }
        verify_videos(bundle, videos)
        verification["input_videos_full_decode"] = {
            camera: validate_video(
                path,
                frames=bundle.inputs.num_frames,
                fps=bundle.inputs.fps,
                size=bundle.inputs.source_size_wh,
            )
            for camera, path in videos.items()
        }
    for directory in SOURCE_PINS:
        inventory = source_inventory(Path(plan["inputs"]), directory)
        root = Path(plan["inputs"]) / directory
        for relative, entry in inventory["files"].items():
            p = root / relative
            if (
                p.stat().st_size != entry["size_bytes"]
                or dual_sha256(p) != entry["sha256"]
            ):
                raise ValueError(f"Immutable source artifact changed: {p}")
        verification[directory] = {
            "manifest_sha256": SOURCE_PINS[directory],
            "verified_files": len(inventory["files"]),
            "unchanged": True,
        }
    (output / "evidence").mkdir(exist_ok=True)
    if len({p.name for p in args.evidence}) != len(args.evidence):
        raise ValueError("Evidence basenames must be unique")
    for source in args.evidence:
        if not source.is_file():
            raise ValueError(f"Explicit evidence is missing: {source}")
        shutil.copyfile(source, output / "evidence" / source.name)
    if args.portable:
        parser = _References()
        parser.feed((output / "index.html").read_text())
        paths = {"index.html", *parser.paths}
        portable = output / "portable"
        portable.mkdir(exist_ok=False)
        for relative in sorted(paths):
            source = (output / relative).resolve()
            if (
                Path(relative).is_absolute()
                or not source.is_relative_to(output.resolve())
                or not source.is_file()
            ):
                raise ValueError(f"Nonportable reference: {relative}")
            destination = portable / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if broken_html_references(portable / "index.html"):
            raise ValueError("Broken copied portable page")
        with zipfile.ZipFile(
            output / "portable-comparison.zip", "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for relative in sorted(paths):
                # Fixed archive metadata; this does not alter the data files.
                info = zipfile.ZipInfo(relative, date_time=(2026, 11, 4, 0, 0, 0))
                with (
                    (portable / relative).open("rb") as source,
                    archive.open(info, "w") as destination,
                ):
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        verification["portable"] = {
            "page": "portable/index.html",
            "files": len(paths),
            "broken_references": [],
            "zip": "portable-comparison.zip",
            "zip_sha256": dual_sha256(output / "portable-comparison.zip"),
        }
    write_json(output / "final-verification.json", verification)
    files = {
        str(p.relative_to(output)): {
            "size_bytes": p.stat().st_size,
            "sha256": dual_sha256(p),
        }
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "final-artifacts.json"
    }
    write_json(
        output / "final-artifacts.json",
        {
            "schema": "ball3d_integrated_final_artifacts.v1",
            "files": files,
            "code_commit": plan["code_commit"],
            "execution_plan_sha256": dual_sha256(output / "execution-plan.json"),
        },
    )
    for relative, entry in files.items():
        p = output / relative
        if p.stat().st_size != entry["size_bytes"] or dual_sha256(p) != entry["sha256"]:
            raise ValueError(f"Final artifact checksum mismatch: {relative}")
    print(
        json.dumps(
            {
                "files_verified": len(files),
                "final_artifacts_sha256": dual_sha256(output / "final-artifacts.json"),
                "portable": verification.get("portable"),
                "immutable_sources_verified": sum(
                    v.get("verified_files", 0) for v in verification.values()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
