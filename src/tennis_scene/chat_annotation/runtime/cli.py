"""Local reference helpers; no Python runtime is distributed with clip inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.configuration.paths import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

from .contracts import (
    Annotation,
    ClipManifest,
    CourtPoint,
    CourtSample,
    FrameRange,
    ValidationReport,
    make_template,
    read_json,
    sha256_file,
    write_json,
)
from .geometry import complete_court, court_mode, interpolate_ball
from .media import check_clip, extract_frames
from .review import final_response, finalize
from .validation import validate_annotation

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.tools",
    fields=(
        BoundaryPathField(
            "kit",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "manifest",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "video",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "annotations",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.ANY,
            required=False,
        ),
        BoundaryPathField(
            "report",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.FILE,
            required=False,
        ),
        BoundaryPathField(
            "annotation_output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.FILE,
            required=False,
        ),
    ),
)


def verify_kit(root: Path, manifest: ClipManifest) -> dict[str, Any]:
    kit = read_json(root / "kit_manifest.json")
    identifier = hashlib.sha256(
        json.dumps(kit["files"], sort_keys=True).encode()
    ).hexdigest()
    if (
        identifier != kit["kit_id"]
        or identifier != manifest.kit_id
        or kit["kit_version"] != manifest.kit_version
    ):
        raise ValueError("Project kit version/hash differs from clip manifest")
    for name, digest in kit["files"].items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError(f"Project kit file missing or modified: {name}")
    definition = read_json(root / "court_definition.json")
    if not isinstance(definition, dict):
        raise ValueError("court definition must be an object")
    return definition


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fixed tennis annotation helpers. Read PROTOCOL.md first."
    )
    result.add_argument("--kit-dir", type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "init",
        "frames",
        "record-review",
        "new-court",
        "complete-court",
        "decide-court",
        "interpolate",
        "validate",
        "finalize",
    ):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True, type=Path)
        if name in {"preflight", "init", "frames", "finalize"}:
            command.add_argument("--video", required=True, type=Path)
        if name not in {"preflight", "init", "frames"}:
            command.add_argument("--annotations", required=True, type=Path)
        if name in {"init", "frames", "finalize"}:
            command.add_argument("--output", required=True, type=Path)
        if name in {"frames", "record-review", "interpolate"}:
            command.add_argument("--start", required=True, type=int)
            command.add_argument("--stop", required=True, type=int)
        if name == "frames":
            command.add_argument("--crop", type=int, nargs=4)
        if name == "record-review":
            command.add_argument("--camera", action="store_true")
        if name == "new-court":
            command.add_argument("--orientation-note", required=True)
        if name in {"new-court", "complete-court"}:
            command.add_argument("--frame", required=True, type=int)
        if name == "interpolate":
            command.add_argument("--track-id", required=True)
        if name == "validate":
            command.add_argument("--report", required=True, type=Path)
        if name == "finalize":
            command.add_argument("--download-base")
    return result


def main(kit_root: Path) -> int:
    args = parser().parse_args()
    manifest: ClipManifest | None = None
    try:
        root = args.kit_dir or kit_root
        if not root.is_absolute():
            raise ValueError("--kit-dir must be an absolute path")
        root = root.resolve()
        paths = {
            name: value
            for name, value in vars(args).items()
            if name in {"manifest", "video", "annotations", "output", "report"}
        }
        paths["kit"] = root
        mutations = {
            "record-review",
            "new-court",
            "complete-court",
            "decide-court",
            "interpolate",
        }
        if args.command in mutations:
            paths["annotation_output"] = args.annotations
        destinations = [
            paths[name]
            for name in ("output", "report", "annotation_output")
            if name in paths
        ]
        if any(not path.is_absolute() for path in paths.values()):
            raise ValueError(
                "all kit input/output paths must be absolute (for example /mnt/data/clip_manifest.json)"
            )
        output_root = destinations[0].parent.resolve() if destinations else root
        resolver = PathResolver(
            RuntimePathRoots(
                project_root=root,
                data_root=root,
                checkpoint_root=root,
                artifact_root=root,
                output_root=output_root,
                cache_root=root,
                external_asset_root=root,
            )
        )
        resolved = PATH_BOUNDARY.validate(
            paths, resolver=resolver, independent_artifact_inputs=True
        )
        for name in ("manifest", "video", "annotations", "output", "report"):
            if name in resolved:
                setattr(args, name, resolved.declared(name).path)
        for name in ("output", "report"):
            if name in resolved and resolved.declared(name).path in {
                resolved.declared(key).path
                for key in ("kit", "manifest", "video", "annotations")
                if key in resolved
            }:
                raise ValueError(
                    "output must not overwrite an input or the Project kit"
                )
        manifest = ClipManifest.model_validate(read_json(args.manifest))
        definition = verify_kit(root, manifest)
        manifest_hash = sha256_file(args.manifest)
        if args.command == "preflight":
            check_clip(args.video, manifest)
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "kit_id": manifest.kit_id,
                        "frames": len(manifest.frames),
                        "targets": sum(f.is_target for f in manifest.frames),
                    }
                )
            )
            return 0
        if args.command == "init":
            check_clip(args.video, manifest)
            if args.output.exists():
                raise FileExistsError("init will not overwrite an annotation")
            write_json(
                args.output,
                make_template(manifest, manifest_hash).model_dump(mode="json"),
            )
            return 0
        if args.command == "frames":
            crop = tuple(args.crop) if args.crop else None
            images = extract_frames(
                args.video, manifest, args.output, args.start, args.stop, crop
            )
            print(
                json.dumps(
                    {"images": [str(path) for path in images], "marked_reviewed": False}
                )
            )
            return 0
        if args.command == "finalize":
            archive, report = finalize(
                args.video,
                args.manifest,
                args.annotations,
                root,
                args.output,
                definition,
                args.download_base,
            )
            print(
                (archive.parent / "FINAL_RESPONSE.txt").read_text(encoding="utf-8"),
                end="",
            )
            return 1 if report.errors else 0
        annotation = Annotation.model_validate(read_json(args.annotations))
        if (annotation.clip_id, annotation.kit_id, annotation.manifest_sha256) != (
            manifest.clip_id,
            manifest.kit_id,
            manifest_hash,
        ):
            raise ValueError("annotation is for another clip/kit/manifest")
        if args.command == "validate":
            report = validate_annotation(
                annotation, manifest, manifest_hash, definition
            )
            write_json(args.report, report.model_dump(mode="json"))
            print(report.model_dump_json(indent=2))
            return 1 if report.errors else 0
        if args.command == "record-review":
            interval = FrameRange(start=args.start, stop=args.stop)
            if interval.stop > len(manifest.frames):
                raise ValueError("review extends beyond clip")
            target_ranges = (
                annotation.camera_review_ranges
                if args.camera
                else annotation.inspection_ranges
            )
            target_ranges.append(interval)
        elif args.command == "new-court":
            if not 0 <= args.frame < len(manifest.frames) or any(
                s.frame_index == args.frame for s in annotation.court_samples
            ):
                raise ValueError("court frame outside clip or already present")
            annotation.court_samples.append(
                CourtSample(
                    frame_index=args.frame,
                    orientation="known",
                    orientation_note=args.orientation_note,
                    points=[
                        CourtPoint(
                            index=i,
                            name=name,
                            point_px=None,
                            visibility="unresolved",
                            source="unresolved",
                            source_frames=[],
                            anchor_indices=[],
                        )
                        for i, name in enumerate(definition["names"])
                    ],
                )
            )
        elif args.command == "complete-court":
            sample_index = next(
                i
                for i, sample in enumerate(annotation.court_samples)
                if sample.frame_index == args.frame
            )
            annotation.court_samples[sample_index] = complete_court(
                annotation.court_samples[sample_index], definition, manifest
            )
        elif args.command == "decide-court":
            mode = court_mode(annotation, manifest)
            annotation.court_mode = "static" if mode == "static" else "dynamic"
            if mode == "static":
                for frame in annotation.frames:
                    frame.court_reference_frame = 0
                    frame.court_review = "complete"
            else:
                # Invalidate stale static references; never manufacture moving-camera labels.
                for frame in annotation.frames:
                    if frame.court_reference_frame != frame.frame_index:
                        frame.court_reference_frame = None
                        frame.court_review = "unreviewed"
            print(mode)
        elif args.command == "interpolate":
            annotation = interpolate_ball(
                annotation, manifest, args.track_id, args.start, args.stop
            )
        write_json(args.annotations, annotation.model_dump(mode="json"))
        return 0
    except Exception as error:
        if manifest is not None:
            report = ValidationReport(
                status="failed",
                reviewed_frames=0,
                target_frames=sum(f.is_target for f in manifest.frames),
                errors=[str(error)],
                issues=[],
            )
            print(
                final_response(
                    manifest, report, f"未生成（{type(error).__name__}: {error}）"
                ),
                end="",
            )
        else:
            reason = " ".join(f"{type(error).__name__}: {error}".splitlines())
            print(
                f"状態: failed\n入力: 未確認\n元動画: 未確認\n処理: 0/未確認フレーム、要確認1件\n成果物: 未生成（{reason}）"
            )
        return 1
