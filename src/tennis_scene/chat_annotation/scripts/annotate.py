"""Local helpers for player/ball annotations; Chat receives no Python runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.configuration.paths import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

from ..runtime.contracts import (
    KIT_VERSION,
    Annotation,
    ClipManifest,
    make_template,
    read_json,
    write_json,
)
from ..runtime.geometry import interpolate_ball
from ..runtime.media import check_clip, extract_frames
from ..runtime.review import finalize
from ..runtime.validation import validate_annotation

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.tools",
    fields=(
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Local player/ball annotation helpers."
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "init", "frames", "interpolate", "validate", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True, type=Path)
        if name in {"preflight", "init", "frames", "finalize"}:
            command.add_argument("--video", required=True, type=Path)
        if name in {"interpolate", "validate", "finalize"}:
            command.add_argument("--annotations", required=True, type=Path)
        if name in {"init", "frames", "finalize"}:
            command.add_argument("--output", required=True, type=Path)
        if name in {"frames", "interpolate"}:
            command.add_argument("--start", required=True, type=int)
            command.add_argument("--stop", required=True, type=int)
        if name == "frames":
            command.add_argument("--crop", type=int, nargs=4)
        if name == "interpolate":
            command.add_argument("--track-id", required=True)
        if name == "validate":
            command.add_argument("--report", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        paths = {
            name: value
            for name, value in vars(args).items()
            if name in {"manifest", "video", "annotations", "output", "report"}
        }
        if args.command == "interpolate":
            paths["annotation_output"] = args.annotations
        if any(not path.is_absolute() for path in paths.values()):
            raise ValueError("all input/output paths must be absolute")
        root = args.manifest.parent.resolve()
        destinations = [
            paths[name]
            for name in ("output", "report", "annotation_output")
            if name in paths
        ]
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
                for key in ("manifest", "video", "annotations")
                if key in resolved
            }:
                raise ValueError("output must not overwrite an input")
        manifest = ClipManifest.model_validate(read_json(args.manifest))
        if manifest.kit_version != KIT_VERSION:
            raise ValueError("clip request version differs; regenerate the inputs")
        if args.command == "preflight":
            check_clip(args.video, manifest)
            print(json.dumps({"status": "ready", "frames": len(manifest.frames)}))
            return 0
        if args.command == "init":
            check_clip(args.video, manifest)
            if args.output.exists():
                raise FileExistsError("init will not overwrite an annotation")
            write_json(args.output, make_template(manifest).model_dump(mode="json"))
            return 0
        if args.command == "frames":
            crop = tuple(args.crop) if args.crop else None
            images = extract_frames(
                args.video, manifest, args.output, args.start, args.stop, crop
            )
            print(
                json.dumps(
                    {"images": [str(p) for p in images], "marked_reviewed": False}
                )
            )
            return 0
        if args.command == "finalize":
            archive, report = finalize(
                args.video, args.manifest, args.annotations, args.output
            )
            print(
                f"{report.status}: {report.reviewed_frames}/{report.target_frames} reviewed."
            )
            print(archive)
            return 0
        annotation = Annotation.model_validate(read_json(args.annotations))
        report = validate_annotation(annotation, manifest)
        if args.command == "validate":
            write_json(args.report, report.model_dump(mode="json"))
            print(report.model_dump_json(indent=2))
            return 1 if report.errors else 0
        if report.errors:
            raise ValueError("; ".join(report.errors))
        annotation = interpolate_ball(
            annotation, manifest, args.track_id, args.start, args.stop
        )
        # The helper updates coordinates only. Completion remains a caller decision.
        write_json(args.annotations, annotation.model_dump(mode="json"))
        return 0
    except Exception as error:
        print(f"failed: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
