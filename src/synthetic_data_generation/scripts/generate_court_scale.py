"""Freeze Court production/comparison requests, then render through the queue."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    QualityPolicy,
    TrajectoryMethod,
    canonical_bytes,
)
from src.synthetic_data_generation.dataset.court.scale.durability import (
    DurableGeneration,
    fetch_completed_dataset,
    restore_inputs,
)
from src.synthetic_data_generation.dataset.court.scale.durable_storage import (
    SkillDriveFiles,
)
from src.synthetic_data_generation.dataset.court.scale.publication import (
    PRODUCTION_REQUEST_SCHEMA,
    execute_request,
    freeze_request,
    load_dataset_version,
    persist_candidate_plan,
    read_json,
)
from src.synthetic_data_generation.dataset.court.scale.recipes import ProductionRecipe
from src.synthetic_data_generation.dataset.court.scale.runtime import (
    ColabNHTRenderer,
    LocalNHTRenderer,
    load_scene_inputs,
    renderer_fingerprint,
)
from src.synthetic_data_generation.dataset.court.scale.telemetry import (
    NvidiaL4Telemetry,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court_scale_generation",
    fields=(
        BoundaryPathField(
            "scenes_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "asset_manifest",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "production_recipe",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "output_root",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            required=False,
        ),
        BoundaryPathField(
            "recipe",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "quality",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "request",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            many=True,
            required=False,
        ),
        BoundaryPathField(
            "dataset_version",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--scenes-root", type=Path, required=True)
    plan.add_argument("--scenes", nargs="+", required=True)
    plan.add_argument("--asset-manifest", type=Path)
    plan.add_argument("--asset-manifest-sha256")
    selection = plan.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--methods",
        nargs="+",
        choices=[
            m
            for m in TrajectoryMethod
            if m is not TrajectoryMethod.SAVED_TRAIN_TRAJECTORIES
        ],
    )
    selection.add_argument("--production-recipe", type=Path)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--recipe", type=Path, help="comparison geometry recipe")
    plan.add_argument("--quality", type=Path, help="comparison quality policy")
    plan.add_argument("--generation-seed", type=int, required=True)
    plan.add_argument("--trajectory-count", type=int, required=True)
    plan.add_argument("--density-level", type=int, required=True)
    plan.add_argument(
        "--method-prefix",
        action="append",
        default=[],
        help="METHOD=COUNT; fixed before rendering",
    )
    plan.add_argument(
        "--scene-prefix",
        action="append",
        default=[],
        help="METHOD:SCENE=COUNT; explicit per-scene trajectory prefix",
    )
    plan.add_argument(
        "--method-density",
        action="append",
        default=[],
        help="METHOD=LEVEL; explicit method-specific density",
    )
    plan.add_argument(
        "--legacy-hull-center",
        action="append",
        default=[],
        help="SCENE=false|true; explicit missing-key addition",
    )
    render = commands.add_parser("render")
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--request", type=Path, nargs="+", required=True)
    render.add_argument("--batch-size", type=int, default=1)
    colab = commands.add_parser(
        "render-colab",
        help="Durable production rendering on an explicitly allocated Colab L4",
    )
    colab.add_argument("--output-root", type=Path, required=True)
    source = colab.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--request", type=Path, nargs="+", help="One frozen production request"
    )
    source.add_argument(
        "--request-sha256", help="Restore this request from Drive after VM loss"
    )
    colab.add_argument("--drive-path", required=True)
    colab.add_argument(
        "--retry-staging",
        action="store_true",
        help="Archive failed staging to Drive before redrawing; never delete it",
    )
    colab.add_argument(
        "--stop-after-caches",
        type=int,
        help="Intentional stop after this many verified cache acknowledgements",
    )
    fetch = commands.add_parser(
        "fetch", help="Recover a completed Drive dataset and strictly decode every JPEG"
    )
    fetch.add_argument("--output-root", type=Path, required=True)
    fetch.add_argument("--drive-path", required=True)
    fetch.add_argument("--request-sha256", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--dataset-version", type=Path, required=True)
    args = parser.parse_args()
    names = {
        "plan": (
            "scenes_root",
            "asset_manifest",
            "production_recipe",
            "output_root",
            "recipe",
            "quality",
        ),
        "render": ("output_root", "request"),
        "render-colab": ("output_root", "request"),
        "fetch": ("output_root",),
        "validate": ("dataset_version",),
    }[args.command]
    arguments: dict[str, Path | list[Path]] = {}
    for name in names:
        value = getattr(args, name)
        if value is not None:
            arguments[name] = (
                [path.expanduser().resolve() for path in value]
                if name == "request"
                else value.expanduser().resolve()
            )
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=args.scenes_root.expanduser().resolve()
        if args.command == "plan"
        else PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=args.output_root.expanduser().resolve().parent
        if args.command != "validate"
        else PROJECT_ROOT,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        arguments, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    for name in arguments:
        setattr(
            args,
            name,
            [entry.path for entry in paths.declared_many(name)]
            if name == "request"
            else paths.declared(name).path,
        )
    if args.command == "validate":
        version = load_dataset_version(args.dataset_version, verify_images=True)
        print(f"verified {version['dataset_version']}", flush=True)
        return
    root = args.output_root.resolve()
    if args.command in {"render-colab", "fetch"}:
        if (
            args.command == "render-colab"
            and args.request is not None
            and len(args.request) != 1
        ):
            raise ValueError(
                "Colab durability requires exactly one frozen production request."
            )
        remote = SkillDriveFiles(args.drive_path)
        if args.command == "fetch":
            path = fetch_completed_dataset(
                root, remote, request_sha256=args.request_sha256
            )
            print(f"verified Drive dataset {path}", flush=True)
            return
        request_path = (
            args.request[0]
            if args.request is not None
            else restore_inputs(root, remote, request_sha256=args.request_sha256)
        )
        request_value = read_json(request_path)
        if request_value["schema"] != PRODUCTION_REQUEST_SCHEMA:
            raise ValueError(
                "Colab generation requires an explicit production recipe request."
            )
        telemetry = NvidiaL4Telemetry()
        colab_backend = ColabNHTRenderer(root, fingerprint=request_value["renderer"])
        generation = DurableGeneration(
            root,
            request_path,
            remote,
            telemetry,
            retry_staging=args.retry_staging,
            stop_after_caches=args.stop_after_caches,
        )
        version_path = generation.run(colab_backend)
        print(
            f"published Drive dataset {version_path}"
            if version_path is not None
            else "interrupted after verified Drive acknowledgement; no dataset version published",
            flush=True,
        )
        return
    if args.command == "render":
        paths = []
        for request_path in args.request:
            request = read_json(request_path)
            backend = LocalNHTRenderer(
                root, fingerprint=request["renderer"], batch_size=args.batch_size
            )
            published_path = execute_request(root, request_path, backend)
            paths.append(str(published_path))
            print(f"published {published_path}", flush=True)
        (root / "review-catalog.json").write_bytes(
            canonical_bytes(
                {"schema": "court_scale_review_catalog_v1", "dataset_versions": paths}
            )
        )
        return
    if root.is_relative_to(
        args.scenes_root.resolve()
    ) or args.scenes_root.resolve().is_relative_to(root):
        raise ValueError("Output root must be outside the read-only scene root.")
    if (args.asset_manifest is None) != (args.asset_manifest_sha256 is None):
        raise ValueError(
            "--asset-manifest and --asset-manifest-sha256 are required together."
        )
    if args.asset_manifest is not None:
        asset_root = args.asset_manifest.parent.resolve()
        if root.is_relative_to(asset_root) or asset_root.is_relative_to(root):
            raise ValueError("Output root must be outside the read-only asset bundle.")
    production_recipe = None
    if args.production_recipe is not None:
        if (
            args.recipe is not None
            or args.quality is not None
            or args.method_prefix
            or args.method_density
            or args.legacy_hull_center
        ):
            raise ValueError("Production recipe cannot use comparison overrides.")
        production_recipe = ProductionRecipe.from_dict(
            read_json(args.production_recipe)
        )
        args.methods = [production_recipe.method.value]
        recipe, quality = production_recipe.geometry, production_recipe.quality
    else:
        if args.recipe is None or args.quality is None:
            raise ValueError(
                "Comparison methods require explicit --recipe and --quality."
            )
        recipe = GeometryRecipe.from_dict(read_json(args.recipe))
        quality = QualityPolicy(**read_json(args.quality))
    if len(args.scenes) != len(set(args.scenes)) or len(args.methods) != len(
        set(args.methods)
    ):
        raise ValueError("Duplicate scene or method selections.")
    overrides = {}
    for text in args.legacy_hull_center:
        scene_id, value = text.split("=", 1)
        if (
            value not in {"false", "true"}
            or scene_id in overrides
            or scene_id not in args.scenes
        ):
            raise ValueError("Invalid explicit legacy hull-centre addition.")
        overrides[scene_id] = value == "true"
    prefixes = {
        TrajectoryMethod(k): int(v)
        for k, v in (text.split("=", 1) for text in args.method_prefix)
    }
    densities = {
        TrajectoryMethod(k): int(v)
        for k, v in (text.split("=", 1) for text in args.method_density)
    }
    if len(prefixes) != len(args.method_prefix) or len(densities) != len(
        args.method_density
    ):
        raise ValueError("Duplicate method-specific prefix/density settings.")
    if set(prefixes).union(densities) - {TrajectoryMethod(m) for m in args.methods}:
        raise ValueError("Method-specific settings reference an unselected method.")
    if any(Path(sid).name != sid or sid in {".", ".."} for sid in args.scenes):
        raise ValueError("Invalid scene ID.")
    scene_prefixes = {}
    for text in args.scene_prefix:
        selection, value = text.split("=", 1)
        method_text, scene_id = selection.split(":", 1)
        key = (TrajectoryMethod(method_text), scene_id)
        if (
            key in scene_prefixes
            or method_text not in args.methods
            or scene_id not in args.scenes
        ):
            raise ValueError("Invalid or duplicate scene-specific prefix setting.")
        scene_prefixes[key] = int(value)
    if overrides and TrajectoryMethod.GRID_BUDGET_REPLAY not in args.methods:
        raise ValueError("Legacy config additions require saved-preset replay.")
    fingerprint = renderer_fingerprint()
    inputs_by_replay: dict[bool, list[dict[str, Any]]] = {}
    requests = []
    for text in args.methods:
        method = TrajectoryMethod(text)
        replay = method is TrajectoryMethod.GRID_BUDGET_REPLAY
        if replay not in inputs_by_replay:
            inputs_by_replay[replay] = [
                load_scene_inputs(
                    args.scenes_root / sid,
                    replay=replay,
                    legacy_hull_center=overrides.get(sid) if replay else None,
                    asset_manifest=args.asset_manifest,
                    asset_manifest_sha256=args.asset_manifest_sha256,
                )
                for sid in args.scenes
            ]
        inputs = inputs_by_replay[replay]
        plans = [
            persist_candidate_plan(
                root,
                scene_id=i["scene_id"],
                method=method,
                recipe=recipe,
                seed=args.generation_seed,
                trajectory_count=scene_prefixes.get(
                    (method, i["scene_id"]), prefixes.get(method, args.trajectory_count)
                ),
                density_level=densities.get(method, args.density_level),
                cameras=i["cameras"],
                layout=i["layout"],
                replay_configuration=i["replay_configuration"]
                if method is TrajectoryMethod.GRID_BUDGET_REPLAY
                else None,
            )
            for i in inputs
        ]
        frozen_path = freeze_request(
            root,
            plans=plans,
            assets=[i["asset"] for i in inputs],
            quality=quality,
            renderer=fingerprint,
            production_recipe=production_recipe,
        )
        requests.append(str(frozen_path))
        print(f"frozen {method.value}: {frozen_path}", flush=True)
    (root / "planned-requests.json").write_bytes(
        canonical_bytes({"schema": "court_scale_request_list_v1", "requests": requests})
    )


if __name__ == "__main__":
    main()
