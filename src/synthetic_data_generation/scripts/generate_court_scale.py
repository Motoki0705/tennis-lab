"""Freeze explicit Court method comparisons, then render them through the queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    QualityPolicy,
    TrajectoryMethod,
    canonical_bytes,
)
from src.synthetic_data_generation.dataset.court.scale.publication import (
    execute_request,
    freeze_request,
    load_dataset_version,
    persist_candidate_plan,
    read_json,
)
from src.synthetic_data_generation.dataset.court.scale.runtime import (
    LocalNHTRenderer,
    load_scene_inputs,
    renderer_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--scenes-root", type=Path, required=True)
    plan.add_argument("--scenes", nargs="+", required=True)
    plan.add_argument(
        "--methods", nargs="+", choices=list(TrajectoryMethod), required=True
    )
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--recipe", type=Path, required=True)
    plan.add_argument("--quality", type=Path, required=True)
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
    check = commands.add_parser("validate")
    check.add_argument("--dataset-version", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        version = load_dataset_version(args.dataset_version, verify_images=True)
        print(f"verified {version['dataset_version']}", flush=True)
        return
    root = args.output_root.resolve()
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
    if len(args.scenes) != len(set(args.scenes)) or len(args.methods) != len(
        set(args.methods)
    ):
        raise ValueError("Duplicate scene or method selections.")
    recipe = GeometryRecipe.from_dict(read_json(args.recipe))
    quality = QualityPolicy(**read_json(args.quality))
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
    fingerprint = renderer_fingerprint()
    inputs = [
        load_scene_inputs(
            args.scenes_root / sid,
            replay=TrajectoryMethod.GRID_BUDGET_REPLAY in args.methods,
            legacy_hull_center=overrides.get(sid),
        )
        for sid in args.scenes
    ]
    requests = []
    for text in args.methods:
        method = TrajectoryMethod(text)
        plans = [
            persist_candidate_plan(
                root,
                scene_id=i["scene_id"],
                method=method,
                recipe=recipe,
                seed=args.generation_seed,
                trajectory_count=prefixes.get(method, args.trajectory_count),
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
        )
        requests.append(str(frozen_path))
        print(f"frozen {method.value}: {frozen_path}", flush=True)
    (root / "planned-requests.json").write_bytes(
        canonical_bytes({"schema": "court_scale_request_list_v1", "requests": requests})
    )


if __name__ == "__main__":
    main()
