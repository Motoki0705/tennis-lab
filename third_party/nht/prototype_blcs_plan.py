#!/usr/bin/env python3
"""Generate an immutable BLCS plan from real physics and an approved alignment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.alignment.artifacts.acceptance_decision import (  # noqa: E402
    load_alignment_acceptance_decision,
    verify_machine_evidence,
)
from src.synthetic_data_generation.alignment.artifacts.calibration import (  # noqa: E402
    load_calibration_artifact,
)
from src.synthetic_data_generation.alignment.artifacts.holdout_validation import (  # noqa: E402
    load_holdout_validation_artifact,
)
from src.synthetic_data_generation.alignment.scene_provider.bundle import (  # noqa: E402
    load_scene_provider_bundle,
)
from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    load_ball_asset_registry,
)
from src.synthetic_data_generation.blcs.planner import (  # noqa: E402
    build_blcs_gaussian_plan_from_scene,
    verify_blcs_gaussian_plan_output,
    write_blcs_gaussian_plan,
)
from src.synthetic_data_generation.scene_contract import (  # noqa: E402
    load_scene_contract,
)
from src.tasks.base.generate_dataset.timeline_composer import (
    TimelineConfig,  # noqa: E402
)
from src.tasks.blcs.generate_dataset.config import (  # noqa: E402
    build_default_generator_config,
)
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (  # noqa: E402
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import (  # noqa: E402
    BLCSSceneData,
    BLCSSceneGenerator,
)
from src.utils.seeding import seed_everything  # noqa: E402

PROTOTYPE_SIMULATION_SCHEMA = "tennis_blcs_prototype_simulation_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--provider-bundle", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--scene-contract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--from-cell", type=int, default=0)
    parser.add_argument("--side", choices=("near", "far"), default="near")
    parser.add_argument("--multi-num-frames", type=int, default=240)
    parser.add_argument("--multi-num-balls", type=int, default=2)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "uri": resolved.as_uri(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _relative_ref(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_verified_reference(
    root: Path,
    *,
    uri: str,
    sha256: str,
    size_bytes: int,
) -> Path:
    relative = PurePosixPath(uri)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"Alignment artifact URI must be safe and relative: {uri!r}")
    path = (root / Path(*relative.parts)).resolve()
    path.relative_to(root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size_bytes or _sha256_file(path) != sha256:
        raise ValueError(f"Alignment artifact bytes differ: {path}")
    return path


def _load_approved_scene(
    *,
    provider_path: Path,
    contract_path: Path,
    contract_root: Path,
) -> tuple[object, object, Path]:
    provider = load_scene_provider_bundle(provider_path, verify_files=True)
    contract = load_scene_contract(contract_path)
    if contract.alignment is None or not contract.alignment.accepted:
        raise ValueError("Prototype rendering requires an accepted court alignment.")
    if contract.scene_fingerprint != provider.manifest.scene_fingerprint:
        raise ValueError("Approved alignment and export scene fingerprints differ.")
    if contract.cameras != provider.manifest.cameras:
        raise ValueError("Approved alignment cameras differ from the verified export.")
    decision_path = _resolve_verified_reference(
        contract_root,
        uri=contract.alignment.manifest.uri,
        sha256=contract.alignment.manifest.sha256,
        size_bytes=contract.alignment.manifest.size_bytes,
    )
    decision, _ = load_alignment_acceptance_decision(decision_path)
    if decision.selected_court_cluster != contract.alignment.selected_court_cluster:
        raise ValueError("Alignment court cluster differs from its user decision.")
    if decision.selected_symmetry != contract.alignment.selected_symmetry:
        raise ValueError("Alignment symmetry differs from its user decision.")
    calibration_path = _resolve_verified_reference(
        contract_root,
        uri=decision.calibration.uri,
        sha256=decision.calibration.sha256,
        size_bytes=decision.calibration.size_bytes,
    )
    holdout_path = _resolve_verified_reference(
        contract_root,
        uri=decision.holdout_validation.uri,
        sha256=decision.holdout_validation.sha256,
        size_bytes=decision.holdout_validation.size_bytes,
    )
    verify_machine_evidence(
        decision,
        calibration=load_calibration_artifact(calibration_path),
        holdout_validation=load_holdout_validation_artifact(holdout_path),
    )
    return provider, contract, decision_path


def _generate_scene(args: argparse.Namespace) -> BLCSSceneData:
    seed_everything(args.seed)
    generator = BLCSSceneGenerator(
        config=build_default_generator_config(),
        device="cpu",
    )
    if args.mode == "single":
        scene = generator.generate_scene(
            args.from_cell,
            args.side,
            f"prototype-physics-single-{args.seed}",
        )
        if scene is None:
            raise RuntimeError("BLCS physical generator returned no single scene.")
        return scene
    if args.multi_num_frames < 64:
        raise ValueError("multi-num-frames must be at least 64.")
    if not 2 <= args.multi_num_balls <= 4:
        raise ValueError("multi-num-balls must lie in [2, 4].")
    timeline = TimelineConfig(
        num_frames=args.multi_num_frames,
        min_tracks=args.multi_num_balls,
        max_tracks=args.multi_num_balls,
        max_concurrent=args.multi_num_balls,
        min_reuse_gap_frames=4,
        start_index_range=(-32, args.multi_num_frames - 32),
        min_active_frames=32,
        overlap_probability=1.0,
        min_gap_frames=8,
        max_gap_frames=32,
    )
    multi = MultiBallSceneGenerator(
        generator,
        timeline=timeline,
        rng=random.Random(args.seed),
    )
    return multi.generate_scene(f"prototype-physics-multi-{args.seed}")


def _trajectory_metrics(scene: BLCSSceneData) -> dict[str, object]:
    positions = scene.ball_pos_world.detach().cpu().numpy()
    velocities = scene.ball_vel_world.detach().cpu().numpy()
    if positions.ndim == 2:
        present = np.ones(positions.shape[:-1], dtype=np.bool_)
    else:
        if scene.ball_present is None:
            raise ValueError("Multi-object scene omitted presence.")
        present = scene.ball_present.detach().cpu().numpy()[:, : scene.num_balls]
        positions = positions[:, : scene.num_balls]
        velocities = velocities[:, : scene.num_balls]
    active_positions = positions[present]
    active_velocities = velocities[present]
    if active_positions.size == 0:
        raise ValueError("Physical simulation produced no active ball frames.")
    return {
        "frame_count": int(positions.shape[0]),
        "object_count": int(scene.num_balls),
        "active_object_frames": int(present.sum()),
        "duration_seconds": float(positions.shape[0] / scene.fps_out),
        "position_min_m": [float(value) for value in active_positions.min(axis=0)],
        "position_max_m": [float(value) for value in active_positions.max(axis=0)],
        "maximum_height_m": float(active_positions[:, 2].max()),
        "maximum_speed_mps": float(
            np.linalg.norm(active_velocities, axis=1).max()
        ),
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.seed < 0:
        raise SystemExit("seed must be non-negative.")
    registry_path = args.registry.resolve()
    provider_path = args.provider_bundle.resolve()
    contract_path = args.scene_contract.resolve()
    contract_root = args.scene_contract_root.resolve()
    registry = load_ball_asset_registry(
        registry_path,
        verify_local_artifacts=True,
    )
    provider, contract, decision_path = _load_approved_scene(
        provider_path=provider_path,
        contract_path=contract_path,
        contract_root=contract_root,
    )
    scene = _generate_scene(args)
    plan = build_blcs_gaussian_plan_from_scene(
        scene,
        registry=registry,
        seed=args.seed,
        scene_from_court=contract.alignment.scene_from_court,
        cameras=contract.cameras,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        plan_root = temporary / "plan"
        plan_manifest = write_blcs_gaussian_plan(plan_root, plan)
        plan_report = verify_blcs_gaussian_plan_output(plan_root)
        unsigned: dict[str, object] = {
            "schema": PROTOTYPE_SIMULATION_SCHEMA,
            "status": "passed",
            "mode": args.mode,
            "seed": args.seed,
            "simulator": {
                "class": (
                    "BLCSSceneGenerator"
                    if args.mode == "single"
                    else "MultiBallSceneGenerator"
                ),
                "physics": "BallPhysics",
                "rally": "RallySimulator",
                "fps_out": scene.fps_out,
                "sim_fps": scene.sim_fps,
                "physics_config": scene.physics_config_dict,
                "court_config": scene.court_config_dict,
            },
            "scene": {
                "scene_id": scene.scene_id,
                "initial_from_cell": scene.initial_from_cell,
                "initial_from_side": scene.initial_from_side,
                "rally_length": scene.rally_length,
                "end_reason": scene.end_reason,
                "winner_side": scene.winner_side,
                "shots": scene.shots,
                "track_instances": scene.track_instances,
            },
            "trajectory_metrics": _trajectory_metrics(scene),
            "inputs": {
                "registry": _file_ref(registry_path),
                "provider_bundle": _file_ref(
                    provider_path / "provider.json"
                    if provider_path.is_dir()
                    else provider_path
                ),
                "provider_bundle_fingerprint": (
                    provider.manifest.bundle_fingerprint
                ),
                "scene_contract": _file_ref(contract_path),
                "scene_fingerprint": contract.scene_fingerprint,
                "alignment_id": contract.alignment.alignment_id,
                "alignment_decision": _file_ref(decision_path),
            },
            "plan": {
                **_relative_ref(temporary, plan_manifest),
                **plan_report,
            },
            "rgb_overlay_used": False,
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        with (temporary / "simulation.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
