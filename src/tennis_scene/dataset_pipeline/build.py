"""Run observation, reconstruction, publication, and RGB feature stages."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline.checkpoint_integrity import (
    verify_checkpoint_integrity,
)
from src.tennis_scene.dataset_pipeline.configuration import (
    DatasetBuildConfig,
    resolved_recipe,
)
from src.tennis_scene.dataset_pipeline.court import observe_static_court
from src.tennis_scene.dataset_pipeline.people import (
    observe_singles_people,
    validate_people_receipts,
)
from src.tennis_scene.dataset_pipeline.provenance import (
    scene_identity,
    validated_scene_cache,
)
from src.tennis_scene.dataset_pipeline.quality import evaluate_reconstruction
from src.tennis_scene.dataset_pipeline.refinement import (
    RefinementSettings,
    check_label_coverage,
    refine_scene,
)
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    SceneRunner,
    generate_pseudo_annotations,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.tennis_scene.reference_pipeline.observations import (
    import_ball,
    sha256,
)
from src.tennis_scene.reference_pipeline.reconstruction import reconstruct
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import FileIntegrityError
from src.utils.io import save_json_atomic


def materialize_dataset(
    source: Path, destination: Path, *, clip_ids: tuple[str, ...] | None = None
) -> None:
    """Copy manifests and hard-link immutable media into an independent version."""
    manifest = load_dataset_manifest(source)
    if clip_ids is not None:
        if not clip_ids or set(clip_ids) - manifest.clips.keys():
            raise ValueError("Materialized dataset must select known nonempty clip IDs")
        manifest = replace(
            manifest, clips={key: manifest.clips[key] for key in clip_ids}
        )
    destination.mkdir(parents=True, exist_ok=True)
    for record in manifest.clips.values():
        clip = ClipManifest.load(source / record.path)
        target = destination / record.path
        target.mkdir(parents=True, exist_ok=True)
        source_manifest, target_manifest = (
            source / record.path / "clip.json",
            target / "clip.json",
        )
        if target_manifest.exists():
            if sha256(source_manifest) != sha256(target_manifest):
                raise ValueError(f"Input manifest changed: {record.clip_id}")
        else:
            shutil.copy2(source_manifest, target_manifest)
        for camera, relative in zip(clip.camera_ids, clip.video_paths, strict=True):
            media = clip.media_path(camera)
            linked = target / relative
            linked.parent.mkdir(parents=True, exist_ok=True)
            if linked.exists():
                if not os.path.samefile(linked, media) and sha256(linked) != sha256(
                    media
                ):
                    raise ValueError(f"Input video changed: {media}")
            else:
                os.link(media, linked)
    target_index = destination / "dataset.json"
    if target_index.exists():
        if json.loads(target_index.read_text()) != manifest.to_dict():
            raise ValueError(
                "Input dataset inventory changed; choose a new dataset version"
            )
    else:
        manifest.save(destination)


def _court(
    cfg: DictConfig, runtime: DatasetBuildConfig, clip: ClipManifest, output: Path
) -> tuple[np.ndarray, np.ndarray]:
    cache, receipt = output / "court.npz", output / "court.json"
    identity = {
        "checkpoint_sha256": sha256(runtime.court.checkpoint),
        "clip_sha256": sha256(clip.clip_dir / "clip.json"),
        "video_sha256": {cam: sha256(clip.media_path(cam)) for cam in clip.camera_ids},
        "ball_annotation_sha256": {
            cam: sha256(clip.clip_dir / "outsource" / f"{cam}_annotations.json")
            for cam in clip.camera_ids
        }
        if runtime.ball_source == "outsource"
        else {
            "saved_scene": sha256(
                clip.clip_dir / "observations/ball_import.metadata.json"
            )
        },
        "settings": OmegaConf.to_container(cfg.court, resolve=True),
    }
    if cache.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity:
            changed = [
                key for key in identity if saved["identity"].get(key) != identity[key]
            ]
            raise ValueError(
                f"Stale court observations at {output}; changed fields: {changed}"
            )
        with np.load(cache) as arrays:
            return arrays["keypoints"], arrays["homographies"]
    kp, homographies, diagnostics = observe_static_court(
        clip,
        output,
        resolver=runtime.resolver,
        settings=runtime.court,
        device=str(cfg.device),
    )
    np.savez_compressed(cache, keypoints=kp, homographies=homographies)
    save_json_atomic({"identity": identity, "diagnostics": diagnostics}, receipt)
    torch.cuda.empty_cache()
    return kp, homographies


def _import_ball(runtime: DatasetBuildConfig, clip: ClipManifest, output: Path) -> None:
    if runtime.ball_source == "outsource":
        import_ball(clip.clip_dir, output)
        return
    source = clip.clip_dir / "observations"
    receipt = json.loads((source / "ball_import.metadata.json").read_text())
    if receipt["video_sha256"] != {
        name: sha256(clip.clip_dir / name) for name in clip.video_paths
    }:
        raise ValueError(f"Saved ball media changed for {clip.clip_id}")
    ball = BallDetectionResult.load(source / "ball_detection_result.json")
    valid, errors = ball.validate()
    if not valid or ball.ball_uv.shape != (len(clip.camera_ids), clip.num_frames, 2):
        raise ValueError(f"Invalid saved ball for {clip.clip_id}: {errors}")
    for name in ("ball_detection_result.json", "ball_import.metadata.json"):
        target = output / name
        if target.exists():
            if sha256(source / name) != sha256(target):
                raise ValueError(f"Saved ball source changed: {target}")
        else:
            shutil.copyfile(source / name, target)


def _observe_court(
    cfg: DictConfig,
    runtime: DatasetBuildConfig,
    clip: ClipManifest,
    output: Path,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_id = runtime.calibration_clips.get(clip.video_id, clip.clip_id)
    if calibration_id == clip.clip_id:
        return _court(cfg, runtime, clip, output)
    manifest = load_dataset_manifest(runtime.source)
    reference = ClipManifest.load(runtime.source / manifest.clips[calibration_id].path)
    if (clip.width, clip.height, clip.camera_ids) != (
        reference.width,
        reference.height,
        reference.camera_ids,
    ):
        raise ValueError(
            "Static camera propagation requires matching image/camera layouts"
        )
    for current, selected in zip(clip.cameras, reference.cameras, strict=True):
        if any(current.get(k) != selected.get(k) for k in ("source_path", "letterbox")):
            raise ValueError(
                "Static camera propagation cannot cross source/crop changes"
            )
    reference_output = runtime.observations / calibration_id
    reference_output.mkdir(parents=True, exist_ok=True)
    _import_ball(runtime, reference, reference_output)
    keypoints, matrices = _court(cfg, runtime, reference, reference_output)
    keypoints = np.repeat(keypoints[:, :1], clip.num_frames, axis=1)
    receipt = {
        **json.loads((reference_output / "court.json").read_text()),
        "calibration_clip_id": calibration_id,
        "target_manifest_sha256": clip.digest(),
        "propagation_assumption": "fixed cameras within the same source recording and letterbox",
    }
    target_receipt = output / "court.json"
    if target_receipt.exists() and json.loads(target_receipt.read_text()) != receipt:
        raise ValueError(f"Court calibration assignment changed: {clip.clip_id}")
    if not target_receipt.exists():
        np.savez_compressed(
            output / "court.npz", keypoints=keypoints, homographies=matrices
        )
        save_json_atomic(receipt, target_receipt)
    return keypoints, matrices


def build_dataset(cfg: DictConfig) -> None:
    runtime = DatasetBuildConfig.from_config(cfg)
    paths = ReferenceClipPaths.from_config(cfg)
    required_assets: dict[str, Path] = {}
    if runtime.stage in {"all", "infer"}:
        required_assets.update(plcs=paths.plcs_checkpoint, blcs=paths.blcs_checkpoint)
    if runtime.stage in {"all", "observe"}:
        required_assets.update(
            dino=paths.dino_checkpoint, vitpose=paths.vitpose_checkpoint
        )
    if runtime.stage in {"all", "court", "observe", "infer"}:
        required_assets["court"] = runtime.court.checkpoint
    if runtime.stage in {"all", "features"} and runtime.features_enabled:
        required_assets["dinov3"] = runtime.feature_checkpoint
    for asset in required_assets.values():
        if not asset.is_file():
            raise FileNotFoundError(
                f"Required dataset producer checkpoint is missing: {asset}"
            )
    verified_checkpoints = verify_checkpoint_integrity(
        required_assets, runtime.checkpoint_sha256
    )
    cv2.setRNGSeed(runtime.seed)
    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    runtime.output.mkdir(parents=True, exist_ok=True)
    recipe = resolved_recipe(cfg, runtime)
    recipe_file = runtime.output / "recipe.json"
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError(
            "Generation recipe changed; choose a new output_dir and dataset version"
        )
    save_json_atomic(recipe, recipe_file)
    if runtime.checkpoint_sha256 is not None:
        save_json_atomic(
            {
                "stage": runtime.stage,
                "expected": runtime.checkpoint_sha256,
                "verified": verified_checkpoints,
            },
            runtime.output / "checkpoint_verification.json",
        )
    OmegaConf.save(
        OmegaConf.create(recipe), runtime.output / "config.yaml", resolve=True
    )
    manifest = load_dataset_manifest(runtime.source)
    if runtime.stage in {"all", "infer", "features"}:
        materialize_dataset(
            runtime.source, runtime.destination, clip_ids=runtime.dataset_clip_ids
        )
    failures = {}
    for clip_id in runtime.clip_ids:
        record = manifest.clips[clip_id]
        clip = ClipManifest.load(runtime.source / record.path)
        output = runtime.output / clip_id
        output.mkdir(parents=True, exist_ok=True)
        if runtime.stage == "features":
            break
        try:
            _process_clip(cfg, runtime, clip, output)
        except FileIntegrityError as error:
            failures[clip_id] = f"{type(error).__name__}: {error}"
            save_json_atomic(
                {"stage": runtime.stage, "failures": failures},
                runtime.output / "failures.json",
            )
            raise
        except Exception as error:
            failures[clip_id] = f"{type(error).__name__}: {error}"
            print(f"FAILED {clip_id}: {failures[clip_id]}", flush=True)
            torch.cuda.empty_cache()
        save_json_atomic(
            {"stage": runtime.stage, "failures": failures},
            runtime.output / "failures.json",
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} clips failed; successful caches retained. See {runtime.output / 'failures.json'}"
        )
    if runtime.stage in {"all", "features"} and runtime.features_enabled:
        _precompute_features(cfg, runtime)


def _process_clip(
    cfg: DictConfig, runtime: DatasetBuildConfig, clip: ClipManifest, output: Path
) -> None:
    observations = runtime.observations / clip.clip_id
    observations.mkdir(parents=True, exist_ok=True)
    _import_ball(runtime, clip, observations)
    kp, homographies = _observe_court(cfg, runtime, clip, observations)
    if runtime.stage == "court":
        return
    local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if not isinstance(local, DictConfig):
        raise TypeError("Dataset configuration must be a mapping")
    local.clip_dir = str(clip.clip_dir.relative_to(runtime.resolver.roots.data_root))
    local.output_dir = str(output.relative_to(runtime.resolver.roots.output_root))
    paths = ReferenceClipPaths.from_config(local)
    if runtime.stage in {"all", "observe"}:
        observe_singles_people(
            local,
            paths,
            clip.clip_dir,
            observations,
            homographies=homographies,
            checkpoint_sha256=runtime.checkpoint_sha256,
        )
    if runtime.stage == "observe":
        return
    if runtime.stage == "infer":
        validate_people_receipts(
            clip.camera_ids, observations, checkpoint_sha256=runtime.checkpoint_sha256
        )
    identity = scene_identity(local, paths, clip, observations)
    destination_manifest = load_dataset_manifest(runtime.destination)
    target_clip = ClipManifest.load(
        runtime.destination / destination_manifest.clips[clip.clip_id].path
    )
    if validated_scene_cache(target_clip, identity):
        print(f"Using validated reconstruction {clip.clip_id}", flush=True)
        return
    outcomes = generate_pseudo_annotations(
        runtime.destination,
        _scene_runner(
            local, paths, clip, output, kp, homographies, observations, identity
        ),
        pipeline_config_yaml=OmegaConf.to_yaml(local, resolve=True),
        clip_ids=[clip.clip_id],
        overwrite=False,
        continue_on_error=False,
    )
    print(f"Published {clip.clip_id}: {outcomes[0].status}", flush=True)


def _scene_runner(
    cfg: DictConfig,
    paths: ReferenceClipPaths,
    clip: ClipManifest,
    output: Path,
    kp: np.ndarray,
    homographies: np.ndarray,
    observations: Path,
    identity: dict[str, object],
) -> SceneRunner:
    def runner(videos: Sequence[Path], cameras: Sequence[str]) -> SceneResult:
        if tuple(cameras) != clip.camera_ids or len(videos) != len(clip.camera_ids):
            raise ValueError("Publication inputs differ from the reconstructed clip")
        scene = reconstruct(
            cfg,
            paths,
            clip.clip_dir,
            output,
            court_observations=(kp, homographies),
            observation_directory=observations,
            coordinate_mode=cast(Literal["reference", "physical"], cfg.coordinate_mode),
        )
        scene.metadata["court_model"] = json.loads(
            (observations / "court.json").read_text()
        )
        scene.metadata["dataset_producer_identity"] = identity
        report, diagnostics = evaluate_reconstruction(
            scene,
            homographies,
            list(cfg.view_half_turns)
            if cfg.coordinate_mode == "reference"
            else [False],
        )
        settings = RefinementSettings.from_config(cfg.refinement)
        if settings.enabled:
            save_json_atomic(report, output / "raw_model_quality.json")
            evidence = refine_scene(scene, homographies, settings)
            save_json_atomic(evidence, output / "label_evidence.json")
            report, diagnostics = evaluate_reconstruction(
                scene,
                homographies,
                list(cfg.view_half_turns)
                if cfg.coordinate_mode == "reference"
                else [False],
            )
            save_scene_result(scene, output / "refined_scene.npz")
            check_label_coverage(evidence, settings)
        save_json_atomic(report, output / "quality.json")
        np.savez_compressed(
            output / "quality_arrays.npz", allow_pickle=False, **diagnostics
        )
        scene.metadata["quality"] = report
        print(json.dumps(report), flush=True)
        return scene

    return runner


def _precompute_features(cfg: DictConfig, runtime: DatasetBuildConfig) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from src.tasks.slcs.configuration import SLCSPrecomputeConfig
    from src.tasks.slcs.data.dino_precompute import precompute_clip_tokens
    from src.tasks.slcs.model_io.factory import create_slcs_frame_token_encoder
    from src.tennis_scene.dataset_pipeline.features import validated_feature_cache
    from src.utils.paths import PROJECT_ROOT

    # Hydra's application singleton is already active in a CLI invocation.
    # Compose from the SLCS config directory in a temporary context and restore it.
    state = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tasks/slcs/configs")
        ):
            settings = compose(config_name="precompute_dino_tokens")
    finally:
        GlobalHydra.instance().hydra = state
    settings.paths = dict(runtime.resolver.roots.as_mapping())
    settings.paths.checkpoint_root = str(runtime.resolver.roots.external_asset_root)
    settings.data.dataset_root = str(
        runtime.destination.relative_to(runtime.resolver.roots.data_root)
    )
    settings.precompute.checkpoint_path = str(
        runtime.feature_checkpoint.relative_to(
            runtime.resolver.roots.external_asset_root
        )
    )
    settings.precompute.device = cfg.device
    feature_runtime = SLCSPrecomputeConfig.from_config(settings)
    spec = feature_runtime.data.pipeline.dino_spec
    manifest = load_dataset_manifest(runtime.destination)
    encoder = None
    for clip_id in runtime.clip_ids:
        clip = ClipManifest.load(runtime.destination / manifest.clips[clip_id].path)
        identity: dict[str, object] = {
            "script": "src/tennis_scene/scripts/build_slcs_dataset.py",
            "checkpoint_sha256": sha256(runtime.feature_checkpoint),
            "video_sha256": {
                cam: sha256(clip.media_path(cam)) for cam in clip.camera_ids
            },
        }
        if validated_feature_cache(clip, spec, identity):
            print(f"Using validated RGB features {clip_id}", flush=True)
            continue
        if encoder is None:
            encoder = create_slcs_frame_token_encoder(feature_runtime)
        precompute_clip_tokens(
            clip,
            encoder,
            spec,
            batch_size=feature_runtime.batch_size,
            overwrite=False,
            generator=identity,
        )
        print(f"Saved RGB features {clip_id}", flush=True)
