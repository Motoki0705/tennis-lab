"""Strict configuration for producing a versioned SLCS real-video dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import exact_config_mapping
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline.court import StaticCourtSettings
from src.tennis_scene.dataset_pipeline.person_association import association_settings
from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest
from src.utils.configuration import PathResolver, PathRole

_KEYS = frozenset(
    {
        "paths",
        "clip_dir",
        "output_dir",
        "device",
        "stage",
        "reference_camera",
        "view_half_turns",
        "people",
        "plcs_checkpoint",
        "blcs_checkpoint",
        "window_size",
        "window_overlap",
        "sample_stride",
        "pose_visibility_threshold",
        "dataset_directory",
        "dataset_output_directory",
        "clip_ids",
        "seed",
        "court",
        "features",
        "observation_directory",
        "ball_source",
        "court_calibration_clips",
        "coordinate_mode",
        "refinement",
        "excluded_clips",
        "dataset_clip_ids",
    }
)


@dataclass(frozen=True)
class DatasetBuildConfig:
    resolver: PathResolver
    source: Path
    destination: Path
    output: Path
    clip_ids: tuple[str, ...]
    stage: str
    seed: int
    court: StaticCourtSettings
    features_enabled: bool
    feature_checkpoint: Path
    observations: Path
    ball_source: str
    calibration_clips: dict[str, str]
    dataset_clip_ids: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: DictConfig) -> DatasetBuildConfig:
        exact_config_mapping(cfg, path="configuration", required_keys=_KEYS)
        from src.tennis_scene.dataset_pipeline.refinement import RefinementSettings

        RefinementSettings.from_config(cfg.refinement)
        if cfg.coordinate_mode not in {"reference", "physical"}:
            raise ValueError("coordinate_mode must be reference or physical")
        if cfg.coordinate_mode == "physical" and (
            cfg.reference_camera is not None or cfg.view_half_turns is not None
        ):
            raise ValueError(
                "physical mode forbids reference camera and view half-turns"
            )
        paths = ReferenceClipPaths.from_config(cfg)
        resolver = paths.resolver
        source = resolver.resolve(PathRole.DATA, str(cfg.dataset_directory))
        destination = resolver.resolve(PathRole.DATA, str(cfg.dataset_output_directory))
        if (
            source == destination
            or destination.is_relative_to(source)
            or source.is_relative_to(destination)
        ):
            raise ValueError("Use a separate versioned output dataset directory")
        manifest = load_dataset_manifest(source)
        if cfg.ball_source not in {"outsource", "saved_scene"}:
            raise ValueError(
                "ball_source must explicitly select outsource or saved_scene"
            )
        calibration = (
            {}
            if cfg.court_calibration_clips is None
            else dict(cfg.court_calibration_clips)
        )
        if calibration and set(calibration) != {
            r.video_id for r in manifest.clips.values()
        }:
            raise ValueError(
                "court_calibration_clips must cover every video or be empty"
            )
        for video, clip_id in calibration.items():
            if (
                clip_id not in manifest.clips
                or manifest.clips[clip_id].video_id != video
            ):
                raise ValueError(f"Invalid calibration clip {video}: {clip_id}")
        excluded = dict(cfg.excluded_clips)
        if set(excluded) - manifest.clips.keys() or any(
            type(reason) is not str or not reason.strip()
            for reason in excluded.values()
        ):
            raise ValueError(
                "excluded_clips must map known clip IDs to explicit QC reasons"
            )
        eligible = tuple(sorted(set(manifest.clips) - set(excluded)))
        if cfg.dataset_clip_ids is not None:
            selected = tuple(cfg.dataset_clip_ids)
            if (
                not selected
                or len(set(selected)) != len(selected)
                or set(selected) - set(eligible)
            ):
                raise ValueError("dataset_clip_ids must select unique eligible clips")
            eligible = tuple(sorted(selected))
        clips = tuple(eligible if cfg.clip_ids is None else cfg.clip_ids)
        if not clips or len(set(clips)) != len(clips) or set(clips) - set(eligible):
            raise ValueError("clip_ids must be unique known dataset clip IDs")
        stage = str(cfg.stage)
        if stage not in {"all", "court", "observe", "infer", "features"}:
            raise ValueError(f"Unsupported dataset stage {stage!r}")
        if type(cfg.seed) is not int:
            raise TypeError("seed must be an integer")
        people = exact_config_mapping(
            cfg.people,
            path="people",
            required_keys={
                "dino_checkpoint",
                "dino_repository",
                "vitpose_checkpoint",
                "confidence",
                "short_side",
                "max_long_side",
                "batch_size",
                "detection_stride",
                "court_half_width_m",
                "court_half_length_m",
                "min_sample_coverage",
                "max_gap_seconds",
                "precision",
            },
            optional_keys={"long_gap_policy", "selection_policy", "association"},
        )
        association_settings(people)
        if "long_gap_policy" in people and people["long_gap_policy"] not in {
            "error",
            "mask",
        }:
            raise ValueError("people.long_gap_policy must be error or mask")
        if people["precision"] not in {"float32", "bfloat16"}:
            raise ValueError("people.precision must be float32 or bfloat16")
        for name in ("short_side", "max_long_side", "batch_size", "detection_stride"):
            if type(people[name]) is not int or cast(int, people[name]) <= 0:
                raise ValueError(f"people.{name} must be a positive integer")
        for name in (
            "confidence",
            "court_half_width_m",
            "court_half_length_m",
            "min_sample_coverage",
            "max_gap_seconds",
        ):
            value = people[name]
            if (
                type(value) not in (float, int)
                or not math.isfinite(cast(float, value))
                or cast(float, value) <= 0
            ):
                raise ValueError(f"people.{name} must be positive and finite")
        if cfg.people.confidence >= 1 or cfg.people.min_sample_coverage > 1:
            raise ValueError("Invalid person confidence/coverage threshold")
        raw = exact_config_mapping(
            cfg.court,
            path="court",
            required_keys={
                "checkpoint",
                "samples_per_clip",
                "min_score",
                "min_points",
                "ransac_px",
                "max_fit_error_px",
                "ball_crop_margins",
            },
        )
        for name in ("samples_per_clip", "min_points"):
            if type(raw[name]) is not int or cast(int, raw[name]) <= 0:
                raise ValueError(f"court.{name} must be a positive integer")
        if not 4 <= cfg.court.min_points <= 14 or not 0 < cfg.court.min_score <= 1:
            raise ValueError("Invalid minimum court evidence")
        if cfg.court.ransac_px <= 0 or cfg.court.max_fit_error_px <= 0:
            raise ValueError("Court fit tolerances must be positive")
        margins = dict(cfg.court.ball_crop_margins)
        for camera, margin in margins.items():
            if not isinstance(camera, str) or (
                margin is not None
                and (type(margin) not in (float, int) or not 0 <= margin <= 2)
            ):
                raise ValueError(
                    "court.ball_crop_margins must map camera IDs to null or margins in [0,2]"
                )
        court = StaticCourtSettings(
            resolver.resolve(PathRole.OUTPUT, str(cfg.court.checkpoint)),
            int(cfg.court.samples_per_clip),
            float(cfg.court.min_score),
            int(cfg.court.min_points),
            float(cfg.court.ransac_px),
            float(cfg.court.max_fit_error_px),
            margins,
        )
        exact_config_mapping(
            cfg.features, path="features", required_keys={"enabled", "checkpoint"}
        )
        if type(cfg.features.enabled) is not bool:
            raise TypeError("features.enabled must be boolean")
        return cls(
            resolver,
            source,
            destination,
            paths.output_dir,
            clips,
            stage,
            int(cfg.seed),
            court,
            bool(cfg.features.enabled),
            resolver.resolve(PathRole.EXTERNAL_ASSET, str(cfg.features.checkpoint)),
            resolver.resolve(PathRole.OUTPUT, str(cfg.observation_directory)),
            str(cfg.ball_source),
            calibration,
            eligible,
        )


def validate_build_config(cfg: DictConfig) -> None:
    DatasetBuildConfig.from_config(cfg)


def resolved_recipe(cfg: DictConfig, runtime: DatasetBuildConfig) -> dict[str, object]:
    """Stage and subset are execution selectors, not dataset content identity."""
    result = cast(dict[str, object], OmegaConf.to_container(cfg, resolve=True))
    for name in ("stage", "clip_ids", "clip_dir"):
        result.pop(name)
    result["paths"] = dict(runtime.resolver.roots.as_mapping())
    return result
