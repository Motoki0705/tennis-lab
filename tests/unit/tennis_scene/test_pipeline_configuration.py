"""Strict, headless automatic pipeline configuration and independent data inputs."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from src.tasks.court_side.hypothesis import CourtSideConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.utils.configuration import ConfigurationError, UnknownConfigurationKeyError
from src.utils.configuration.errors import PathContractError


def _runtime(
    overrides: list[str], *, bind_inputs: bool = True
) -> PipelineRuntimeConfig:
    directory = Path(__file__).parents[3] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name="pipeline", overrides=overrides)
    return PipelineRuntimeConfig.from_config(cfg, bind_inputs=bind_inputs)


def test_defaults_associate_players_and_decide_sides() -> None:
    runtime = _runtime([])
    assert runtime.camera_geometry.reference_camera is None
    assert runtime.component_sources["player_association"] == "execute"
    assert runtime.component_sources["court_side"] == "execute"
    assert runtime.court_side_evidence == "ball"
    # The method's file supplies the parameters, the pipeline the game format and the weight location.
    association = runtime.player_association
    assert association.players_per_side == 1 and association.appearance is not None
    assert runtime.association_encoder_weights == runtime.roots.checkpoint_root / "player_association/person_vit_clip_reid.pth"
    doubles = _runtime(["player_association.players_per_side=2"]).player_association
    assert doubles.players_per_side == 2 and doubles.geometry == association.geometry
    assert isinstance(runtime.court_side, CourtSideConfig)
    assert runtime.court_side.reprojection_px > 0 and runtime.court_side.min_margin > 0
    assert "blcs_association" not in runtime.enabled
    assert len(runtime.camera_ids) == 3
    assert runtime.player_placement.temporal_weight == 0.05
    assert runtime.player_placement.min_joints == 5
    assert runtime.people.runtime.dino_detector.precision == "fp32"
    assert runtime.people.runtime.vitpose.precision == "fp32"
    assert runtime.people.runtime.vitpose.flip_test is True


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_person_precision_accepts_explicit_cuda_autocast(precision: str) -> None:
    runtime = _runtime([f"people_models.runtime.{model}.precision={precision}" for model in ("dino_detector", "vitpose")])
    assert runtime.people.runtime.dino_detector.precision == precision
    assert runtime.people.runtime.vitpose.precision == precision


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (["people_models.runtime.dino_detector.precision=half"], "dino_detector.precision"),
        (["people_models.runtime.vitpose.precision=32"], "vitpose.precision"),
        (["~people_models.runtime.vitpose.precision"], "vitpose.precision"),
        (["device=cpu", "people_models.runtime.dino_detector.precision=bf16"], "DINO requires a CUDA device"),
        (["device=cpu", "people_models.runtime.vitpose.precision=fp16"], "ViTPose requires CUDA"),
    ],
)
def test_person_precision_rejects_invalid_configuration(overrides: list[str], error: str) -> None:
    with pytest.raises(ConfigurationError, match=error):
        _runtime(overrides)


def test_saved_six_threshold_court_side_config_restores_clip_mode(tmp_path: Path) -> None:
    directory = Path(__file__).parents[3] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name="pipeline", overrides=["court_side.min_margin=0.31"])
    current = PipelineRuntimeConfig.from_config(cfg)
    thresholds = {"reprojection_px": 20.0, "min_motion_px": 5.0, "min_frames": 8,
                  "max_cost": 0.8, "min_support": 0.2, "min_margin": 0.31}
    cfg.court_side = thresholds
    saved = tmp_path / "composed.yaml"
    OmegaConf.save(cfg, saved, resolve=True)
    restored = PipelineRuntimeConfig.from_config(OmegaConf.load(saved))
    assert restored.video_court_side is None and restored.court_side_evidence == "ball"
    assert restored.court_side == current.court_side and restored.court_side.min_margin == 0.31
    assert restored.processing_settings["court_side"] == current.processing_settings["court_side"] == thresholds


@pytest.mark.parametrize("overrides,missing", [
    (["~court_side.mode"], "mode"),
    (["~court_side.video_shared"], "video_shared"),
    (["~court_side.mode", "~court_side.video_shared", "~court_side.min_margin"], "min_margin"),
])
def test_partial_court_side_formats_are_rejected(overrides: list[str], missing: str) -> None:
    with pytest.raises(ValueError, match=rf"court_side\.{missing}\b"):
        _runtime(overrides)


@pytest.mark.parametrize(("overrides", "error"), [
    (["court_side.evidence=automatic"], "court_side.evidence"),
    (["court_side.evidence=player", "~court_side.player"], "court_side.player"),
    (["court_side.evidence=player", "court_side.player.min_margin=0"], "margin"),
])
def test_invalid_player_court_side_config_is_rejected(overrides: list[str], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _runtime(overrides)


@pytest.mark.parametrize("stage", ["ball_detection", "ball_points", "ball_triangulation"])
def test_omitted_ball_stage_rejects_individual_load(stage: str) -> None:
    with pytest.raises(ConfigurationError, match=rf"execution\.{stage}=load"):
        _runtime(["court_side.evidence=player", "ball_detection.enabled=false",
                  "ball_reconstruction.enabled=false", f"execution.{stage}=load"])


def test_player_only_configuration_allows_global_cache_load() -> None:
    runtime = _runtime(["court_side.evidence=player", "ball_detection.enabled=false",
                        "ball_reconstruction.enabled=false", "cache.source=load"])
    assert runtime.omit_ball_components and runtime.cache_source == "load"


def test_ball_evidence_retains_the_loaded_detector_and_points_chain() -> None:
    runtime = _runtime(["ball_detection.enabled=false", "ball_reconstruction.enabled=false", "execution.court_side=load",
                        "execution.ball_detection=load", "execution.ball_points=load"])
    assert not runtime.omit_ball_components
    assert all(runtime.component_sources[stage] == "load" for stage in ("court_side", "ball_detection", "ball_points"))


def test_default_sampling_covers_long_meiji_clip_and_still_refuses_truncation() -> None:
    runtime = _runtime([], bind_inputs=False)
    indices = sampled_frame_indices(2108, 60000 / 1001, max_frames=runtime.sampling_max_frames)
    assert len(indices) == 1055
    np.testing.assert_array_equal(indices[[0, 250, 251, 500, 1000, 1054]], [0, 500, 501, 999, 1998, 2106])
    boundary = sampled_frame_indices(4096, 60., max_frames=runtime.sampling_max_frames)
    assert len(boundary) == 2048 and boundary[-1] == 4094
    with pytest.raises(ReconstructionUnavailable, match="maximum"):
        sampled_frame_indices(4098, 60., max_frames=runtime.sampling_max_frames)


def test_explicit_x3d_config_changes_model_and_cache_settings() -> None:
    directory = Path(__file__).parents[3] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name="pipeline_x3d_dpt")
    runtime = PipelineRuntimeConfig.from_config(cfg)
    assert runtime.ball_detection is not None
    assert runtime.ball_detection.model_type == "x3d_csn_dpt"
    assert runtime.ball_detection.score_threshold == 0.5380124449729919
    assert runtime.ball_detection.num_frames == 32
    assert runtime.ball_detection.precision == "bf16"
    assert runtime.processing_settings["ball_detection"] != _runtime([]).processing_settings["ball_detection"]


def test_missing_model_selection_fails_instead_of_guessing_checkpoint() -> None:
    with pytest.raises(ValueError, match="model_type"):
        _runtime(["~ball_detection.model_type"])


def test_placement_override_is_typed_and_part_of_cache_settings() -> None:
    value = _runtime(["player_reconstruction.placement.temporal_weight=0.2"])
    assert value.player_placement.temporal_weight == 0.2
    assert (
        cast(dict[str, Any], value.processing_settings["player_reconstruction"])[
            "placement"
        ]["temporal_weight"]
        == 0.2
    )


def test_court_methods_and_paint_measurements_change_cache_identity() -> None:
    baseline = _runtime([])
    assert baseline.court_kp.postprocess.support_method == "legacy"
    assert baseline.court_kp.region_search.acceptance_method == "legacy"
    assert baseline.court_kp.region_search.refinement_method == "model"
    for overrides in (
        ["court_kp.region_search.acceptance_method=quality"],
        ["court_kp.region_search.acceptance_method=quality", "court_kp.region_search.refinement_method=rgb_paint"],
        ["court_kp.postprocess.support_method=candidate_subsets"],
        ["court_kp.postprocess.support_method=fixed_subsets"],
        ["court_kp.postprocess.max_support_candidates=4"],
        ["court_kp.region_search.paint.radius_px=40.0"],
        ["court_kp.region_search.min_inliers=9"],
    ):
        changed = _runtime(overrides)
        assert changed.processing_settings["court_kp"] != baseline.processing_settings["court_kp"]
    with pytest.raises(ValueError, match="requires acceptance_method=quality"):
        _runtime(["court_kp.region_search.refinement_method=rgb_paint"])
    with pytest.raises(ValueError, match="paint extraction version"):
        _runtime(["court_kp.region_search.paint.version=paired_edges_v1"])
    with pytest.raises(ValueError, match="support_method"):
        _runtime(["court_kp.postprocess.support_method=adaptive"])
    with pytest.raises(ValueError, match="refinement_method"):
        _runtime(["court_kp.region_search.refinement_method=automatic"])


@pytest.mark.parametrize(
    "override",
    [
        "temporal_weight=-1",
        "data_sigma_m=.nan",
        "min_scale=2",
        "max_nfev=0",
        "min_joints=18",
    ],
)
def test_invalid_placement_is_rejected_before_model_loading(override: str) -> None:
    with pytest.raises(ValueError):
        _runtime([f"player_reconstruction.placement.{override}"])


def test_unknown_placement_parameter_is_rejected() -> None:
    with pytest.raises(ValueError):
        _runtime(["+player_reconstruction.placement.body_pose_weight=1"])


def test_stage_only_composition_does_not_bind_dummy_video_inputs() -> None:
    runtime = _runtime(
        [
            "camera_ids=[unused,unused]",
            "video_paths=[]",
            "camera_geometry.reference_camera=actual_camera",
        ],
        bind_inputs=False,
    )
    assert runtime.camera_ids == () and runtime.video_paths == ()
    assert runtime.camera_geometry.reference_camera == "actual_camera"


def test_pipeline_outputs_and_stage_cache_use_their_declared_roots(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        [
            f"paths.output_root={tmp_path / 'runs'}",
            f"paths.artifact_root={tmp_path / 'stage-artifacts'}",
            "output_directory=tennis_scene/generate/integration/s42",
            "output_name=clip",
        ]
    )
    relative = Path("tennis_scene/generate/integration/s42")
    assert runtime.output_path == tmp_path / "runs" / relative / "clip.npz"
    # The component cache is shared across runs, so resume can find earlier artifacts.
    assert runtime.cache_directory == tmp_path / "stage-artifacts" / "tennis_scene/cache/pipeline"


@pytest.mark.parametrize("fragment", ["../escape", "/tmp/escape", "outputs/repeated"])
def test_output_paths_cannot_escape_authority(fragment: str) -> None:
    with pytest.raises(PathContractError):
        _runtime([f"output_directory={fragment}"])


@pytest.mark.parametrize(
    "overrides",
    [
        ["video_paths=[a.mp4,b.mp4]", "camera_ids=[a,b]"],
        ["camera_ids=[a,a,c]"],
        ["person_observations.enabled=false"],
        ["people_models.runtime.static_cam=false"],
        ["cache.source=load", "cache.overwrite=true"],
    ],
)
def test_invalid_inputs_or_dependency_choices_fail_before_models(
    overrides: list[str],
) -> None:
    with pytest.raises(ValueError):
        _runtime(overrides)


@pytest.mark.parametrize("prefix", ["", "+"])
@pytest.mark.parametrize("override", [
    "people_models.detector=dino",
    "people_models.yolo_checkpoint=retired.pt",
    "people_models.runtime.tracking.yolo_confidence=0.5",
])
def test_removed_detector_settings_are_rejected(prefix: str, override: str) -> None:
    """Removed YOLO/selector keys must fail even when added to an old config."""
    error = UnknownConfigurationKeyError if prefix else ConfigCompositionException
    with pytest.raises(error, match=override.split("=")[0].rsplit(".", 1)[-1]):
        _runtime([prefix + override], bind_inputs=False)


def test_default_coco_source_and_explicit_finetuned_comparison() -> None:
    runtime = _runtime([])
    assert not hasattr(runtime.people, "detector")
    assert runtime.people.dino_checkpoint == (
        runtime.roots.checkpoint_root / "dino/checkpoint0029_4scale_swin.pth"
    )
    legacy = _runtime(["people_models.dino_checkpoint=player_detection/chat-player-v1-e8-best-pr937.pth"])
    assert legacy.people.dino_checkpoint == (
        legacy.roots.checkpoint_root / "player_detection/chat-player-v1-e8-best-pr937.pth"
    )
    assert legacy.people.runtime.dino_detector == runtime.people.runtime.dino_detector

    assert runtime.people.runtime.dino_detector.confidence == .30
    assert runtime.people.runtime.dino_detector.short_side == 800
    assert runtime.people.runtime.dino_detector.max_long_side == 1333


def test_ball_detection_is_used_without_refiner_assets() -> None:
    current = _runtime([])
    assert current.ball_detection is not None
    assert current.ball_detection.checkpoint.name == "i935-mixed-ft-s42-epoch09.ckpt"
    assert current.ball_detection.overlap_aggregation == "nearest_window_centre_then_earlier_start"
    assert "ball_refiner_2d" not in current.component_sources
    assert not hasattr(current, "ball_refiner")


def test_retired_refiner_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="ball_refiner"):
        _runtime(["+ball_refiner.path=old"])
