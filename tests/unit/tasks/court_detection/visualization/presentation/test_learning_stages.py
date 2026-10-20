"""Real B-run evidence, config-derived states and CPU story rendering."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import av
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from matplotlib.colors import to_rgb
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from src.tasks.base.visualization.presentation import style
from src.tasks.court_detection.visualization.presentation.configuration import (
    DEFAULT_CONFIG,
    build_request,
    default_scalar_summary,
    default_tags,
)
from src.tasks.court_detection.visualization.presentation.learning_stages import (
    CourtLearningStory,
    CourtRunLog,
    court_architecture,
    learning_clock,
    load_court_run,
    render_learning_stages,
    supervision_paths,
    trailing_mean,
)
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.unit
BUNDLE = PROJECT_ROOT / "knowledge/runs/run-i983-court-dinov3-b-s42"


@pytest.fixture(scope="module")
def run() -> CourtRunLog:
    return load_court_run(BUNDLE)


@pytest.fixture
def config() -> DictConfig:
    with initialize_config_dir(
        config_dir=str(DEFAULT_CONFIG.parent), version_base="1.3"
    ):
        return compose(config_name=DEFAULT_CONFIG.stem)


def _seconds() -> Mapping[str, float]:
    return {"intro": 4, "warmup": 8, "train": 26, "select": 4, "handoff": 3}


def test_reads_all_real_measurements_and_best_checkpoint(run: CourtRunLog) -> None:
    assert run.steps_per_epoch == 1658
    assert run.warmup_steps == 4974
    assert run.total_steps == 33160
    assert (run.best_epoch, run.best_step) == (17, 29844)
    for name, curve in run.series.items():
        assert len(curve.steps) == (
            6 if name.startswith("val_") else 20 if name == "learning_rate" else 663
        )
    np.testing.assert_array_equal(
        run.series["val_kp"].steps, [4974, 9948, 14922, 19896, 24870, 29844]
    )
    assert run.series["val_kp"].values[0] == pytest.approx(5.9925289154)
    assert run.series["val_loss"].values[-1] == pytest.approx(1.1920050383)
    assert run.series["train_loss"].steps[0] == 50
    assert run.series["learning_rate"].steps[0] == 0
    assert run.series["learning_rate"].values[3] == pytest.approx(1e-4)
    assert (
        run.input_sha256[str(default_scalar_summary())]
        == json.loads(
            default_scalar_summary().with_name("derivation.json").read_text()
        )["sha256"]
    )


def test_missing_tag_is_an_error() -> None:
    with pytest.raises(
        ValueError, match="Missing required TensorBoard tags.*train/missing"
    ):
        load_court_run(BUNDLE, tags={**default_tags(), "train_loss": "train/missing"})


def test_incomplete_validation_log_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    payload = json.loads(default_scalar_summary().read_text())
    values = payload["scalars"]["val/kp_mean_distance_px"]
    for key in values:
        values[key] = values[key][:-1]
    summary = tmp_path / "scalars.json"
    summary.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="val/kp_mean_distance_px.*expected 6 points"):
        load_court_run(bundle, scalar_summary=summary)


@pytest.mark.parametrize(
    "field,value", [("schema_version", 2), ("source_sha256", "0" * 64)]
)
def test_scalar_summary_schema_and_source_are_strict(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    payload = json.loads(default_scalar_summary().read_text())
    payload[field] = value
    summary = tmp_path / "scalars.json"
    summary.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema|source SHA-256"):
        load_court_run(bundle, scalar_summary=summary)


def test_hydra_and_media_use_sibling_destinations_in_one_run(tmp_path: Path) -> None:
    with initialize_config_dir(
        config_dir=str(DEFAULT_CONFIG.parent), version_base="1.3"
    ):
        config = compose(
            config_name=DEFAULT_CONFIG.stem,
            return_hydra_config=True,
            overrides=[
                f"paths.{role}_root={tmp_path / role}"
                for role in ("output", "data", "artifact", "checkpoint")
            ],
        )
    run_dir = tmp_path / "output" / config.run.output_dir
    assert run_dir.relative_to(tmp_path / "output").parts[:3] == (
        "court_detection",
        "visualize",
        "learning_stages",
    )
    log_dir = Path(config.hydra.run.dir)
    assert log_dir == run_dir / "hydra"
    with open_dict(config):
        del config["hydra"]
    request = build_request(config)
    assert request.output_dir == run_dir / "presentation"
    log_dir.mkdir(parents=True)
    assert not request.output_dir.exists()


def test_output_location_preserves_still_and_sample_frame_rgb(
    config: DictConfig, tmp_path: Path, run: CourtRunLog
) -> None:
    config.width, config.height = 640, 360
    original = build_request(config)
    config.paths.output_root = str(tmp_path / "relocated")
    config.run.output_dir = "court_detection/visualize/learning_stages/another-run"
    relocated = build_request(config)
    assert original.output_dir != relocated.output_dir
    before = CourtLearningStory(
        run,
        seconds=original.seconds,
        fps=original.fps,
        width=original.width,
        height=original.height,
    )
    after = CourtLearningStory(
        run,
        seconds=relocated.seconds,
        fps=relocated.fps,
        width=relocated.width,
        height=relocated.height,
    )
    assert before.stills == after.stills
    for frame in sorted({240, 600, *before.stills.values()}):
        assert (
            hashlib.sha256(before.render_frame(frame).tobytes()).digest()
            == hashlib.sha256(after.render_frame(frame).tobytes()).digest()
        )


def test_b_run_config_mismatch_is_an_error(tmp_path: Path, run: CourtRunLog) -> None:
    config = OmegaConf.create(OmegaConf.to_container(run.config))
    config.mixed.train_batch_counts.synthetic_court = 3
    OmegaConf.save(config, tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="config mismatch.*synthetic_court"):
        load_court_run(tmp_path)


def test_architecture_state_and_labels_follow_config(run: CourtRunLog) -> None:
    config = OmegaConf.create(OmegaConf.to_container(run.config))
    blocks, edges, states = court_architecture(config)
    assert states["train"].blocks["dinov3"] == "frozen"
    assert not states["train"].badges
    assert ("transformer", "pose") in {(edge.src, edge.dst) for edge in edges}
    assert ("dpt", "pose") not in {(edge.src, edge.dst) for edge in edges}
    assert {block.label for block in blocks} >= {"合成 ×4", "実画像 ×4", "1×1\n×4"}
    config.model.encoder.train_mode = "full"
    assert court_architecture(config)[2]["train"].blocks["dinov3"] == "training"
    config.model.encoder.lora.enabled = True
    lora = court_architecture(config)[2]["train"]
    assert lora.blocks["dinov3"] == "frozen"
    assert lora.badges["dinov3"] == "LoRA"


def test_real_supervision_never_reaches_pose_loss(run: CourtRunLog) -> None:
    blocks, _, _ = court_architecture(run.config)
    paths = supervision_paths(blocks)
    assert len([key for key in paths if key.startswith("rgb_syn:loss_")]) == 5
    assert len([key for key in paths if key.startswith("rgb_real:loss_")]) == 4
    assert "rgb_real:loss_pose" not in paths
    assert paths["rgb_real:pose_gate"][-1][0] < paths["rgb_syn:loss_pose"][-1][0]
    pose_center = next(block for block in blocks if block.id == "pose")
    pose_xy = (
        pose_center.xy[0] + pose_center.wh[0] / 2,
        pose_center.xy[1] + pose_center.wh[1] / 2,
    )
    assert pose_xy in paths["rgb_syn:loss_pose"]


def test_clock_matches_agreed_update_counts(run: CourtRunLog) -> None:
    clock = learning_clock(run, _seconds(), 30)
    assert clock.frame_count() == 1350
    assert clock.at(119)[0].name == "intro"
    assert clock.at(359)[2] == 4974
    assert clock.at(1139)[2] == 33160
    assert clock.at(1140)[0].name == "select"
    assert clock.at(1140)[2] == 29844
    assert clock.at(1260)[0].name == "handoff"


def test_moving_average_uses_only_observed_values() -> None:
    values = np.array([10, 20, 30, 40], dtype=np.float64)
    np.testing.assert_array_equal(trailing_mean(values, 3), [10, 15, 20, 30])
    values[-1] = 1000
    np.testing.assert_array_equal(trailing_mean(values, 3)[:3], [10, 15, 20])


def test_three_frame_smoke_and_handoff_colors(run: CourtRunLog) -> None:
    story = CourtLearningStory(run, seconds=_seconds(), fps=30, width=640, height=360)
    frames = [
        story.render_frame(story.stills[name]) for name in ("intro", "train", "handoff")
    ]
    for frame in frames:
        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8
        assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 20
    assert not np.array_equal(frames[0], frames[1])
    for head in ("kp", "line"):
        patch = next(
            patch
            for patch in story.diagram.patches
            if patch.get_gid() == f"block:{head}"
        )
        np.testing.assert_allclose(
            patch.get_facecolor()[:3], to_rgb(style.COURT_HEAD_COLORS[head])
        )
        assert patch.get_alpha() == 1
    frozen = next(
        patch for patch in story.diagram.patches if patch.get_gid() == "block:dinov3"
    )
    assert frozen.get_alpha() == pytest.approx(0.18)


def test_validation_points_appear_only_at_cursor_and_selection_keeps_history(
    run: CourtRunLog,
) -> None:
    story = CourtLearningStory(run, seconds=_seconds(), fps=30, width=640, height=360)
    story.render_frame(358)
    assert len(story.metrics[0].lines[0].get_xdata()) == 0
    story.render_frame(359)
    assert list(story.metrics[0].lines[0].get_xdata()) == [4974]
    story.render_frame(story.stills["select"])
    assert len(story.metrics[0].lines[0].get_xdata()) == 6
    assert story.loss.lines[0].get_xdata()[-1] == 33150


@pytest.mark.parametrize(
    "field,value",
    [("width", 319), ("fps", 0), ("seconds.train", -1), ("gif_fps", float("nan"))],
)
def test_invalid_output_config_is_rejected(
    config: DictConfig, field: str, value: object
) -> None:
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError):
        build_request(config)


def test_small_export_preserves_measurements_provenance_and_stills(
    config: DictConfig, tmp_path: Path
) -> None:
    config.paths.output_root = str(tmp_path)
    config.run.output_dir = "story"
    config.width, config.height, config.gif_width = 320, 180, 160
    config.fps, config.gif_fps = 10, 10
    for name in config.seconds:
        config.seconds[name] = 0.1
    request = build_request(config)
    output = render_learning_stages(request)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frame_count"] == 5
    assert manifest["checkpoint"]["step"] == 29844
    assert manifest["provenance"]["run_id"] == BUNDLE.name
    assert "schematic" in manifest["provenance"]
    assert "val_semline" in manifest["scalars"]
    assert len(manifest["inputs"]["sha256"]) >= 7
    assert len(manifest["git_commit"]) == 40
    for name in ("intro", "train", "select", "handoff"):
        with Image.open(output / f"stills/{name}.png") as image:
            assert image.size == (320, 180)
    with av.open(str(output / "learning_stages.mp4")) as video:
        assert len(list(video.decode(video=0))) == 5
    with Image.open(output / "learning_stages.gif") as gif:
        assert gif.size == (160, 90)
    with pytest.raises(FileExistsError):
        render_learning_stages(request)
