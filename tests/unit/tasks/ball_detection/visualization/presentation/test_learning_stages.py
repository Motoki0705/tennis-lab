"""Evidence integrity, real state timing, and CPU drawing of the agreed stages."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.ball_detection.visualization.presentation.configuration import (
    InputClip,
    build_learning_stages_config,
)
from src.tasks.ball_detection.visualization.presentation.input_panel import (
    real_clip_panel,
    schematic_panel,
)
from src.tasks.ball_detection.visualization.presentation.learning_stages import (
    LearningStagesPresentation,
    StageLogs,
    ball_architecture,
    load_stage_logs,
)
from src.tasks.ball_detection.visualization.presentation.stage_logs import trailing_mean
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.ball_detection.store import frame, write_store_clip

_PRE = PROJECT_ROOT / "knowledge/runs/run-i1050-convnext-v2-pretrain-s42"
_POST = PROJECT_ROOT / "knowledge/runs/run-i1049-convnext-query-posttrain-s42"
_SUMMARY = PROJECT_ROOT / "assets/ball_detection/learning_stages/scalars.json"


@pytest.fixture(scope="module")
def logs() -> StageLogs:
    return load_stage_logs(_PRE, _POST, training_summary=_SUMMARY)


def _config(
    tmp_path: Path, *overrides: str, return_hydra_config: bool = False
) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/ball_detection/configs"),
        version_base="1.3",
    ):
        return compose(
            config_name="visualize_learning_stages",
            return_hydra_config=return_hydra_config,
            overrides=[
                "input_panel=schematic",
                "width=640",
                "height=360",
                f"paths.output_root={tmp_path}",
                *overrides,
            ],
        )


@pytest.fixture
def scene(tmp_path: Path, logs: StageLogs) -> LearningStagesPresentation:
    return LearningStagesPresentation(
        build_learning_stages_config(_config(tmp_path)), logs, schematic_panel()
    )


def test_hydra_and_media_share_one_output_run_with_separate_destinations(
    tmp_path: Path,
) -> None:
    raw = _config(
        tmp_path / "output",
        f"paths.artifact_root={tmp_path / 'artifacts'}",
        f"paths.data_root={tmp_path / 'data'}",
        f"paths.checkpoint_root={tmp_path / 'checkpoints'}",
        return_hydra_config=True,
    )
    run_dir = tmp_path / "output" / raw.run.output_dir
    log_dir = Path(raw.hydra.run.dir)
    assert log_dir == run_dir / "hydra"
    with open_dict(raw):
        del raw["hydra"]
    runtime = build_learning_stages_config(raw)
    assert runtime.output_dir == run_dir / "presentation"
    log_dir.mkdir(parents=True)
    assert not runtime.output_dir.exists()


def test_output_location_does_not_change_rendered_frames(
    scene: LearningStagesPresentation, tmp_path: Path, logs: StageLogs
) -> None:
    relocated = LearningStagesPresentation(
        build_learning_stages_config(
            _config(tmp_path / "relocated", "run_id=another-run")
        ),
        logs,
        schematic_panel(),
    )
    assert replace(relocated.config, output_dir=scene.config.output_dir) == scene.config
    assert relocated.still_frames() == scene.still_frames()
    for name in ("intro", "handoff12", "phase2"):
        frame = scene.still_frames()[name]
        np.testing.assert_array_equal(
            scene.render_frame(frame), relocated.render_frame(frame)
        )


def test_real_bundles_have_measured_rows_and_global_step_offset(
    logs: StageLogs,
) -> None:
    first, second = logs.pretraining, logs.posttraining
    assert len(first.metrics) == len(second.metrics) == 10
    assert len(first.training) == len(second.training) == 1200
    np.testing.assert_array_equal(first.common.steps, np.arange(6000, 60001, 6000))
    np.testing.assert_array_equal(second.common.steps, np.arange(66000, 120001, 6000))
    np.testing.assert_array_equal(second.loss.steps, np.arange(60050, 120001, 50))
    assert second.common.values[0] == pytest.approx(48.40786227246914)
    assert second.common.values[-1] == pytest.approx(13.489567152302158)
    assert first.best_point[:2] == pytest.approx((48000, 17.34708861378516))
    assert second.best_point[:2] == pytest.approx((120000, 13.489567152302158))
    assert len(logs.input_files) == 10
    assert str(_SUMMARY) in logs.input_files
    assert all(not path.endswith("train-summary.jsonl") for path in logs.input_files)
    assert all(len(value) == 64 for value in logs.input_files.values())


@pytest.fixture
def bundles(tmp_path: Path) -> tuple[Path, Path]:
    paths = (tmp_path / "pretraining", tmp_path / "posttraining")
    for source, target in zip((_PRE, _POST), paths, strict=True):
        target.mkdir()
        for name in (
            "metrics.jsonl",
            "artifact-manifest.json",
            "config.json",
            "best.json",
        ):
            shutil.copyfile(source / name, target / name)
    return paths


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("sha256", "0" * 64, "parent SHA-256"),
        ("global_step", 60000, "parent global_step"),
    ],
)
def test_parent_lineage_mismatch_is_rejected(
    bundles: tuple[Path, Path], key: str, value: Any, message: str
) -> None:
    path = bundles[1] / "config.json"
    config = json.loads(path.read_text())
    config["recipe"]["parent"][key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        load_stage_logs(*bundles, training_summary=_SUMMARY)


@pytest.mark.parametrize("stage", [0, 1])
def test_evaluation_schema_mismatch_is_rejected(
    bundles: tuple[Path, Path], stage: int
) -> None:
    path = bundles[stage] / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["schema"] = "unexpected.v1"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="evaluation schema"):
        load_stage_logs(*bundles, training_summary=_SUMMARY)


def test_different_fixed_manifest_is_rejected(bundles: tuple[Path, Path]) -> None:
    path = bundles[1] / "config.json"
    config = json.loads(path.read_text())
    config["recipe"]["manifest_sha256"] = "1" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="same fixed manifest"):
        load_stage_logs(*bundles, training_summary=_SUMMARY)


def test_inconsistent_freeze_log_is_rejected(
    bundles: tuple[Path, Path], tmp_path: Path
) -> None:
    path = tmp_path / "scalars.json"
    payload = json.loads(_SUMMARY.read_text())
    payload["runs"]["posttraining"]["columns"]["encoder_frozen"][0] = False
    path.write_text(json.dumps(payload))
    receipt = json.loads(_SUMMARY.with_name("derivation.json").read_text())
    receipt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("derivation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="encoder_frozen disagrees"):
        load_stage_logs(*bundles, training_summary=path)


def test_freeze_state_and_badge_follow_the_summary_intervals(
    scene: LearningStagesPresentation,
) -> None:
    run = scene.logs.posttraining
    assert run.frozen_at(0)
    assert run.frozen_at(6000)
    assert not run.frozen_at(6001)
    frames = scene.segment_frames["phase2"]
    frozen_frame = next(
        frame for frame in frames if 65000 < scene.cursor_at(frame) < 66000
    )
    joint_frame = next(
        frame for frame in frames if 66000 < scene.cursor_at(frame) < 67000
    )
    assert scene.architecture_at(frozen_frame).blocks["cnn"] == "frozen"
    assert "cnn" not in scene.architecture_at(frozen_frame).badges
    joint = scene.architecture_at(joint_frame)
    assert joint.blocks["cnn"] == "training"
    assert joint.badges["cnn"] == "×0.1"
    assert joint.blocks["aug"] == "training"
    assert scene.cursor_at(scene.still_frames()["phase1"]) >= scene.logs.parent_step


def test_future_metrics_and_concept_values_are_never_drawn(
    scene: LearningStagesPresentation,
) -> None:
    frames = scene.segment_frames["phase2"]
    before = next(frame for frame in frames if 65900 < scene.cursor_at(frame) < 66000)
    after = next(frame for frame in frames if 66000 < scene.cursor_at(frame) < 66150)
    scene.render_frame(before)
    assert len(scene.error_axes.lines[1].get_xdata()) == 0
    scene.render_frame(after)
    np.testing.assert_array_equal(scene.error_axes.lines[1].get_xdata(), [66000])
    np.testing.assert_allclose(
        scene.error_axes.lines[1].get_ydata(), [48.40786227246914]
    )
    scene.render_frame(scene.segment_frames["phase3"][-1])
    assert len(scene.error_axes.lines[0].get_xdata()) == 10
    assert len(scene.error_axes.lines[1].get_xdata()) == 10
    assert scene.cursor_at(scene.segment_frames["handoff12"][-1]) == 60000


def test_causal_loss_smoothing_does_not_use_future_samples(logs: StageLogs) -> None:
    values = np.array([10.0, 6.0, 2.0, 100.0], dtype=np.float64)
    np.testing.assert_allclose(trailing_mean(values, window=2), [10, 8, 4, 51])
    assert logs.pretraining.loss.values.max() == pytest.approx(1)
    assert logs.posttraining.loss.values.max() == pytest.approx(1)
    assert logs.pretraining.loss.name != logs.posttraining.loss.name


def test_architecture_handoff_and_concept_are_explicit() -> None:
    blocks, _, states = ball_architecture()
    assert len({block.id for block in blocks}) == len(blocks)
    assert states["intro"].blocks["mdd"] == "fixed"
    assert states["handoff12"].blocks["dpt"] == "removed"
    assert states["handoff12"].blocks["query"] == "new"
    assert set(states["phase3"].badges.values()) == {"構想"}
    assert "cnn" not in states["phase3"].blocks


def test_concept_circuit_repeats_every_three_seconds(
    scene: LearningStagesPresentation,
) -> None:
    start = scene.segment_frames["phase3"][0] + 9
    image = scene.render_frame(start)
    repeated = scene.render_frame(start + int(3 * scene.config.fps))
    different = scene.render_frame(start + int(scene.config.fps))
    np.testing.assert_array_equal(image, repeated)
    assert not np.array_equal(image, different)


def test_three_frame_cpu_smoke_and_static_outro(
    scene: LearningStagesPresentation,
) -> None:
    indices = (
        scene.segment_frames["intro"][-1],
        scene.segment_frames["phase1"][-1],
        scene.segment_frames["phase3"][-1],
    )
    images = [scene.render_frame(frame) for frame in indices]
    assert all(
        image.shape == (360, 640, 3) and image.dtype == np.uint8 for image in images
    )
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[1], images[2])
    np.testing.assert_array_equal(
        scene.render_frame(scene.segment_frames["outro"][-1]), images[-1]
    )


@pytest.mark.parametrize(
    "override",
    [
        "input_panel=unknown",
        "input_panel=???",
        "input_panel=real_clip",
        "width=321",
        "fps=0",
        "gif_fps=0",
        "durations.phase1=0",
        "+unexpected=true",
    ],
)
def test_invalid_configuration_never_falls_back(tmp_path: Path, override: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        build_learning_stages_config(_config(tmp_path, override))


def test_real_panel_uses_saved_cpu_mdd_contract(
    tmp_path: Path, logs: StageLogs
) -> None:
    import cv2
    import torch

    from src.tasks.ball_detection.data.store import BallFrameStore
    from src.tasks.ball_detection.preprocessing.mdd import RGBToMDD

    directory = write_store_clip(
        tmp_path / "store", "tracknet/game/clip", [frame(index) for index in range(33)]
    )
    clip = InputClip(directory.resolve(), "tracknet/game/clip", 1)
    contract = logs.posttraining.recipe["parent"]["input_contract"]
    panel = real_clip_panel(clip, contract)
    store = BallFrameStore(directory.resolve())
    record = store.clip_by_id(clip.clip_id)
    rgb = np.stack(
        [
            cv2.cvtColor(store.read_bgr(store.row_of(record, i)), cv2.COLOR_BGR2RGB)
            for i in range(1, 33)
        ]
    )
    features = RGBToMDD.from_contract(contract)(
        torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy())[None]
    )[0].numpy()
    for position, index in enumerate((1, 16, 31)):
        np.testing.assert_array_equal(panel.rgb[position], rgb[index])
        np.testing.assert_array_equal(
            panel.mdd[position][..., 0], (features[1, index] * 255).astype(np.uint8)
        )
        np.testing.assert_array_equal(
            panel.mdd[position][..., 1], (features[0, index] * 255).astype(np.uint8)
        )
        assert not panel.mdd[position][..., 2].any()
    assert panel.provenance["frames"] == [2, 17, 32]
    assert panel.provenance["origin"] == "measured"
    assert len(panel.input_files) == 3
    with pytest.raises(ValueError, match="complete 32-frame"):
        real_clip_panel(InputClip(directory.resolve(), clip.clip_id, 2), contract)
    with pytest.raises(ValueError, match="input contract"):
        real_clip_panel(clip, dict(contract, schema="wrong"))
