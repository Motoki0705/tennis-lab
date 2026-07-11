"""Tests for src/tennis_scene/clip_studio/export.py (pure planning logic)."""

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.export import ExportSettings, plan_clip_export
from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject
from src.utils.video import VideoInfo


def make_settings(tmp_path: Path, **kwargs: object) -> ExportSettings:
    return ExportSettings(output_dir=tmp_path / "clips", **kwargs)  # type: ignore[arg-type]


class TestPlanClipExport:
    def test_frame_mapping_with_offsets(
        self,
        two_camera_project: ClipStudioProject,
        two_camera_infos: list[VideoInfo],
        tmp_path: Path,
    ) -> None:
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        plan = plan_clip_export(
            two_camera_project, two_camera_infos, clip, make_settings(tmp_path)
        )
        assert plan.fps == 30.0
        assert (plan.width, plan.height) == (64, 36)
        assert plan.num_frames == 30
        # cam0: local = global -> indices 60..89
        assert plan.cameras[0].frame_indices == tuple(range(60, 90))
        # cam1: offset -1 -> local = global - 1 -> indices 30..59
        assert plan.cameras[1].frame_indices == tuple(range(30, 60))

    def test_explicit_fps_resamples(
        self,
        two_camera_project: ClipStudioProject,
        two_camera_infos: list[VideoInfo],
        tmp_path: Path,
    ) -> None:
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        plan = plan_clip_export(
            two_camera_project,
            two_camera_infos,
            clip,
            make_settings(tmp_path, fps=15.0),
        )
        assert plan.num_frames == 15
        # 30fps source sampled at 15fps -> every second frame
        assert plan.cameras[0].frame_indices == tuple(range(60, 90, 2))

    def test_mixed_fps_requires_explicit_target(
        self, two_camera_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        infos = [
            VideoInfo(fps=30.0, width=64, height=36, frame_count=300),
            VideoInfo(fps=25.0, width=64, height=36, frame_count=200),
        ]
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        with pytest.raises(ValueError, match="mixed fps"):
            plan_clip_export(two_camera_project, infos, clip, make_settings(tmp_path))

    def test_mixed_resolution_requires_explicit_target(
        self, two_camera_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        infos = [
            VideoInfo(fps=30.0, width=64, height=36, frame_count=300),
            VideoInfo(fps=30.0, width=96, height=64, frame_count=240),
        ]
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        with pytest.raises(ValueError, match="mixed resolutions"):
            plan_clip_export(two_camera_project, infos, clip, make_settings(tmp_path))
        plan = plan_clip_export(
            two_camera_project,
            infos,
            clip,
            make_settings(tmp_path, width=96, height=64),
        )
        assert (plan.width, plan.height) == (96, 64)

    def test_width_without_height_raises(
        self,
        two_camera_project: ClipStudioProject,
        two_camera_infos: list[VideoInfo],
        tmp_path: Path,
    ) -> None:
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        with pytest.raises(ValueError, match="set together"):
            plan_clip_export(
                two_camera_project,
                two_camera_infos,
                clip,
                make_settings(tmp_path, width=96),
            )

    def test_odd_target_size_raises(
        self,
        two_camera_project: ClipStudioProject,
        two_camera_infos: list[VideoInfo],
        tmp_path: Path,
    ) -> None:
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        with pytest.raises(ValueError, match="even"):
            plan_clip_export(
                two_camera_project,
                two_camera_infos,
                clip,
                make_settings(tmp_path, width=95, height=64),
            )

    def test_out_of_coverage_names_camera(
        self,
        two_camera_project: ClipStudioProject,
        two_camera_infos: list[VideoInfo],
        tmp_path: Path,
    ) -> None:
        # cam1 covers global [1, 9]; clip starts before that
        clip = Clip(name="clip", start_sec=0.0, end_sec=2.0)
        with pytest.raises(ValueError, match="camera 'cam1' does not cover"):
            plan_clip_export(
                two_camera_project, two_camera_infos, clip, make_settings(tmp_path)
            )

    def test_sub_frame_clip_raises(
        self,
        two_camera_project: ClipStudioProject,
        two_camera_infos: list[VideoInfo],
        tmp_path: Path,
    ) -> None:
        clip = Clip(name="clip", start_sec=2.0, end_sec=2.01)
        with pytest.raises(ValueError, match="shorter"):
            plan_clip_export(
                two_camera_project, two_camera_infos, clip, make_settings(tmp_path)
            )

    def test_infos_length_mismatch_raises(
        self, two_camera_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        clip = Clip(name="clip", start_sec=2.0, end_sec=3.0)
        with pytest.raises(ValueError, match="must match sources"):
            plan_clip_export(
                two_camera_project,
                [VideoInfo(fps=30.0, width=64, height=36, frame_count=300)],
                clip,
                make_settings(tmp_path),
            )
