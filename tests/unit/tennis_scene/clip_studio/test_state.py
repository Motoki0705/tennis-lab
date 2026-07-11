"""Tests for src/tennis_scene/clip_studio/state.py."""

import pytest

from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.utils.video import VideoInfo


class TestConstruction:
    def test_initial_positions(self, studio_state: ClipStudioState) -> None:
        assert studio_state.extent_sec() == (0.0, 10.0)
        assert studio_state.playhead_sec == 0.0
        assert (studio_state.view_start_sec, studio_state.view_end_sec) == (0.0, 10.0)
        assert not studio_state.dirty

    def test_infos_length_mismatch_raises(
        self, two_camera_project: ClipStudioProject
    ) -> None:
        with pytest.raises(ValueError, match="must match sources"):
            ClipStudioState(
                two_camera_project,
                [VideoInfo(fps=30.0, width=64, height=36, frame_count=300)],
            )

    def test_invalid_info_raises(self, two_camera_project: ClipStudioProject) -> None:
        with pytest.raises(ValueError, match="invalid metadata"):
            ClipStudioState(
                two_camera_project,
                [
                    VideoInfo(fps=30.0, width=64, height=36, frame_count=300),
                    VideoInfo(fps=0.0, width=64, height=36, frame_count=240),
                ],
            )


class TestPlayback:
    def test_seek_clamps_to_extent(self, studio_state: ClipStudioState) -> None:
        studio_state.seek(-5.0)
        assert studio_state.playhead_sec == 0.0
        studio_state.seek(99.0)
        assert studio_state.playhead_sec == 10.0

    def test_step_frames_uses_selected_camera_fps(
        self, studio_state: ClipStudioState
    ) -> None:
        studio_state.seek(1.0)
        studio_state.step_frames(3)
        assert studio_state.playhead_sec == pytest.approx(1.1)

    def test_advance_stops_at_end(self, studio_state: ClipStudioState) -> None:
        studio_state.playing = True
        studio_state.seek(9.9)
        studio_state.advance(0.5)
        assert studio_state.playhead_sec == 10.0
        assert not studio_state.playing

    def test_advance_negative_raises(self, studio_state: ClipStudioState) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            studio_state.advance(-0.1)

    def test_frame_indices_at(self, studio_state: ClipStudioState) -> None:
        # cam1 has offset -1 -> local = global - 1
        assert studio_state.frame_indices_at(2.0) == [60, 30]
        assert studio_state.frame_indices_at(0.5) == [15, None]


class TestSync:
    def test_nudge_selected_offset_sets_dirty(self, studio_state: ClipStudioState) -> None:
        studio_state.select_camera(1)
        new_offset = studio_state.nudge_selected_offset(1.0 / 30.0)
        assert new_offset == pytest.approx(-1.0 + 1.0 / 30.0)
        assert studio_state.dirty

    def test_select_camera_out_of_range_raises(
        self, studio_state: ClipStudioState
    ) -> None:
        with pytest.raises(ValueError, match="out of range"):
            studio_state.select_camera(2)

    def test_apply_offsets(self, studio_state: ClipStudioState) -> None:
        studio_state.apply_offsets([0.0, 0.5])
        assert studio_state.project.sources[1].offset_sec == 0.5
        assert studio_state.dirty
        # extent changed: cam1 now covers [-0.5, 7.5]
        assert studio_state.extent_sec() == (-0.5, 10.0)

    def test_apply_offsets_wrong_length_raises(
        self, studio_state: ClipStudioState
    ) -> None:
        with pytest.raises(ValueError, match="must match cameras"):
            studio_state.apply_offsets([0.0])


class TestClips:
    def test_make_clip_from_marks(self, studio_state: ClipStudioState) -> None:
        studio_state.seek(5.0)
        studio_state.set_mark_in()
        studio_state.seek(6.5)
        studio_state.set_mark_out()
        clip = studio_state.make_clip_from_marks()
        assert clip.name == "clip_001"
        assert (clip.start_sec, clip.end_sec) == (5.0, 6.5)
        assert studio_state.selected_clip == 1
        assert studio_state.dirty

    def test_make_clip_requires_marks(self, studio_state: ClipStudioState) -> None:
        with pytest.raises(ValueError, match="marks are required"):
            studio_state.make_clip_from_marks()

    def test_make_clip_requires_order(self, studio_state: ClipStudioState) -> None:
        studio_state.seek(6.0)
        studio_state.set_mark_in()
        studio_state.seek(5.0)
        studio_state.set_mark_out()
        with pytest.raises(ValueError, match="must be after"):
            studio_state.make_clip_from_marks()

    def test_delete_selected_clip(self, studio_state: ClipStudioState) -> None:
        studio_state.selected_clip = 0
        removed = studio_state.delete_selected_clip()
        assert removed.name == "clip_000"
        assert studio_state.project.clips == []
        assert studio_state.selected_clip is None

    def test_delete_without_selection_raises(self, studio_state: ClipStudioState) -> None:
        with pytest.raises(ValueError, match="no clip selected"):
            studio_state.delete_selected_clip()

    def test_select_clip_at(self, studio_state: ClipStudioState) -> None:
        assert studio_state.select_clip_at(3.0) == 0
        assert studio_state.select_clip_at(9.0) is None

    def test_cycle_selected_clip_by_start_time(self, studio_state: ClipStudioState) -> None:
        studio_state.project.clips.append(Clip(name="early", start_sec=0.5, end_sec=1.0))
        assert studio_state.cycle_selected_clip(1) == 1  # "early" starts first
        assert studio_state.cycle_selected_clip(1) == 0
        assert studio_state.cycle_selected_clip(1) == 1


class TestView:
    def test_zoom_in_keeps_anchor(self, studio_state: ClipStudioState) -> None:
        studio_state.zoom_view(0.5, anchor_sec=5.0)
        assert studio_state.view_end_sec - studio_state.view_start_sec == pytest.approx(5.0)
        assert studio_state.view_start_sec == pytest.approx(2.5)

    def test_zoom_out_clamps_to_extent(self, studio_state: ClipStudioState) -> None:
        studio_state.zoom_view(0.5)
        studio_state.zoom_view(10.0)
        assert (studio_state.view_start_sec, studio_state.view_end_sec) == (0.0, 10.0)

    def test_zoom_respects_min_span(self, studio_state: ClipStudioState) -> None:
        for _ in range(20):
            studio_state.zoom_view(0.5, anchor_sec=5.0)
        span = studio_state.view_end_sec - studio_state.view_start_sec
        assert span == pytest.approx(ClipStudioState.MIN_VIEW_SPAN_SEC)

    def test_pan_clamps(self, studio_state: ClipStudioState) -> None:
        studio_state.zoom_view(0.5, anchor_sec=5.0)
        studio_state.pan_view(100.0)
        assert studio_state.view_end_sec == pytest.approx(10.0)
        studio_state.pan_view(-100.0)
        assert studio_state.view_start_sec == pytest.approx(0.0)

    def test_fit_view(self, studio_state: ClipStudioState) -> None:
        studio_state.zoom_view(0.3, anchor_sec=2.0)
        studio_state.fit_view()
        assert (studio_state.view_start_sec, studio_state.view_end_sec) == (0.0, 10.0)

    def test_zoom_invalid_factor_raises(self, studio_state: ClipStudioState) -> None:
        with pytest.raises(ValueError, match="factor"):
            studio_state.zoom_view(0.0)
