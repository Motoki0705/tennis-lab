from __future__ import annotations

import pytest

from src.synthetic_data_generation.visualization.reveal.contracts import (
    REVEAL_PHASE_ORDER,
    RevealDurations,
    RevealPhase,
)
from src.synthetic_data_generation.visualization.reveal.timeline import (
    build_timeline,
    frame_index_at,
    phase_frame_counts,
    smoothstep,
)


def _durations(seconds: float = 1.0) -> RevealDurations:
    return RevealDurations(tuple(seconds for _ in REVEAL_PHASE_ORDER))


def test_every_phase_gets_rounded_frames_in_order() -> None:
    timeline = build_timeline(_durations(0.5), fps=10.0)

    counts = phase_frame_counts(_durations(0.5), 10.0)
    assert counts == tuple(5 for _ in REVEAL_PHASE_ORDER)
    assert len(timeline) == sum(counts)
    phases = [state.phase for state in timeline]
    assert phases == [phase for phase in REVEAL_PHASE_ORDER for _ in range(5)]


def test_each_phase_ends_on_its_completed_state() -> None:
    timeline = build_timeline(_durations(), fps=4.0)

    def last(phase: RevealPhase):
        return [state for state in timeline if state.phase is phase][-1]

    assert last(RevealPhase.CAMERAS).camera_fraction == 1.0
    assert last(RevealPhase.SFM).sfm_fraction == 1.0
    crossfaded = last(RevealPhase.SFM_TO_GAUSSIANS)
    assert crossfaded.gaussian_opacity == 1.0
    assert crossfaded.sfm_opacity == 0.0 and crossfaded.camera_opacity == 0.0
    assert last(RevealPhase.COLORIZE).color_progress == 1.0
    splatted = last(RevealPhase.SPLAT)
    assert splatted.nht.splat_progress == 1.0 and splatted.matte_release == 1.0
    assert last(RevealPhase.VIEW_DEPENDENCE).nht.view_dependence == 1.0


def test_hold_ends_on_the_unmodified_nht_render() -> None:
    timeline = build_timeline(_durations(), fps=4.0)
    final = [state for state in timeline if state.phase is RevealPhase.HOLD][-1]

    assert final.gaussian_opacity == 1.0
    assert final.color_progress == 1.0 and final.matte_release == 1.0
    assert final.nht.splat_progress == 1.0 and final.nht.view_dependence == 1.0
    assert final.nht.view_progress == 0.0 and final.alignment_progress == 0.0
    assert final.sfm_opacity == 0.0 and final.camera_opacity == 0.0


def test_alignment_is_drawn_on_the_final_render_before_the_camera_moves() -> None:
    timeline = build_timeline(_durations(), fps=4.0)
    alignment = [state for state in timeline if state.phase is RevealPhase.ALIGNMENT]
    move = [state for state in timeline if state.phase is RevealPhase.CAMERA_MOVE]
    hold = [state for state in timeline if state.phase is RevealPhase.HOLD][-1]

    # Alignment reuses the held render: only the overlay changes.
    assert {state.nht for state in alignment} == {hold.nht}
    assert alignment[-1].alignment_progress == 1.0
    # The camera then moves with the overlay kept and the render complete.
    assert all(state.alignment_progress == 1.0 for state in move)
    assert all(
        state.nht.splat_progress == 1.0 and state.nht.view_dependence == 1.0
        for state in move
    )
    assert move[-1].nht.view_progress == 1.0
    assert timeline[-1] is move[-1]


def test_camera_leaves_the_reveal_viewpoint_only_after_the_alignment() -> None:
    timeline = build_timeline(_durations(), fps=4.0)
    move_start = REVEAL_PHASE_ORDER.index(RevealPhase.CAMERA_MOVE)

    for state in timeline:
        moving = REVEAL_PHASE_ORDER.index(state.phase) >= move_start
        assert (state.nht.view_progress > 0.0) is moving
        if moving:
            # Procedural SfM/camera layers are projected for the fixed viewpoint.
            assert state.sfm_opacity == 0.0 and state.camera_opacity == 0.0


def test_progress_values_never_decrease() -> None:
    timeline = build_timeline(_durations(), fps=8.0)
    for name in ("camera_fraction", "sfm_fraction", "gaussian_opacity", "color_progress"):
        values = [getattr(state, name) for state in timeline]
        assert values == sorted(values), name
    for name in ("splat_progress", "view_dependence", "view_progress"):
        values = [getattr(state.nht, name) for state in timeline]
        assert values == sorted(values), name
    alignment = [state.alignment_progress for state in timeline]
    assert alignment == sorted(alignment)


def test_nht_is_needed_only_once_gaussians_fade_in() -> None:
    timeline = build_timeline(_durations(), fps=4.0)

    for state in timeline:
        early = state.phase in {RevealPhase.CAMERAS, RevealPhase.SFM}
        assert state.needs_nht is not early


def test_smoothstep_is_clamped_and_eased() -> None:
    assert smoothstep(-1.0) == 0.0 and smoothstep(2.0) == 1.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(0.1) < 0.1


def test_frame_index_rejects_times_outside_timeline() -> None:
    assert frame_index_at(0.99, 10.0, 20) == 9
    with pytest.raises(ValueError, match="outside"):
        frame_index_at(2.0, 10.0, 20)


def test_durations_require_every_phase_exactly_once() -> None:
    mapping = {phase.value: 1.0 for phase in REVEAL_PHASE_ORDER}
    assert RevealDurations.from_mapping(mapping).of(RevealPhase.SPLAT) == 1.0
    with pytest.raises(ValueError, match="unknown=\\['extra'\\]"):
        RevealDurations.from_mapping({**mapping, "extra": 1.0})
    del mapping["hold"]
    with pytest.raises(ValueError, match="hold"):
        RevealDurations.from_mapping(mapping)
    with pytest.raises(ValueError, match="positive"):
        RevealDurations(tuple(0.0 for _ in REVEAL_PHASE_ORDER))
