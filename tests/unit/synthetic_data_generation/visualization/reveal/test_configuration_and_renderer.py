from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray

from src.synthetic_data_generation.rendering.nht import NHTRenderCommandRequest
from src.synthetic_data_generation.visualization.reveal import renderer
from src.synthetic_data_generation.visualization.reveal.configuration import (
    build_scene_reveal_request,
)
from src.synthetic_data_generation.visualization.reveal.contracts import (
    REVEAL_PHASE_ORDER,
    RevealDurations,
    RevealMode,
    SceneRevealRequest,
)
from src.synthetic_data_generation.visualization.reveal.timeline import (
    NHTAppearanceState,
    build_timeline,
)
from src.synthetic_data_generation.visualization.reveal.viewpoint import RevealCamera
from src.utils.paths import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "src/synthetic_data_generation/configs"


def _request(overrides: list[str] | None = None) -> SceneRevealRequest:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="visualize_scene_reveal", overrides=list(overrides or [])
        )
    return build_scene_reveal_request(config)


def test_production_config_builds_a_complete_request() -> None:
    request = _request()

    assert request.mode is RevealMode.VIDEO
    assert request.scene_root.parts[-2:] == ("scenes", "B00")
    assert request.output_video.name == "scene-reveal.mp4"
    assert request.metadata_path.name == "scene-reveal.json"
    assert request.viewpoint.anchor_court_id == "court-000"
    assert request.nht.executable == "nht-render"
    assert len(request.durations.seconds) == len(REVEAL_PHASE_ORDER)


def test_config_rejects_unknown_keys_and_invalid_modes() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _request(["+reveal.output.bitrate=1"])
    with pytest.raises(ValueError, match="video or stills"):
        _request(["reveal.mode=gif"])


def test_stills_mode_requires_times_inside_the_timeline() -> None:
    request = _request(["reveal.mode=stills", "reveal.still_times_seconds=[1.0]"])
    assert request.still_times_seconds == (1.0,)
    with pytest.raises(ValueError, match="requires still_times_seconds"):
        _request(["reveal.mode=stills"])
    with pytest.raises(ValueError, match="lie in"):
        _request(["reveal.mode=stills", "reveal.still_times_seconds=[1000.0]"])
    with pytest.raises(ValueError, match="only valid in stills"):
        _request(["reveal.still_times_seconds=[1.0]"])


def test_work_directory_must_stay_outside_the_reconstruction(tmp_path: Path) -> None:
    request = _request()
    with pytest.raises(ValueError, match="reconstruction"):
        SceneRevealRequest(
            scene_root=tmp_path,
            output_video=tmp_path / "out.mp4",
            work_directory=tmp_path / "reconstruction" / "work",
            mode=request.mode,
            still_times_seconds=(),
            output=request.output,
            durations=request.durations,
            viewpoint=request.viewpoint,
            style=request.style,
            nht=request.nht,
        )


def test_executable_must_be_the_public_command_or_absolute() -> None:
    with pytest.raises(ValueError, match="executable"):
        _request(["reveal.nht.executable=bin/nht-render"])


def test_distinct_nht_states_are_ordered_by_first_use() -> None:
    timeline = build_timeline(
        RevealDurations(tuple(1.0 for _ in REVEAL_PHASE_ORDER)), fps=4.0
    )

    states = renderer.ordered_nht_states(timeline, range(len(timeline)))

    assert len(states) == len(set(states))
    assert states[0] == NHTAppearanceState(splat_progress=0.0, view_dependence=0.0)
    assert states[-1] == NHTAppearanceState(splat_progress=1.0, view_dependence=1.0)
    # Cameras/SfM phases need no NHT; crossfade and colorize share one state;
    # splat and view-dependence add one state per frame (4 each).
    assert len(states) == 1 + 4 + 4


def test_state_renderer_renders_bounded_chunks_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(
        [
            f"roots.output_root={tmp_path / 'out'}",
            "reveal.nht.states_per_invocation=2",
            "reveal.output.width=8",
            "reveal.output.height=6",
        ]
    )
    calls: list[tuple[str, ...]] = []

    class FakeArrays:
        def __init__(self, value: float) -> None:
            self.rgb: NDArray[np.float32] = np.full((6, 8, 3), value, dtype=np.float32)
            self.alpha: NDArray[np.float32] = np.ones((6, 8, 1), dtype=np.float32)

    class FakeRecord:
        def __init__(self, value: float) -> None:
            self._value = value

        def load_arrays(self) -> FakeArrays:
            return FakeArrays(self._value)

    class FakeResult:
        def __init__(self, command: NHTRenderCommandRequest) -> None:
            assert command.arbitrary_cameras is not None
            self._records = {
                camera.camera_id: FakeRecord(camera.appearance.splat_progress)
                for camera in command.arbitrary_cameras.cameras
                if camera.appearance is not None
            }

        def record(self, camera_id: str) -> FakeRecord:
            return self._records[camera_id]

    class FakeClient:
        def render(self, command: NHTRenderCommandRequest) -> FakeResult:
            assert command.arbitrary_cameras is not None
            calls.append(
                tuple(camera.camera_id for camera in command.arbitrary_cameras.cameras)
            )
            command.output_directory.mkdir(parents=True)
            assert command.arbitrary_request_path is not None
            command.arbitrary_request_path.write_text("{}")
            return FakeResult(command)

    monkeypatch.setattr(renderer, "NHTRenderClient", FakeClient)
    renderer._prepare_work_directory(request.work_directory)
    states = tuple(
        NHTAppearanceState(splat_progress=value, view_dependence=0.0)
        for value in (0.0, 0.5, 1.0)
    )
    camera = RevealCamera(
        width=8,
        height=6,
        intrinsics=np.asarray(((4.0, 0.0, 3.5), (0.0, 4.0, 2.5), (0.0, 0.0, 1.0))),
        camera_to_scene=np.eye(4),
    )
    scene_path = tmp_path / "scenes/B00/reconstruction/export/scene.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}")
    state_renderer = renderer._NHTStateRenderer(
        request, camera, scene_path, policy=_policy(), states=states
    )

    values = [float(state_renderer.layer(state)[0][0, 0, 0]) for state in states]
    state_renderer.layer(states[2])

    assert values == [0.0, 0.5, 1.0]
    assert calls == [("state-00000", "state-00001"), ("state-00002",)]
    assert list((request.work_directory / "nht").iterdir()) == []


def _policy():
    from src.synthetic_data_generation.rendering.nht import NHTAppearancePolicy

    return NHTAppearancePolicy(
        point_radius=0.01, splat_order="largest_first", splat_stagger=0.3
    )


def test_work_directory_must_be_absent_or_empty(tmp_path: Path) -> None:
    (tmp_path / "leftover").write_text("x")
    with pytest.raises(FileExistsError, match="work_directory"):
        renderer._prepare_work_directory(tmp_path)
