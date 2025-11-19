from __future__ import annotations

from typing import Any

import torch
from pytest import MonkeyPatch

from src.tennis.sim.generator import (
    GenConfig,
    TennisPoseSceneGenerator,
    _PlayerSequence,
)


def test_generate_scene_produces_multi_player_payload(monkeypatch: MonkeyPatch) -> None:
    def _fake_players(
        self: TennisPoseSceneGenerator, frames_total: int
    ) -> list[_PlayerSequence]:
        base = torch.zeros((frames_total, 17, 3), dtype=torch.float32)
        base[..., 2] = 1.5
        racket = torch.zeros((frames_total, 3, 3), dtype=torch.float32)
        racket[..., 2] = 1.2
        player_a = _PlayerSequence(base, racket)
        offset = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
        player_b = _PlayerSequence(base + offset, racket)
        return [player_a, player_b]

    monkeypatch.setattr(TennisPoseSceneGenerator, "_build_players", _fake_players)

    class _StubAssetLibrary:
        """Minimal stub for dependency injection."""

        def sample_sequence(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:  # pragma: no cover - not used in test
            raise AssertionError("should not be called in test")

    cfg = GenConfig(
        duration_sec=0.1, fps=10, num_cameras=1, min_players=2, max_players=2
    )
    scene = TennisPoseSceneGenerator(
        cfg,
        asset_library=_StubAssetLibrary(),  # type: ignore[arg-type]
    ).generate_scene("demo")

    assert scene["num_cameras"] == 1
    frame = scene["frames"][0]
    assert frame["num_players"] == 2
    assert len(frame["player_joints_3d"]) == 2
    cam_payload = frame["cam_0"]
    assert len(cam_payload["player_keypoints_2d"]["joints"]) == 2
    assert len(cam_payload["racket_keypoints_2d"]["points"]) == 2
