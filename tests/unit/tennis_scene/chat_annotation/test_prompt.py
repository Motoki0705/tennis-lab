from __future__ import annotations

from pathlib import Path

from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.prompt import render_request
from src.tennis_scene.chat_annotation.runtime.contracts import Policies


def test_request_needs_no_video_catalog(tmp_path: Path) -> None:
    contents, _ = build_kit(
        tmp_path / "project_kits", Policies(ball_max_gap_seconds=0.1)
    )
    request = render_request(contents)
    assert "入力一覧" not in request
    assert "jsonl" not in request
    assert "全フレーム" in request
    assert "0.1秒以下" in request
    assert "{{" not in request
    assert '"tennis_chat_ball_annotation.v1"' in request
    assert "$defs" in request
    assert "players" not in request
    assert len(request) > 3000


def test_gap_policy_is_in_ball_request_and_kit_identity(tmp_path: Path) -> None:
    first, first_id = build_kit(
        tmp_path / "first" / "project_kits", Policies(ball_max_gap_seconds=0.1)
    )
    second, second_id = build_kit(
        tmp_path / "second" / "project_kits", Policies(ball_max_gap_seconds=0.25)
    )
    assert first_id != second_id
    assert first["ball_detection_annotation.schema.json"] == second[
        "ball_detection_annotation.schema.json"
    ]
    assert "0.25秒以下" in render_request(second, "ball_detection")
    assert "0.25秒以下" not in render_request(second, "player_detection")
