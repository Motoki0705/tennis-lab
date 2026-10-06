from pathlib import Path

import pytest

from src.tennis_scene.chat_annotation.player_pose import orchestrator, reporting
from src.tennis_scene.chat_annotation.player_pose.storage import (
    clip_root,
    read_json,
    write_json,
)


def test_pose_queue_resumes_failed_attempt_and_excludes_unapproved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tennis_scene.chat_annotation.player_pose import publication

    campaign = tmp_path / "campaign"
    dataset = tmp_path / "poses"
    config = {
        "queue_dir": str(tmp_path / "queue"),
        "generation_attempts": 2,
        "dataset": str(dataset),
    }
    plan = {
        "clips": [
            {"selected": True, "action": "generate", "index": i} for i in range(2)
        ]
    }
    write_json(
        dataset / "manifest.json",
        {"clips": [{"pose_status": "pose_pending"}, {"pose_status": "needs_review"}]},
    )
    write_json(campaign / "review_status.json", {"status": "partial"})
    root = clip_root(campaign, 0)
    write_json(root / "review.json", {"status": "approved", "files": {}})
    write_json(root / "pose-queue-attempt-00.json", {"job": "failed-owned.job"})
    calls = []
    monkeypatch.setattr(orchestrator, "load_campaign", lambda _: (config, plan, None))
    monkeypatch.setattr(reporting, "save_metrics", lambda *_: None)

    def enqueue(_: dict, __: Path, index: int, attempt: int, stage: str) -> str:
        assert index == 0 and attempt == 1 and stage == "pose"
        calls.append(index)
        return "retry-owned.job"

    def queue_state(_: Path, name: str) -> str:
        if name == "failed-owned.job":
            return "failed"
        write_json(root / "pose.json", {"status": "complete", "files": {}})
        return "done"

    def publish(_: Path, index: int) -> dict:
        assert index == 0 and (root / "pose.json").exists()
        write_json(root / "publication.json", {"status": "published", "files": {}})
        return {"status": "published"}

    monkeypatch.setattr(orchestrator, "enqueue_clip", enqueue)
    monkeypatch.setattr(orchestrator, "queue_state", queue_state)
    monkeypatch.setattr(publication, "publish", publish)
    orchestrator.pose_worker(campaign)
    assert calls == [0]
    assert read_json(root / "pose-queue-attempt-01.json")["job"] == "retry-owned.job"
    orchestrator.pose_worker(campaign)
    assert calls == [0]
    # Interruption during publication needs no additional GPU admission.
    (root / "publication.json").unlink()
    orchestrator.pose_worker(campaign)
    assert calls == [0]


def test_cancelled_pose_job_stays_unpublished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    dataset = tmp_path / "dataset"
    config = {
        "queue_dir": str(tmp_path),
        "generation_attempts": 2,
        "dataset": str(dataset),
    }
    plan = {"clips": [{"selected": True, "action": "generate", "index": 0}]}
    root = clip_root(campaign, 0)
    write_json(root / "review.json", {"status": "approved", "files": {}})
    write_json(root / "pose-queue-attempt-00.json", {"job": "cancelled.job"})
    write_json(dataset / "manifest.json", {"clips": [{"pose_status": "pose_pending"}]})
    monkeypatch.setattr(orchestrator, "load_campaign", lambda _: (config, plan, None))
    monkeypatch.setattr(orchestrator, "queue_state", lambda *_: "cancelled")
    monkeypatch.setattr(reporting, "save_metrics", lambda *_: None)
    with pytest.raises(RuntimeError, match="cancelled"):
        orchestrator.pose_worker(campaign)
    assert not (root / "publication.json").exists()
