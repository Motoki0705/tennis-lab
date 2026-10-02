from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.chat_annotation.local_agent import ct, intake, phase2
from src.tennis_scene.chat_annotation.local_agent.__main__ import main
from src.tennis_scene.chat_annotation.local_agent.campaign_state import (
    next_candidates,
    read_control,
    read_state,
)
from src.tennis_scene.chat_annotation.local_agent.common import (
    atomic_write_json,
    load_annotation,
    load_task,
)
from src.tennis_scene.chat_annotation.local_agent.configuration import (
    ControlConfig,
    campaign_context,
    file_sha256,
)

from .conftest import CampaignFixture, directory_hashes


def phase2_candidate(
    campaign: CampaignFixture, offset: float = 0
) -> tuple[str, Path, Path]:
    old_task, _ = campaign.finished_task(campaign.annotation(unresolved=1, offset=1))
    intake.adopt(old_task, "全フレームを確認済み。未解決位置をpartialで保存。")
    task_id, directory = campaign.finished_task(
        campaign.annotation(offset=offset), phase=2
    )
    assert (
        intake.adopt(task_id, "新版の形式と全フレーム確認を検証")["decision"] == "held"
    )
    phase2.cmd_compare(argparse.Namespace(task_ids=[task_id], video=False))
    comparison = (
        campaign.config.campaign_dir
        / "qa"
        / "phase2"
        / campaign.clip_id
        / "compare.json"
    )
    return task_id, directory, comparison


def test_dry_run_has_no_side_effects(campaign: CampaignFixture) -> None:
    before = (
        directory_hashes(campaign.config.annotation_root),
        directory_hashes(campaign.config.campaign_dir),
    )
    assert (
        main(["--campaign", str(campaign.config.campaign_dir), "run", "--dry-run"]) == 0
    )
    assert before == (
        directory_hashes(campaign.config.annotation_root),
        directory_hashes(campaign.config.campaign_dir),
    )


def test_init_refuses_existing_campaign_without_changes(
    campaign: CampaignFixture,
) -> None:
    before = directory_hashes(campaign.config.campaign_dir)
    assert (
        main(
            [
                "--campaign",
                str(campaign.config.campaign_dir),
                "init",
                "--root",
                str(campaign.config.annotation_root),
            ]
        )
        == 1
    )
    assert before == directory_hashes(campaign.config.campaign_dir)


def test_concurrent_adoption_is_idempotent(campaign: CampaignFixture) -> None:
    task_id, directory = campaign.finished_task(campaign.annotation())

    def publish() -> dict[str, Any]:
        with campaign_context(campaign.config):
            return intake.adopt(task_id, "合成球の位置を目視確認済み")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert all(r["decision"] == "accepted" for r in results)
    raw = list((campaign.config.annotated / "raw").glob("*.zip"))
    records = list((campaign.config.annotated / "processing").glob("*.json"))
    assert len(raw) == len(records) == 1
    assert read_state()["tasks"][task_id]["status"] == "adopted"
    assert file_sha256(
        campaign.config.annotated / "processed" / "ball" / f"{campaign.clip_id}.json"
    ) == file_sha256(directory / f"annotation_{campaign.clip_id}.json")


def test_adoption_rejects_modified_finished_annotation(
    campaign: CampaignFixture,
) -> None:
    task_id, directory = campaign.finished_task(campaign.annotation())
    atomic_write_json(
        directory / f"annotation_{campaign.clip_id}.json",
        campaign.annotation(offset=1).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="does not match"):
        intake.adopt(task_id, "QA note")
    assert not (
        campaign.config.annotated / "processed" / "ball" / f"{campaign.clip_id}.json"
    ).exists()


def test_held_candidate_keeps_existing_output(campaign: CampaignFixture) -> None:
    task_id, directory, _ = phase2_candidate(campaign)
    output = (
        campaign.config.annotated / "processed" / "ball" / f"{campaign.clip_id}.json"
    )
    old_hash = file_sha256(output)
    assert old_hash != file_sha256(directory / f"annotation_{campaign.clip_id}.json")
    intake.keep_old(task_id, "新の対象範囲を確認できないため旧を維持")
    assert file_sha256(output) == old_hash
    assert read_state()["tasks"][task_id]["status"] == "kept_old"


def test_replacement_creates_durable_history(campaign: CampaignFixture) -> None:
    task_id, directory, comparison = phase2_candidate(campaign)
    old_hash = json.loads(comparison.read_text())["old_sha256"]
    result = intake.replace(
        task_id, comparison, "位置改善を確認。球の有無は一致。", False
    )
    assert result["decision"] == "replaced"
    history = (
        campaign.config.annotated
        / "history"
        / "ball"
        / campaign.clip_id
        / f"{old_hash}.json"
    )
    assert file_sha256(history) == old_hash
    output = (
        campaign.config.annotated / "processed" / "ball" / f"{campaign.clip_id}.json"
    )
    assert file_sha256(output) == file_sha256(
        directory / f"annotation_{campaign.clip_id}.json"
    )
    adoption = read_state()["tasks"][task_id]["adoption"]
    intake.adopt(task_id, "再実行でも差し替え履歴を保持")
    assert read_state()["tasks"][task_id]["adoption"] == adoption


def test_replacement_rejects_changed_baseline(campaign: CampaignFixture) -> None:
    task_id, _, comparison = phase2_candidate(campaign)
    output = (
        campaign.config.annotated / "processed" / "ball" / f"{campaign.clip_id}.json"
    )
    atomic_write_json(output, campaign.annotation(offset=4).model_dump(mode="json"))
    changed = file_sha256(output)
    with pytest.raises(ValueError, match="changed after comparison"):
        intake.replace(task_id, comparison, "old review", True)
    assert file_sha256(output) == changed


@pytest.mark.parametrize("after_publish", [False, True])
def test_replacement_recovers_from_interrupted_publication(
    campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch, after_publish: bool
) -> None:
    task_id, directory, comparison = phase2_candidate(campaign)
    original = intake.replace_bytes
    failed = False

    def interrupted(path: Path, payload: bytes) -> None:
        nonlocal failed
        if not failed:
            failed = True
            if after_publish:
                original(path, payload)
            raise RuntimeError("simulated crash")
        original(path, payload)

    monkeypatch.setattr(intake, "replace_bytes", interrupted)
    with pytest.raises(RuntimeError, match="simulated crash"):
        intake.replace(task_id, comparison, "reviewed before crash", False)
    assert read_state()["tasks"][task_id]["status"] == "held"
    assert (
        intake.replace(task_id, comparison, "reviewed before crash", False)["decision"]
        == "replaced"
    )
    output = (
        campaign.config.annotated / "processed" / "ball" / f"{campaign.clip_id}.json"
    )
    assert file_sha256(output) == file_sha256(
        directory / f"annotation_{campaign.clip_id}.json"
    )
    record_path = intake.processing_path(
        read_state()["tasks"][task_id]["adoption"]["artifact_id"]
    )
    record = json.loads(record_path.read_text())
    assert record["state"] == "completed" and "replacement_pending" not in record


def test_exact_distance_threshold_is_not_rounded(campaign: CampaignFixture) -> None:
    task_id, _, comparison = phase2_candidate(campaign, offset=3.04)
    assert json.loads(comparison.read_text())["centre_distance_px"]["median"] > 2
    with pytest.raises(ValueError, match="median <= 2"):
        intake.replace(task_id, comparison, "review", True)


def test_one_frame_presence_difference_requires_visual_review(
    campaign: CampaignFixture,
) -> None:
    old_task, _ = campaign.finished_task(campaign.annotation(unresolved=1, offset=1))
    intake.adopt(old_task, "original QA")
    new = campaign.annotation()
    new.frames[0].balls = []
    task_id, _ = campaign.finished_task(new, phase=2)
    intake.adopt(task_id, "new QA")
    phase2.cmd_compare(argparse.Namespace(task_ids=[task_id], video=False))
    comparison = (
        campaign.config.campaign_dir
        / "qa"
        / "phase2"
        / campaign.clip_id
        / "compare.json"
    )
    assert json.loads(comparison.read_text())["ball_only_in_old"] == 1
    with pytest.raises(ValueError, match="explicit visual review"):
        intake.replace(task_id, comparison, "toss starts at release", False)
    assert (
        intake.replace(
            task_id,
            comparison,
            "f0 is held before release, new exclusion is correct",
            True,
        )["decision"]
        == "replaced"
    )


def test_unreview_ranges_survive_continuation_without_mutating_previous(
    campaign: CampaignFixture,
) -> None:
    task_id, previous = campaign.finished_task(campaign.annotation())
    previous_json = previous / f"annotation_{campaign.clip_id}.json"
    before = file_sha256(previous_json)
    result = intake.revise(task_id, "離手境界を再確認", ["2:5"])
    assert result["unreview_frames"] == [2, 3, 4]
    directory = previous.parent / "attempt_02"
    directory.mkdir()
    task = json.loads((previous / "task.json").read_text())
    task.update(
        attempt=2,
        attempt_dir=str(directory),
        annotation=str(directory / previous_json.name),
        previous_annotation=str(previous_json),
        unreview_frames=[2, 3, 4],
    )
    atomic_write_json(directory / "task.json", task)
    assert ct.main(["init", str(directory)]) == 0
    current = load_annotation(directory / previous_json.name)
    assert [f.frame_index for f in current.frames if not f.reviewed] == [2, 3, 4]
    assert current.status == "partial"
    assert file_sha256(previous_json) == before


def test_all_reviewed_unresolved_annotation_can_request_continuation(
    campaign: CampaignFixture,
) -> None:
    _, directory = campaign.finished_task(campaign.annotation(unresolved=1))
    assert (
        ct.main(
            [
                "finish",
                str(directory),
                "--outcome",
                "needs_continuation",
                "--summary",
                "位置未解決を引き継ぎ",
            ]
        )
        == 0
    )
    result = json.loads((directory / "result.json").read_text())
    assert result["outcome"] == "needs_continuation"
    assert result["validation"]["reviewed"] == result["validation"]["frames"]


def test_worker_rejects_output_escape(campaign: CampaignFixture) -> None:
    _, directory = campaign.finished_task(campaign.annotation())
    task = json.loads((directory / "task.json").read_text())
    task["annotation"] = str(campaign.config.annotated / "escaped.json")
    atomic_write_json(directory / "task.json", task)
    with pytest.raises(ValueError, match="inside this attempt"):
        load_task(directory)


def test_scheduler_never_selects_a_running_clip(campaign: CampaignFixture) -> None:
    state = read_state()
    task_id = f"{campaign.clip_id}__ball"
    state["tasks"][task_id]["status"] = "running"
    state["tasks"][f"{task_id}__p2"] = {
        **state["tasks"][task_id],
        "status": "pending",
        "phase": 2,
        "rank": 1,
    }
    assert next_candidates(state, read_control()) == []


@pytest.mark.parametrize("value", [0, -1, 31])
def test_invalid_parallel_count_is_rejected(value: int) -> None:
    with pytest.raises(ValueError):
        ControlConfig(max_parallel=value)


def test_shared_candidates_do_not_mark_frames_reviewed(
    campaign: CampaignFixture,
) -> None:
    from src.tennis_scene.chat_annotation.local_agent import worker_candidates

    _, directory = campaign.finished_task(campaign.annotation())
    annotation_file = directory / f"annotation_{campaign.clip_id}.json"
    before = file_sha256(annotation_file)
    checkpoint = campaign.config.campaign_dir / "tiny-checkpoint"
    checkpoint.write_bytes(b"test identity only; this checkpoint must never be loaded")
    config = campaign.config.model_copy(
        update={
            "ball_checkpoint": checkpoint,
            "ball_checkpoint_sha256": file_sha256(checkpoint),
        }
    )
    with campaign_context(config):
        identity = worker_candidates.ball_identity(campaign.manifest, 0.05, 3)
        cache = (
            config.campaign_dir / "cache" / "cands_ball" / f"{campaign.clip_id}.json"
        )
        atomic_write_json(
            cache,
            {
                "identity": identity,
                "proposals_only": True,
                "frames": {
                    str(i): {"candidates": [{"center_px": [320, 180], "score": 0.9}]}
                    for i in range(12)
                },
            },
        )
        assert ct.main(["cands-ball", str(directory)]) == 0
    assert file_sha256(annotation_file) == before


def test_incompatible_candidate_cache_is_not_silently_recomputed(
    campaign: CampaignFixture,
) -> None:
    _, directory = campaign.finished_task(campaign.annotation())
    checkpoint = campaign.config.campaign_dir / "tiny-checkpoint"
    checkpoint.write_bytes(b"test checkpoint identity")
    config = campaign.config.model_copy(
        update={
            "ball_checkpoint": checkpoint,
            "ball_checkpoint_sha256": file_sha256(checkpoint),
        }
    )
    cache = config.campaign_dir / "cache" / "cands_ball" / f"{campaign.clip_id}.json"
    atomic_write_json(cache, {"identity": {"checkpoint_sha256": "wrong"}, "frames": {}})
    with campaign_context(config):
        assert ct.main(["cands-ball", str(directory)]) == 1
    assert json.loads(cache.read_text())["identity"]["checkpoint_sha256"] == "wrong"


def test_cuda_prefetch_requires_training_queue(
    campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tennis_scene.chat_annotation.local_agent import prefetch

    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    monkeypatch.delenv("TENNIS_GPU_RESOURCE", raising=False)
    with pytest.raises(ValueError, match="training queue"):
        prefetch.main(["--device", "cuda"])


def test_undefined_distance_median_is_not_automatically_accepted(
    campaign: CampaignFixture,
) -> None:
    old = campaign.annotation(unresolved=12)
    new = campaign.annotation()
    comparison = phase2.compare_pair(old, new, campaign.manifest)
    assert comparison["centre_distance_px"]["median"] is None
    old_task, _ = campaign.finished_task(old)
    intake.adopt(old_task, "old QA")
    task_id, _ = campaign.finished_task(new, phase=2)
    intake.adopt(task_id, "new QA")
    phase2.cmd_compare(argparse.Namespace(task_ids=[task_id], video=False))
    path = (
        campaign.config.campaign_dir
        / "qa"
        / "phase2"
        / campaign.clip_id
        / "compare.json"
    )
    with pytest.raises(ValueError, match="defined centre-distance"):
        intake.replace(task_id, path, "presence checked", True)


def test_comparison_matches_every_ball_not_only_the_closest(
    campaign: CampaignFixture,
) -> None:
    old = campaign.annotation()
    new = campaign.annotation()
    for row in old.frames:
        row.balls.append(
            row.balls[0].model_copy(update={"track_id": "b2", "center_px": [100, 100]})
        )
    for row in new.frames:
        row.balls.append(
            row.balls[0].model_copy(update={"track_id": "b2", "center_px": [110, 100]})
        )
    result = phase2.compare_pair(old, new, campaign.manifest)
    assert result["matched_centres"] == 24
    assert result["centre_distance_px"]["median"] == 5
    assert phase2.presence_runs(
        phase2.centres(old), phase2.centres(campaign.annotation())
    ) == [(0, 11, "COUNT")]
