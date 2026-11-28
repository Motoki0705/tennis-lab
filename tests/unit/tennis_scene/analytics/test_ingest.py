"""Read-only audit contracts for the frozen release and the score ledger."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.ingest import v4_reader
from src.tennis_scene.analytics.ingest.audit import audit_report, prepare_ingest
from src.tennis_scene.analytics.ingest.score_adapter import ScoredVideo, read_scores
from src.tennis_scene.analytics.ingest.supplement import read_supplement
from src.tennis_scene.analytics.ingest.terminal_cause import derive_terminal_cause
from src.tennis_scene.analytics.ingest.v4_reader import read_v4
from tests.support.tennis_scene.analytics_ingest import (
    IngestFixture,
    digest,
    make_ingest_fixture,
    write_json,
    write_sums,
)


@pytest.fixture
def inputs(tmp_path: Path) -> IngestFixture:
    return make_ingest_fixture(tmp_path)


def _scores(inputs: IngestFixture) -> ScoredVideo:
    return read_scores(
        read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256),
        inputs.projects,
        inputs.resolver(),
    )


def _project_clips(
    inputs: IngestFixture,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(inputs.projects.read_bytes())
    project = document["projects"]["video_000"]
    return (
        document,
        project,
        sorted(project["clips"], key=lambda clip: clip["start_sec"]),
    )


def test_v4_preserves_original_event_ids_adoption_and_split(
    inputs: IngestFixture,
) -> None:
    release = read_v4(
        inputs.root, manifest_sha256=digest(inputs.root / "manifest.json")
    )
    assert [(clip.clip_id, clip.split) for clip in release.clips] == [
        ("video_000/clip_009", "train"),
        ("video_000/clip_012", "val"),
        ("video_000/clip_030", "test"),
    ]
    assert release.verified_files == 37
    assert release.listed_files == 40
    early = release.clips[0]
    first = early.events[0]
    assert (first.source_event_id, first.frame, first.subtype, first.court_side) == (
        "e001",
        4,
        "serve_overhead",
        -1,
    )
    assert first.status == "certain"
    assert first.frame_uncertainty == 2
    assert first.evidence["id"] == "edited-e001"
    let_bounce = early.events[2]
    assert (
        let_bounce.source_event_id,
        let_bounce.adopted_in_v4,
        let_bounce.in_point,
    ) == ("e003", True, False)
    assert let_bounce.excluded_reason is None
    excluded = early.events[-1]
    assert (excluded.source_event_id, excluded.adopted_in_v4, excluded.in_point) == (
        "e008",
        False,
        False,
    )
    assert excluded.excluded_reason == "bounce after terminal net"
    assert excluded.evidence["old_in_point"] is True
    assert excluded.subtype is None
    assert early.coverage[32:36].tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    "pattern",
    [
        "scene.npz",
        "scene.metadata.json",
        "event_detection.json",
        "array_0000.npy",
        "clip.json",
    ],
)
def test_corrupted_v4_files_fail_before_decoding(
    inputs: IngestFixture, pattern: str
) -> None:
    path = next(inputs.root.rglob(pattern))
    path.write_bytes(path.read_bytes() + b"extra unrecorded bytes")
    with pytest.raises(ValueError, match=r"SHA-256 mismatch"):
        read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256)


def test_sha256_inventory_must_agree_with_manifest(inputs: IngestFixture) -> None:
    path = next(inputs.root.rglob("scene.npz"))
    path.write_bytes(path.read_bytes() + b"updated inventory without updating manifest")
    write_sums(inputs.root)
    with pytest.raises(ValueError, match=r"SHA256SUMS disagrees.*scene\.npz"):
        read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256)


def test_split_cannot_assign_a_clip_twice(inputs: IngestFixture) -> None:
    path = inputs.root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["splits"]["val"]["clip_ids"] = ["video_000/clip_009"]
    write_json(path, manifest)
    write_sums(inputs.root)
    with pytest.raises(ValueError, match="repeated split assignment"):
        read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256)


def test_default_cli_rejects_a_split_swap_even_with_rewritten_checksums(
    inputs: IngestFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = inputs.root / "manifest.json"
    sums = inputs.root / "SHA256SUMS"
    monkeypatch.setattr(v4_reader, "MEIJI_V4_MANIFEST_SHA256", inputs.manifest_sha256)
    monkeypatch.setattr(v4_reader, "MEIJI_V4_SHA256SUMS_SHA256", digest(sums))
    assert main(["audit", *inputs.cli_args(alternate=False)]) == 0
    capsys.readouterr()

    manifest = json.loads(path.read_bytes())
    manifest["splits"]["train"]["clip_ids"] = ["video_000/clip_030"]
    manifest["splits"]["test"]["clip_ids"] = ["video_000/clip_009"]
    write_json(path, manifest)
    write_sums(inputs.root)
    for command, extra in (("audit", []), ("ingest", ["--owner-team", "1"])):
        assert main([command, *inputs.cli_args(alternate=False), *extra]) == 1
        assert "V4 manifest SHA-256 mismatch" in capsys.readouterr().err
    assert main(["audit", *inputs.cli_args()]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["clips"][0]["split"] == "test"


def test_frozen_pin_also_checks_the_checksum_inventory(
    inputs: IngestFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    sums = inputs.root / "SHA256SUMS"
    monkeypatch.setattr(v4_reader, "MEIJI_V4_MANIFEST_SHA256", inputs.manifest_sha256)
    monkeypatch.setattr(v4_reader, "MEIJI_V4_SHA256SUMS_SHA256", digest(sums))
    sums.write_bytes(sums.read_bytes() + f"{'0' * 64}  other.txt\n".encode())
    with pytest.raises(ValueError, match="SHA256SUMS SHA-256 mismatch"):
        read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("time", "source time mismatch"),
        ("camera", "source path or offset disagrees"),
        ("count", "score chain or server order"),
        ("anchor", "score chain or server order"),
        ("missing", "clip IDs differ"),
    ],
)
def test_score_join_rejects_bad_inputs(
    inputs: IngestFixture, mutation: str, error: str
) -> None:
    document, project, clips = _project_clips(inputs)
    if mutation == "time":
        clips[1]["end_sec"] += 0.001
    elif mutation == "camera":
        project["sources"][0]["offset_sec"] = 1.252
    elif mutation == "missing":
        project["clips"].remove(clips[0])
    else:
        clips[1]["score"]["before"]["points"] = [0, 1]
        if mutation == "anchor":
            clips[1]["score"]["before_source"] = "anchor"
    write_json(inputs.projects, document)
    with pytest.raises(ValueError, match=error):
        _scores(inputs)


def test_projects_digest_is_checked_against_the_bytes_used_for_scores(
    inputs: IngestFixture,
) -> None:
    old_hash = digest(inputs.projects)
    inputs.projects.write_bytes(inputs.projects.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"projects.json SHA-256 mismatch"):
        read_scores(
            read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256),
            inputs.projects,
            inputs.resolver(),
            projects_sha256=old_hash,
        )


def test_score_completion_uses_the_match_rules(inputs: IngestFixture) -> None:
    document, project, clips = _project_clips(inputs)
    project["match"]["rules"]["match_games"] = 1
    for clip, serving_team, points in zip(
        clips, ("A", "B", "B"), ([5, 0], [6, 0], [6, 1]), strict=True
    ):
        clip["score"]["server"] = serving_team
        clip["score"]["before"]["points"] = points
    clips[-1]["score"]["winner"] = "A"
    write_json(inputs.projects, document)
    scored = _scores(inputs)
    assert scored.reached_match_end is True
    assert scored.match_complete is False
    assert scored.chain_continuous is False
    final_score = scored.clips[-1].score_after
    assert final_score is not None and final_score.games == (1, 0)
    assert scored.chain_links == 2

    clips[0]["score"]["before"]["points"] = [6, 1]
    clips[0]["score"]["server"] = "B"
    write_json(inputs.projects, document)
    with pytest.raises(ValueError, match="confirmed point after match completion"):
        _scores(inputs)


@pytest.mark.parametrize(
    ("missing", "links", "opens"),
    [((), 6, True), ((0,), 5, False), ((3,), 4, True), (tuple(range(6)), 0, False)],
    ids=("full-ledger", "opening-missing", "middle-missing", "last-anchor-only"),
)
def test_terminal_score_and_complete_match_ledger_are_independent(
    tmp_path: Path, missing: tuple[int, ...], links: int, opens: bool
) -> None:
    inputs = make_ingest_fixture(tmp_path, full_match=True)
    document, _, clips = _project_clips(inputs)
    for index in missing:
        clips[index]["score"] = None
    for index, clip in enumerate(clips[1:], 1):
        if index - 1 in missing and clip["score"] is not None:
            clip["score"]["before_source"] = "anchor"
    write_json(inputs.projects, document)

    report = audit_report(prepare_ingest(_scores(inputs), None, {}))
    chain = report["score_chain"]
    assert chain["starts_at_match_open"] is opens
    assert chain["reached_match_end"] is True
    assert chain["match_complete"] is (not missing)
    assert chain["status"] == ("partial" if missing else "continuous")
    assert chain["verified_links"] == links


def test_a_later_tennis_set_with_zero_game_and_point_counts_is_not_match_open(
    inputs: IngestFixture,
) -> None:
    document, project, clips = _project_clips(inputs)
    project["match"]["rules"] = {
        "type": "tennis",
        "best_of": 3,
        "no_ad": False,
        "final_set": "tie_break",
    }
    for clip in clips:
        clip["score"]["before"]["sets"] = [1, 0]
    write_json(inputs.projects, document)

    chain = audit_report(prepare_ingest(_scores(inputs), None, {}))["score_chain"]
    assert chain["starts_at_match_open"] is False
    assert chain["status"] == "partial"
    assert chain["verified_links"] == 2


@pytest.mark.parametrize(
    "status", [None, "auto", "unknown", "not_point", "multiple_points"]
)
def test_unconfirmed_clips_are_audited_without_inventing_a_point(
    inputs: IngestFixture, status: str | None
) -> None:
    document, _, clips = _project_clips(inputs)
    score = clips[1]["score"]
    if status is None:
        clips[1]["score"] = None
    else:
        score.update(status=status, winner=None, note="manually flagged")
    if status == "not_point":
        del score["before"]
        clips[2]["score"]["before"]["points"] = [1, 0]
    else:
        clips[2]["score"]["before_source"] = "anchor"
    write_json(inputs.projects, document)
    report = audit_report(prepare_ingest(_scores(inputs), None, {}))
    middle = report["clips"][1]
    assert middle["score_status"] == status
    assert middle["match_point_key"] is None
    assert all(
        middle[key] is None
        for key in ("score_before", "score_after", "server_team", "winner_team")
    )
    assert middle["terminal_cause"] == "unknown"
    assert report["columns"]["winner_team"]["missing_by_reason"] == {
        status or "missing_score": 1
    }
    assert report["match_point_count"] == 2


def test_overlap_deduplicates_the_point_without_advancing_the_score_twice(
    tmp_path: Path,
) -> None:
    inputs = make_ingest_fixture(tmp_path, duplicate="shared_serve")
    video = _scores(inputs)
    primary, duplicate, following, _ = video.clips
    assert primary.match_point_key == duplicate.match_point_key
    assert primary.score_after == duplicate.score_after
    assert duplicate.match_point_key != following.match_point_key
    assert video.chain_links == 2
    report = audit_report(prepare_ingest(video, None, {}))
    assert report["clip_count"] == 4
    assert report["match_point_count"] == 3

    document, _, clips = _project_clips(inputs)
    clips[1]["score"]["winner"] = "B"
    write_json(inputs.projects, document)
    with pytest.raises(
        ValueError, match="duplicate point has conflicting confirmed scores"
    ):
        _scores(inputs)


def test_overlap_after_serve_matches_source_events_and_reports_cross_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = make_ingest_fixture(
        tmp_path, duplicate="after_serve", duplicate_split="test"
    )
    scored = _scores(inputs)
    original, duplicate = scored.clips[:2]
    assert [event.subtype for event in duplicate.v4.events] == [
        "bounce",
        "groundstroke_forehand",
        "net_fault",
    ]
    assert original.match_point_key == duplicate.match_point_key
    assert original.score_after == duplicate.score_after
    assert scored.chain_links == 2
    assert main(["audit", *inputs.cli_args()]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cross_split_point_count"] == 1
    assert report["cross_split_point_keys"] == [original.match_point_key]
    assert report["match_point_count"] == 3
    assert main(["audit", *inputs.cli_args(), "--format", "csv"]) == 0
    rows = {
        row["key"]: row["data"]
        for row in csv.DictReader(StringIO(capsys.readouterr().out))
        if row["section"] == "summary"
    }
    assert rows["cross_split_point_count"] == "1"
    assert json.loads(rows["cross_split_point_keys"]) == [original.match_point_key]


def test_overlap_without_certain_shared_events_reports_pending_point_identity(
    tmp_path: Path,
) -> None:
    inputs = make_ingest_fixture(
        tmp_path, duplicate="ambiguous", duplicate_split="test"
    )
    report = audit_report(prepare_ingest(_scores(inputs), None, {}))
    duplicate = report["clips"][1]
    assert duplicate["score_status"] == "confirmed"
    assert duplicate["score_after"] == {"games": [0, 0], "points": [1, 0]}
    assert duplicate["match_point_key"] is None
    assert duplicate["point_identity_hold"] == {
        "reason": "overlapping_same_score_without_shared_event",
        "candidate_clip_ids": ["video_000/clip_009"],
    }
    assert report["pending_point_identity_clip_ids"] == ["video_000/clip_010"]
    assert report["columns"]["match_point_key"]["available"] == 3
    assert report["match_point_count"] == 3
    assert report["cross_split_point_count"] == 0
    assert report["score_chain"]["status"] == "partial"
    assert report["score_chain"]["verified_links"] == 2


def test_terminal_derivation_requires_certain_labels_and_winner_consistency(
    inputs: IngestFixture,
) -> None:
    net, unknown, double = _scores(inputs).clips
    derived = derive_terminal_cause(net)
    assert derived.cause == "net_error"
    assert derived.event_ids == ("e004", "e006", "e007")
    assert derive_terminal_cause(unknown).cause == "unknown"
    derived = derive_terminal_cause(double)
    assert derived.cause == "double_fault"
    assert derived.event_ids == ("e201", "e202", "e204", "e205", "e206", "e207")

    uncertain = tuple(
        replace(event, status="probable") if event.source_event_id == "e007" else event
        for event in net.v4.events
    )
    assert (
        derive_terminal_cause(replace(net, v4=replace(net.v4, events=uncertain))).cause
        == "unknown"
    )
    assert net.recorded_score is not None
    contradictory = replace(net, recorded_score=replace(net.recorded_score, winner="B"))
    assert (
        derive_terminal_cause(contradictory).missing_reason
        == "terminal_contact_conflicts_with_winner"
    )

    last_attempt = double.v4.events[-2:]
    ambiguous = replace(double, v4=replace(double.v4, events=last_attempt))
    assert derive_terminal_cause(ambiguous).cause == "unknown"

    after_let = tuple(
        event for event in net.v4.events if event.source_event_id != "e004"
    )
    dead_ball = replace(net, v4=replace(net.v4, events=after_let))
    assert derive_terminal_cause(dead_ball).cause == "unknown"


@pytest.mark.parametrize("extension", ["json", "csv"])
def test_manual_point_supplement_overrides_only_reviewed_fields(
    inputs: IngestFixture, extension: str
) -> None:
    path = inputs.data_root / f"reviews.{extension}"
    row = {
        "clip_id": "video_000/clip_009",
        "point_key": "point",
        "reviewer": "Reviewer",
        "terminal_cause": "out_wide",
        "error_assessment": "forced",
        "note": "Checked clip",
        "serve_attempts": [
            {
                "source_event_id": "e001",
                "result": "let",
                "serve_number": 1,
                "replay_index": 0,
            },
            {
                "source_event_id": "e004",
                "result": "in",
                "serve_number": 1,
                "replay_index": 1,
            },
        ],
    }
    if extension == "json":
        write_json(path, {"version": 1, "points": [row]})
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(
                {**row, "serve_attempts": json.dumps(row["serve_attempts"])}
            )
    batch = prepare_ingest(_scores(inputs), *read_supplement(path))
    first = batch.clips[0]
    assert batch.manual_supplement_sha256 == digest(path)
    assert first.terminal.cause == "net_error"
    assert (first.terminal_cause, first.terminal_cause_source) == (
        "out_wide",
        "manual_confirmed",
    )
    assert (first.error_assessment, first.error_assessment_source) == (
        "forced",
        "manual_confirmed",
    )
    assert batch.clips[1].terminal_cause == "unknown"
    assert (
        first.manual_attempts is not None
        and first.manual_attempts.serve_attempts[1].replay_index == 1
    )


@pytest.mark.parametrize(
    ("attempts", "message"),
    [
        ([{"result": "fault", "source_event_id": "e003"}], "in-point V4 serve"),
        (
            [{"result": "in", "source_event_id": "e004"}, {"result": "let"}],
            "conflicts with the preceding serve results",
        ),
        (
            [
                {"result": "let", "serve_number": 1},
                {"result": "fault", "serve_number": 2},
            ],
            "conflicts with the preceding serve results",
        ),
        (
            [{"result": "let"}, {"result": "in", "replay_index": 0}],
            "conflicts with the preceding serve results",
        ),
        (
            [
                {"result": "unknown", "serve_number": 2},
                {"result": "in", "serve_number": 1},
            ],
            "conflicts with the preceding serve results",
        ),
        ([{"result": "fault", "serve_number": True}], "serve_number"),
    ],
)
def test_manual_attempt_input_rejects_invalid_references_and_transitions(
    inputs: IngestFixture,
    attempts: list[dict[str, Any]],
    message: str,
) -> None:
    path = inputs.data_root / "manual.json"
    write_json(
        path,
        {
            "version": 1,
            "points": [
                {
                    "clip_id": "video_000/clip_009",
                    "point_key": "point",
                    "reviewer": "Reviewer",
                    "serve_attempts": attempts,
                }
            ],
        },
    )
    with pytest.raises(ValueError, match=message):
        prepare_ingest(_scores(inputs), *read_supplement(path))


@pytest.mark.parametrize(
    ("clip", "cause", "results", "message"),
    [
        ("clip_009", "double_fault", [], "double fault must lose the point"),
        ("clip_012", "double_fault", [(2, "in")], "confirmed faults on both"),
        ("clip_012", "double_fault", [(1, "in")], "confirmed faults on both"),
        ("clip_012", "double_fault", [(2, "fault")], "confirmed faults on both"),
        (
            "clip_012",
            "out_wide",
            [(2, "fault")],
            "conflicts with terminal cause out_wide",
        ),
        ("clip_009", None, [(2, "fault")], "second serve fault must lose the point"),
        (
            "clip_030",
            None,
            [(2, "in")],
            "legal serve conflicts with terminal cause double_fault",
        ),
        ("clip_012", "ace", [(2, "fault")], "Invalid manual terminal cause"),
    ],
    ids=(
        "server-won",
        "second-in",
        "first-in",
        "single-fault",
        "out-and-second-fault",
        "second-fault-and-server-won",
        "v4-double-fault-and-legal-serve",
        "unsupported-ace",
    ),
)
def test_manual_terminal_cause_matches_attempts_and_projects_winner(
    inputs: IngestFixture,
    clip: str,
    cause: str | None,
    results: list[tuple[int, str]],
    message: str,
) -> None:
    path = inputs.data_root / "manual.json"
    write_json(
        path,
        {
            "version": 1,
            "points": [
                {
                    "clip_id": f"video_000/{clip}",
                    "point_key": "point",
                    "reviewer": "Reviewer",
                    "terminal_cause": cause,
                    "serve_attempts": [
                        {"serve_number": number, "result": result}
                        for number, result in results
                    ],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match=message):
        prepare_ingest(_scores(inputs), *read_supplement(path))


@pytest.mark.parametrize(
    "with_attempts", [False, True], ids=("cause-only", "both-faults-with-let")
)
def test_manual_double_fault_can_be_confirmed_for_a_receiver_win(
    inputs: IngestFixture, with_attempts: bool
) -> None:
    path = inputs.data_root / "manual.json"
    attempts = [
        {
            "serve_number": 1,
            "result": "fault",
            "source_event_id": "e090",
            "fault_cause": "net",
        },
        {"serve_number": 2, "result": "let", "replay_index": 0},
        {
            "serve_number": 2,
            "result": "fault",
            "replay_index": 1,
            "fault_cause": "out_wide",
        },
    ]
    write_json(
        path,
        {
            "version": 1,
            "points": [
                {
                    "clip_id": "video_000/clip_012",
                    "point_key": "point",
                    "reviewer": "Reviewer",
                    "terminal_cause": "double_fault",
                    "serve_attempts": attempts if with_attempts else [],
                }
            ],
        },
    )
    point = prepare_ingest(_scores(inputs), *read_supplement(path)).clips[1]
    assert point.terminal_cause == "double_fault"
    assert point.terminal_cause_source == "manual_confirmed"


def test_cli_audit_is_deterministic_and_recovers_counts_reasons_and_ids(
    inputs: IngestFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    formats = {}
    for format in ("json", "csv"):
        command = ["audit", *inputs.cli_args(), "--format", format]
        assert main(command) == 0
        first = capsys.readouterr()
        assert main(command) == 0
        assert capsys.readouterr() == first
        formats[format] = first.out
    report = json.loads(formats["json"])
    assert report["input_hashes"]["v4_sha256sums_sha256"] == digest(
        inputs.root / "SHA256SUMS"
    )
    assert report["input_hashes"]["projects_json_sha256"] == digest(inputs.projects)
    assert report["clip_count"] == 3
    assert report["time_alignment"]["max_error_sec"] == 0
    assert report["score_chain"]["verified_links"] == 2
    assert report["score_chain"]["match_complete"] is False
    assert report["columns"]["terminal_cause"] == {
        "available": 2,
        "total": 3,
        "rate": 2 / 3,
        "missing_by_reason": {"no_terminal_net_fault_label": 1},
    }
    assert report["event_counts"]["adopted"] == 18
    assert report["event_counts"]["excluded_by_reason"] == {
        "bounce after terminal net": 1
    }
    assert report["event_counts"]["by_subtype"]["smash"] == 0
    assert report["clips"][0]["excluded_source_events"] == [
        {"source_event_id": "e008", "reason": "bounce after terminal net"}
    ]
    assert report["clips"][0]["score_after"] == {"games": [0, 0], "points": [1, 0]}
    rows = {
        (row["section"], row["key"]): row["data"]
        for row in csv.DictReader(StringIO(formats["csv"]))
    }
    assert json.loads(rows["summary", "input_hashes"]) == report["input_hashes"]
    assert (
        json.loads(rows["column", "terminal_cause"])
        == report["columns"]["terminal_cause"]
    )
    assert json.loads(rows["clip", "video_000/clip_009"]) == report["clips"][0]
