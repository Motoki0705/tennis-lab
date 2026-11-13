"""Real Git and public entry contracts with an offline GitHub transport."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.worker_logs import LogEnv, git, journal, schema, sources


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LogEnv:
    return LogEnv(tmp_path, monkeypatch)


class Crash(BaseException):
    """Emulate process loss without allowing the publisher's error handler to run."""


def advance_main(logs: LogEnv, files: dict[str, str]) -> str:
    """Make an independent upstream change without touching a log PR's branch."""
    git(logs.root, "fetch", "origin")
    git(logs.root, "merge", "--ff-only", "origin/main")
    for name, content in files.items():
        target = logs.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git(logs.root, "add", "--", *files)
    git(logs.root, "commit", "-qm", "independent main change")
    git(logs.root, "push", "origin", "main")
    return git(logs.root, "rev-parse", "HEAD")


def test_capture_adapter_failure_preserves_successful_log_publication(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture: Any = importlib.import_module("decision_logs.capture")
    observations: Any = importlib.import_module("decision_logs.observations")
    service: Any = importlib.import_module("worker_logs.service")
    path = logs.ready()
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )

    def invalid_capture_metadata(raw: bytes) -> Any:
        raise ValueError("injected capture metadata conversion failure")

    monkeypatch.setattr(observations, "decode", invalid_capture_metadata)
    value = service.publish(path)
    assert value["saved_to_main"] is True and value["state"] == "merged"
    assert logs.github.calls == ["create", "merge"]
    view = capture.status(logs.state, "o/r", 1)
    assert view["status"] == "incomplete"
    assert view["failures"] == [
        {"source": "worker_logs.publish", "error": "capture_delivery_missing"}
    ]
    # Delivery retry cannot fabricate the missing conversion or rerun publication.
    assert capture.status(logs.state, "o/r", 1, retry=True)["status"] == "incomplete"
    assert logs.github.calls == ["create", "merge"]


def test_gitignore_tracks_only_public_worker_log_files(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(
        (Path(__file__).resolve().parents[3] / ".gitignore").read_bytes()
    )
    directory = ".agents/logs/parents/parent-1/child-10/d-example/0001-initial"
    allowed = {
        f"{directory}/{name}" for name in ("instruction.md", "final.md", "meta.json")
    }
    allowed.update(
        {
            ".agents/logs/parents/README.md",
            ".agents/logs/parents/parent-1/README.md",
            ".agents/logs/parents/parent-1/child-10/README.md",
            ".agents/logs/parents/parent-1/child-10/d-example/README.md",
            f"{directory}/README.md",
            ".agents/logs/parents/parent-1/activity-example/README.md",
            ".agents/logs/parents/parent-1/activity-example/event-example.json",
        }
    )
    ignored = {
        ".agents/logs/raw.md",
        ".agents/logs/instruction.md",
        ".agents/logs/final.md",
        ".agents/logs/meta.json",
        ".agents/logs/notes/meta.json",
        ".agents/logs/parents/raw.md",
        ".agents/logs/parents/parent-1/private.txt",
        f"{directory}/raw.md",
        f"{directory}/stderr.log",
        f"{directory}/events.json",
        f"{directory}/payload.txt",
        f"{directory}/code.py",
        ".agents/logs/parents/README.txt",
        ".agents/logs/parents/raw/README.md",
        ".agents/logs/parents/parent-1/activity-example/private.json",
        ".agents/logs/parents/parent-1/private/event-example.json",
    }
    for name in allowed | ignored:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n")
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=tmp_path,
        input="\0".join(sorted(allowed | ignored)) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.rstrip("\0").split("\0")) == ignored
    git(tmp_path, "add", ".")
    assert set(git(tmp_path, "ls-files").splitlines()) == {".gitignore", *allowed}


@pytest.mark.parametrize(
    "point",
    [
        "before_commit",
        "after_commit",
        "after_push",
        "create_response_lost",
        "merge_response_lost",
        "closed_unmerged",
    ],
)
def test_publication_recovers_after_interruption(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    path = logs.ready()
    publisher = logs.publisher(path)
    before = logs.main_files()
    if point in {"before_commit", "after_commit", "after_push"}:
        real = publisher.git
        fired = False

        def interrupted(*args: str, **kwargs: Any) -> bytes:
            nonlocal fired
            if not fired and point == "before_commit" and args[0] == "add":
                fired = True
                raise Crash()
            result: bytes = real(*args, **kwargs)
            if not fired and args[0] == {
                "after_commit": "commit",
                "after_push": "push",
            }.get(point):
                fired = True
                raise Crash()
            return result

        monkeypatch.setattr(publisher, "git", interrupted)
        with pytest.raises(Crash):
            publisher.publish()
        monkeypatch.setattr(publisher, "git", real)
        assert logs.main_files() == before
        if point == "before_commit":
            assert publisher.publish()["error"] == "worktree_dirty"
            row = journal.read(path)
            tree = publisher.worktree(row)
            # Explicit repair of this fixture's interrupted worktree only.
            git(tree, "reset", "--hard", row["base"])
            git(tree, "clean", "-fd")
    else:
        logs.github.failure = point
        first = publisher.publish()
        assert first["saved_to_main"] is False
        if point == "merge_response_lost":
            saved = journal.read(path)
            premature = {**saved, "state": "merged"}
            journal.atomic(path, premature)
            corrupted = path.read_bytes()
            with pytest.raises(journal.PublicationError, match="journal_invalid"):
                journal.read(path)
            assert path.read_bytes() == corrupted
            journal.atomic(path, saved)
        if point == "closed_unmerged":
            assert first["error"] == "closed_unmerged"
            assert publisher.publish()["saved_to_main"] is False
            assert logs.main_files() == before and logs.github.calls == ["create"]
            return
    result = publisher.publish(retry=True)
    assert result["state"] == "merged", result
    assert result["saved_to_main"] is True
    assert logs.github.calls.count("create") == logs.github.calls.count("merge") == 1
    assert publisher.publish()["saved_to_main"] is True
    assert logs.github.calls.count("create") == logs.github.calls.count("merge") == 1
    row = journal.read(path)
    assert set(logs.main_files()) - set(before) == {
        row["directory"] + "/" + n for n in schema.FILENAMES
    }
    assert "[skip ci]" not in git(logs.upstream, "log", "-3", "--format=%s")


def test_two_parents_append_and_merge_without_shared_index(logs: LogEnv) -> None:
    paths = [logs.ready(parent=1, child=10), logs.ready(parent=2, child=20)]
    before = set(logs.main_files())
    # Both PRs were created before either publication advanced main. GitHub
    # retains their original base.sha even after the first one merges.
    logs.github.pause_reads = True
    for path in paths:
        assert logs.publisher(path).publish()["state"] == "pending"
    for pr in logs.github.rows.values():
        pr["mergeable"] = True
    logs.github.pause_reads = False
    for path in paths:
        assert logs.publisher(path).publish()["state"] == "merged"
    added = set(logs.main_files()) - before
    assert len(added) == 6 and all(p.startswith(".agents/logs/parents/") for p in added)
    assert not any(p.endswith("index.md") for p in added)
    assert logs.github.calls == ["create", "create", "merge", "merge"]


@pytest.mark.parametrize("state", ["ready", "pending"])
def test_legacy_prepared_mask_order_resumes_without_rewriting_files(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    guard = importlib.import_module("worker_logs.safety").Guard
    original_init = guard.__init__

    class LegacyPrivateOrder(set[str]):
        def __iter__(self) -> Any:
            # One order the historical set iteration can produce, independent
            # of this pytest process's random hash seed.
            return iter(sorted(super().__iter__(), key=lambda v: v != "rivate-abc"))

    def legacy_order(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.private = LegacyPrivateOrder(self.private)

    monkeypatch.setattr(guard, "__init__", legacy_order)
    path, proof = logs.turn(instruction="private-abc")
    captured = journal.read(path)
    captured["source"]["private"].extend(["private-ab", "rivate-abc"])
    journal.atomic(path, captured)
    assert sources.receive(path, proof)["state"] == "ready"
    row = journal.read(path)
    # The historical ready/pending bytes from the review reproduction.
    row["files"]["instruction.md"] = "p[private identifier]"
    meta = json.loads(row["files"]["meta.json"])
    meta["files"]["instruction.md"] = schema.sha(
        row["files"]["instruction.md"].encode()
    )
    row["files"]["meta.json"] = schema.json_bytes(meta).decode()
    row["state"] = state
    journal.atomic(path, row)
    fixed = dict(row["files"])
    published = logs.publisher(path).publish()
    assert published["state"] == "merged", published
    after = journal.read(path)
    assert after["files"] == fixed
    for name, value in fixed.items():
        assert (
            subprocess.check_output(
                ["git", "show", f"main:{after['directory']}/{name}"], cwd=logs.upstream
            )
            == value.encode()
        )


@pytest.mark.parametrize("previous_state", ["pending", "blocked"])
def test_stale_pr_base_merges_without_changing_pinned_record(
    logs: LogEnv, previous_state: str
) -> None:
    path = logs.ready()
    publisher = logs.publisher(path)
    logs.github.pause_reads = True
    pending = publisher.publish()
    logs.github.pause_reads = False
    assert pending["state"] == "pending" and pending["saved_to_main"] is False
    original = journal.read(path)
    number = original["pr_number"]
    advanced = advance_main(logs, {"code.py": "value = 2\n"})
    assert logs.github.get(number)["base"]["sha"] == original["base"] != advanced
    if previous_state == "blocked":
        # Old publishers did not leave a diagnostic for this base-SHA-only error.
        original.pop("diagnostic", None)
        journal.fail(path, original, "pr_identity_changed")
        assert "diagnostic" not in journal.read(path)
        assert publisher.publish()["state"] == "blocked"
    logs.github.rows[number]["mergeable"] = True
    result = publisher.publish(retry=previous_state == "blocked")
    assert result["state"] == "merged" and result["saved_to_main"] is True, result
    assert pending["stage"] == "pr_open" and pending["error"] == "github_failed"
    row = journal.read(path)
    for key in ("source", "receipt", "files", "base", "head", "branch", "pr_number"):
        assert row[key] == original[key]
    assert logs.github.calls == ["create", "merge"]
    assert git(logs.upstream, "show", "main:code.py") == "value = 2"
    assert git(logs.upstream, "merge-base", "--is-ancestor", advanced, "main") == ""
    assert git(logs.upstream, "merge-base", "--is-ancestor", row["head"], "main") == ""
    for name, content in row["files"].items():
        published = subprocess.check_output(
            ["git", "show", f"main:{row['directory']}/{name}"], cwd=logs.upstream
        )
        assert published == content.encode("utf-8")
    diagnostic = json.loads(row["diagnostic"])
    assert diagnostic == {
        "reason": "stale_pr_base_sha",
        "pr_base": original["base"],
        "base": original["base"],
        "main": advanced,
    }
    assert "diagnostic" not in result
    assert publisher.publish()["saved_to_main"] is True
    assert logs.github.calls == ["create", "merge"]


@pytest.mark.parametrize(
    ("change", "error", "diagnostic"),
    [
        ("non_ancestor", "base_not_ancestor", "base"),
        ("base_ref", "pr_identity_changed", "base.ref"),
        ("branch", "pr_identity_changed", "head.ref"),
        ("head", "pr_identity_changed", "head.sha"),
        ("marker", "pr_identity_changed", "body"),
        ("existing_directory", "record_already_exists", None),
        ("github_extra", "unsafe_diff", None),
        ("github_replaced_file", "unsafe_diff", None),
    ],
)
def test_blocked_retry_rechecks_identity_and_addition_only_policy(
    logs: LogEnv,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    error: str,
    diagnostic: str | None,
) -> None:
    if change == "non_ancestor":
        advance_main(logs, {"pinned-base.txt": "must remain an ancestor\n"})
    path = logs.ready()
    publisher = logs.publisher(path)
    logs.github.pause_reads = True
    assert publisher.publish()["state"] == "pending"
    logs.github.pause_reads = False
    row = journal.read(path)
    pr = logs.github.rows[row["pr_number"]]
    pr["mergeable"] = True
    journal.fail(path, row, "pr_identity_changed")
    if change == "non_ancestor":
        git(logs.upstream, "update-ref", "refs/heads/main", logs.base)
        # A normal fetch refreshes the remote-tracking ref after a rewrite.
        # The publisher must reject the now-unrelated pinned base itself.
        git(logs.root, "fetch", "origin")
    elif change == "existing_directory":
        advance_main(logs, {row["directory"] + "/instruction.md": "existing record\n"})
    elif change.startswith("github_"):
        changes = logs.github.files(row["pr_number"])
        if change == "github_extra":
            changes.append({"filename": "code.py", "status": "modified"})
        else:
            changes[0] = {"filename": "outside.txt", "status": "added"}
        monkeypatch.setattr(logs.github, "files", lambda number: changes)
    else:
        fields = {
            "base_ref": (pr["base"], "ref", "release"),
            "branch": (pr["head"], "ref", "other-branch"),
            "head": (pr["head"], "sha", "f" * 40),
            "marker": (pr, "body", "different marker"),
        }
        target, key, value = fields[change]
        target[key] = value
    main_before = logs.github.main()
    result = publisher.publish(retry=True)
    assert result["state"] == "blocked" and result["error"] == error, result
    assert result["saved_to_main"] is False
    assert logs.github.main() == main_before and logs.github.calls == ["create"]
    if diagnostic is not None:
        assert diagnostic in journal.read(path)["diagnostic"]
    assert "diagnostic" not in result


@pytest.mark.parametrize("unknown_at", ["created", "fresh"])
def test_unknown_mergeability_still_merges_pinned_head(
    logs: LogEnv,
    monkeypatch: pytest.MonkeyPatch,
    unknown_at: str,
) -> None:
    path = logs.ready()
    get = logs.github.get

    def become_unknown(number: int) -> dict[str, Any]:
        value = get(number)
        if logs.github.gets == 2 and unknown_at == "fresh":
            value["mergeable"] = None
        return value

    monkeypatch.setattr(logs.github, "get", become_unknown)
    if unknown_at == "created":
        logs.github.mergeable = None
    result = logs.publisher(path).publish()
    assert result["state"] == "merged" and result["saved_to_main"] is True
    assert result["error"] is None
    assert logs.github.calls == ["create", "merge"]
    assert logs.publisher(path).publish()["saved_to_main"] is True


def test_concurrent_retry_reuses_same_pr(logs: LogEnv) -> None:
    path = logs.ready()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: logs.publisher(path).publish(), range(2)))
    assert all(r["state"] == "merged" for r in results), results
    assert logs.github.calls == ["create", "merge"]


@pytest.mark.parametrize("previously_blocked", [False, True])
def test_local_path_masks_are_published_and_visible_to_parent(
    logs: LogEnv, previously_blocked: bool
) -> None:
    instruction = "観測/補間。保存先:[資料](/資料/非公開.txt)。\n"
    final = r"確認は~/.ssh/config と C:\Users\unknown\report.txt"
    path, proof = logs.turn(instruction=instruction)
    directory = Path(journal.read(path)["source"]["run_directory"])
    (directory / "final.md").write_text(final)
    ready = sources.receive(path, proof)
    expected = {
        "version": 2,
        "instruction": 1,
        "final": 2,
        "additional": 0,
        "local_paths": {"posix": 1, "home": 1, "windows": 1},
    }
    assert ready["state"] == "ready" and ready["sanitization"] == expected
    row = journal.read(path)
    # Original inputs stay private and pinned, including on an old blocked turn's retry.
    assert row["source"]["instruction"] == instruction
    assert row["receipt"]["final"] == final
    if previously_blocked:
        del row["files"]
        journal.fail(path, row, "absolute_path")
    publisher = logs.publisher(path)
    result = publisher.publish(retry=previously_blocked)
    assert result["saved_to_main"] is True and result["sanitization"] == expected
    row = journal.read(path)
    published = {
        name: git(logs.upstream, "show", f"main:{row['directory']}/{name}")
        for name in schema.FILENAMES
    }
    assert published["instruction.md"] == "観測/補間。保存先:[資料]([local path])。"
    assert published["final.md"] == "確認は[local path] と [local path]"
    assert json.loads(published["meta.json"])["sanitization"] == expected
    assert journal.view(row)["sanitization"] == expected
    assert publisher.publish()["sanitization"] == expected
    assert logs.github.calls == ["create", "merge"]


@pytest.mark.parametrize("change", ["remote_head", "full_diff"])
def test_publication_gate_rechecks_git_and_remote(logs: LogEnv, change: str) -> None:
    path = logs.ready()
    publisher = logs.publisher(path)
    before = logs.main_files()
    if change == "remote_head":
        logs.github.change_head_on_second_get = True
    else:
        logs.github.pause_reads = True
        assert publisher.publish()["state"] == "pending"
        logs.github.pause_reads = False
        row = journal.read(path)
        tree = publisher.worktree(row)
        (tree / "code.py").write_text("unexpected = 2\n")
        git(tree, "add", "code.py")
        git(tree, "commit", "--amend", "--no-edit")
        row["head"] = git(tree, "rev-parse", "HEAD")
        logs.github.rows[row["pr_number"]]["head"]["sha"] = row["head"]
        journal.atomic(path, row)
    result = publisher.publish()
    assert result["error"] == (
        "pr_identity_changed" if change == "remote_head" else "unsafe_diff"
    )
    assert logs.main_files() == before
    assert "merge" not in logs.github.calls


@pytest.mark.parametrize(
    "failure", ["failed", "missing_final", "prompt_changed", "unsafe_input"]
)
def test_runner_failure_is_not_log_success(logs: LogEnv, failure: str) -> None:
    credential = "ghp_" + "x" * 30
    path, proof = logs.turn(
        instruction=credential if failure == "unsafe_input" else "調査指示。"
    )
    row = journal.read(path)
    directory = Path(row["source"]["run_directory"])
    if failure == "failed":
        proof["record"].update(result="failed", turn_end_reason="cli_error")
    elif failure == "missing_final":
        (directory / "final.md").unlink()
    elif failure == "prompt_changed":
        (directory / "prompt.md").write_text("different prompt")
    result = sources.receive(path, proof)
    assert result["saved_to_main"] is False
    assert (
        result["state"]
        == {
            "failed": "worker_failed",
            "missing_final": "missing_final",
            "prompt_changed": "blocked",
            "unsafe_input": "blocked",
        }[failure]
    )
    assert "files" not in journal.read(path)
    assert not logs.github.calls
    assert credential not in json.dumps(result)
    assert not any(p.startswith(".agents/logs/") for p in logs.main_files())


def test_standalone_worker_uses_same_log_entry(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = "opaque-fixture-credential"
    monkeypatch.setenv("worker_api_key", credential)
    instruction = (
        "\n実際に送った指示\r\n"
        + str(logs.root / ".claude/worktrees/worker-7-7/code.py")
        + f" private-parent-session {credential}\r\n"
    )
    path = logs.ready(parent=7, child=7, instruction=instruction)
    service: Any = importlib.import_module("worker_logs.service")
    session = logs.sessions[0]
    store = type(
        "Store",
        (),
        {
            "directory": logs.state,
            "read_data": lambda self: {"parents": {"o/r#7": session.row}},
        },
    )()
    github = type("GitHub", (), {"repo": "o/r"})()
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    before = path.read_bytes()
    journal.atomic(
        logs.state / "registry.json", {"version": 1, "parents": {"o/r#7": session.row}}
    )
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    environment.update(
        CODEX_SESSION_ID="private-parent-session",
        CODEX_THREAD_ID="private-parent-session",
        PYTHONPATH=str(Path(__file__).resolve().parents[3] / ".agents/ops"),
    )
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "parent",
            "--repo",
            "o/r",
            "--parent",
            "7",
            "--state-dir",
            str(logs.state),
            "publish-worker-log",
            "--child",
            "7",
            "--dry-run",
        ],
        cwd=logs.root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["logs"][0]["dry_run"] is True
    assert path.read_bytes() == before and not logs.github.calls
    preview = service.command(
        github,
        store,
        7,
        7,
        "codex:private-parent-session",
        publish_logs=True,
        dry_run=True,
    )
    assert path.read_bytes() == before and not logs.github.calls
    assert preview["logs"][0]["directory"].startswith(
        ".agents/logs/parents/parent-7/child-7/"
    )
    result = service.command(
        github, store, 7, 7, "codex:private-parent-session", publish_logs=True
    )
    assert result["logs"][0]["saved_to_main"] is True
    row = journal.read(path)
    meta = json.loads(row["files"]["meta.json"])
    assert meta["parent"] == meta["issue"] == 7
    assert (
        row["files"]["instruction.md"]
        == "\n実際に送った指示\r\n./code.py [private identifier] [private identifier]\r\n"
    )
    assert "親の計画判断" not in row["files"]["instruction.md"]
    public = json.dumps(logs.github.rows, ensure_ascii=False) + json.dumps(result)
    for name in logs.main_files():
        public += git(logs.upstream, "show", "main:" + name)
    assert str(logs.root) not in public and "private-parent-session" not in public
    assert credential not in public
