"""Review I/O probes acquire the real registry flock from another process."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_lock_scope import rows
from tests.e2e.agents_ops.test_parent_session import OPS, ReviewEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_recovery import gate_for, recovery_request

TRACE_IO = r"""
import fcntl, json, os, pathlib, sys
txn_directory = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
txn_held = False
with (txn_directory / "registry.lock").open("a") as txn_probe:
    try:
        fcntl.flock(txn_probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        txn_held = True
with open(os.environ["TXN_IO_TRACE"], "a") as txn_trace:
    txn_trace.write(json.dumps({"tool": os.environ.get("TXN_IO_TOOL", pathlib.Path(sys.argv[0]).name), "args": sys.argv[1:], "held": txn_held}) + "\n")
txn_mutation_method = os.environ.get("TXN_MUTATE_METHOD")
txn_matching_method = not txn_mutation_method or "--method" in sys.argv and sys.argv[sys.argv.index("--method") + 1] == txn_mutation_method
if os.environ.get("TXN_MUTATE_ACTIVE") and txn_matching_method and not pathlib.Path(os.environ["TXN_MUTATED"]).exists():
    assert not txn_held, "mutation probe could not acquire registry lock"
    from parent.state import Store as TxnStore
    with TxnStore(txn_directory).locked() as txn_store:
        txn_row = txn_store.parents["o/r#1"]
        txn_field = os.environ["TXN_MUTATE_FIELD"]
        if txn_field == "session":
            txn_row["session"] = "claude:changed-owner"
        elif txn_field == "status":
            txn_row["status"] = "released"
        elif txn_field == "generation":
            txn_row["generation"] += 1
        elif txn_field == "tasks":
            txn_row["tasks"]["10"]["stopped"] = True
        elif txn_field == "reviews":
            txn_row["reviews"][-1]["evidence"] = "concurrent review instruction"
        txn_store.save()
        pathlib.Path(os.environ["TXN_MUTATED"]).write_text(json.dumps(txn_store.data))
"""


def instrument(r: ReviewEnv) -> Path:
    p = r.p
    executable = p.root / "bin/gh"
    shebang, body = executable.read_text().split("\n", 1)
    executable.write_text(shebang + "\n" + TRACE_IO + body)
    real_git = shutil.which("git")
    assert real_git
    git = p.root / "bin/git"
    git.write_text(
        f"#!{sys.executable}\n"
        + TRACE_IO
        + f"\nos.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])\n"
    )
    git.chmod(0o700)
    trace = Path(p.root) / "io-trace.jsonl"
    p.env.update(TXN_IO_TRACE=str(trace), TXN_MUTATED=str(p.root / "mutated.json"))
    if r.clock_script is not None:
        content = r.clock_script.read_text()
        content = content.replace(
            "runpy.run_path(", evidence_probes() + "\nrunpy.run_path("
        )
        r.clock_script.write_text(content)
    return trace


def evidence_probes() -> str:
    return f"""
import functools, os, subprocess
sys.path.insert(0, {str(OPS / "lib")!r})
import review_recovery as txn_recovery
def txn_probe(function):
    @functools.wraps(function)
    def checked(*args, **kwargs):
        subprocess.run([sys.executable, "-c", {TRACE_IO!r}], check=True,
            env={{**os.environ, "TXN_IO_TOOL": function.__name__}})
        return function(*args, **kwargs)
    return checked
txn_recovery.file_digest = txn_probe(txn_recovery.file_digest)
txn_recovery.completion_evidence = txn_probe(txn_recovery.completion_evidence)
"""


def proof(r: ReviewEnv, *, whole_diff: bool = False) -> Path:
    snapshot = r.request["prs"][0]
    paths = (
        run_git(
            r.p.repo,
            "diff",
            "--name-only",
            snapshot["merge_base_sha"],
            snapshot["head_sha"],
        ).splitlines()
        if whole_diff
        else []
    )
    path = Path(r.p.root) / "proof.json"
    path.write_text(
        json.dumps(
            {
                "ci": "Python succeeded for this head",
                "regression_tests": "boundary tests passed",
                "resolution": "parent verified the change",
                "reviewed_paths": paths,
            }
        )
    )
    return path


def prepare_entry(r: ReviewEnv, entry: str) -> tuple[ReviewEnv, list[str]]:
    if entry == "run":
        return r, ["run", "--request", str(r.request_path), "--agent-exec", str(r.fake)]
    if entry == "complete-evidence":
        output = json.loads(r.output.read_text())
        output["prs"][0]["evidence_waits"] = [
            {
                "question": "boundary result",
                "required_evidence": "fixed head test result",
            }
        ]
        r.output.write_text(json.dumps(output))
    r.run()
    if entry == "recover":
        adopted = recovery_request(r)
        return adopted, [
            entry,
            "--request",
            str(adopted.request_path),
            "--source-run",
            r.request["review_run_id"],
            "--reason-file",
            r.p.note(),
        ]
    args = [entry, "--run", r.request["review_run_id"]]
    if entry == "close-interrupted":
        execution = r.run_dir / "execution.json"
        value = json.loads(execution.read_text())
        value["status"] = "running"
        execution.write_text(json.dumps(value))
        args += ["--reason-file", r.p.note()]
    elif entry == "complete-evidence":
        args += [
            "--pr",
            "100",
            "--wait",
            "1",
            "--evidence",
            str(proof(r, whole_diff=True)),
        ]
    elif entry == "resolve":
        args += [
            "--pr",
            "100",
            "--finding",
            "changes",
            "--fixed-head",
            r.request["prs"][0]["head_sha"],
            "--evidence",
            str(proof(r)),
        ]
    return r, args


ENTRIES = [
    "run",
    "resume-publication",
    "close-interrupted",
    "recover",
    "complete-evidence",
    "resolve",
]


@pytest.mark.parametrize("entry", ENTRIES)
def test_all_review_entry_io_releases_registry(
    single_review_env: ReviewEnv, entry: str
) -> None:
    r, args = prepare_entry(single_review_env, entry)
    trace = instrument(r)
    r.cli(*args)
    calls = rows(str(trace))
    assert any(c["tool"] == "gh" for c in calls)
    assert any(c["tool"] == "git" for c in calls)
    assert not [c for c in calls if c["held"]]


def test_runtime_replay_rejects_changed_frozen_events(
    single_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = single_review_env
    r.run()
    monkeypatch.syspath_prepend(str(OPS / "review"))
    monkeypatch.syspath_prepend(str(OPS / "lib"))
    recovery = importlib.import_module("review_recovery")
    state = importlib.import_module("parent.state")
    original = recovery.completion_evidence

    def changed(*args: Any, **kwargs: Any) -> Any:
        events = original(*args, **kwargs)
        path = r.run_dir / "agent-runtime/events.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        return events

    monkeypatch.setattr(recovery, "completion_evidence", changed)
    with pytest.raises(state.RegistryConflict, match="frozen review input changed"):
        recovery.runtime_evidence(r.run_dir, r.request)


@pytest.mark.parametrize(
    "entry,field",
    list(
        zip(
            ENTRIES,
            ["session", "status", "generation", "tasks", "reviews", "generation"],
            strict=True,
        )
    ),
)
def test_review_entry_conflict_preserves_concurrent_row(
    single_review_env: ReviewEnv, entry: str, field: str
) -> None:
    r, args = prepare_entry(single_review_env, entry)
    instrument(r)
    response = r.cli(
        *args, ok=False, extra={"TXN_MUTATE_ACTIVE": "1", "TXN_MUTATE_FIELD": field}
    )
    assert response.returncode == 2 and "registry changed" in response.stderr
    store = importlib.import_module("parent.state").Store(r.run_dir.parent.parent)
    assert store.read_data() == json.loads(Path(r.p.env["TXN_MUTATED"]).read_text())


STAGE_DRIVER = r"""
import datetime, os, sys
class FixedClock(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 18, tzinfo=tz)
datetime.datetime = FixedClock
import parent_review as pr
txn_authorizations = 0
txn_original_authorized = pr.authorized
def txn_authorized(*args, **kwargs):
    global txn_authorizations
    txn_authorizations += 1
    if txn_authorizations == int(os.environ["TXN_AUTHORIZATION"]):
        os.environ["TXN_MUTATE_ACTIVE"] = "1"
    return txn_original_authorized(*args, **kwargs)
pr.authorized = txn_authorized
sys.exit(pr.main())
"""


@pytest.mark.parametrize(
    "stage,field,authorizations",
    [
        ("before-agent", "tasks", 3),
        ("postcheck", "reviews", 4),
        ("publication", "status", 5),
        ("publication-post", "generation", 0),
        ("publication-patch", "reviews", 0),
    ],
)
def test_run_late_conflicts_exit_two_without_stale_result(
    single_review_env: ReviewEnv, stage: str, field: str, authorizations: int
) -> None:
    r = single_review_env
    instrument(r)
    assert r.clock_script is not None
    r.clock_script.write_text(
        f"import sys\nsys.path.insert(0, {str(OPS / 'review')!r})\n" + STAGE_DRIVER
    )
    extra = {"TXN_AUTHORIZATION": str(authorizations), "TXN_MUTATE_FIELD": field}
    if stage.startswith("publication-"):
        extra.update(
            TXN_MUTATE_ACTIVE="1", TXN_MUTATE_METHOD=stage.split("-")[1].upper()
        )
    response = r.run(ok=False, extra=extra)
    assert response.returncode == 2, response.stderr + response.stdout
    assert Path(r.p.env["TXN_MUTATED"]).exists(), stage
    store = importlib.import_module("parent.state").Store(r.run_dir.parent.parent)
    assert store.read_data() == json.loads(Path(r.p.env["TXN_MUTATED"]).read_text())
    assert len(r.calls()) == (0 if stage == "before-agent" else 1)
    execution = json.loads((r.run_dir / "execution.json").read_text())
    assert execution["status"] != "published"
    if not stage.startswith("publication"):
        assert not (r.run_dir / "result.json").exists()


def test_merge_git_and_evidence_verification_release_registry(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    r.run()
    gate = gate_for(r, monkeypatch)
    gate.confirm()
    trace = instrument(r)
    driver = r.p.root / "merge-probed.py"
    driver.write_text(
        f"import sys\nsys.path.insert(0, {str(OPS / 'review')!r})\n"
        + "import merge_gate\n"
        + evidence_probes()
        + "\nsys.exit(merge_gate.main())\n"
    )
    module = importlib.import_module("tests.e2e.agents_ops.test_merge_gate")
    monkeypatch.setattr(module, "SCRIPT", driver)
    result = gate.cli("merge", AGENT_AUTO_MERGE="0")
    assert result.returncode == 0, result.stderr + result.stdout
    calls = rows(str(trace))
    assert any(c["tool"] == "git" for c in calls)
    assert any(c["tool"] == "completion_evidence" for c in calls)
    assert not [c for c in calls if c["held"]]


def test_snapshot_preparation_and_instruction_release_registry(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    trace = instrument(r)
    prepared = json.loads(
        r.p.cli(
            "review-prepare",
            *r.p.review_flags(),
            "--prs",
            "100",
            "--reason-file",
            r.p.note(),
        ).stdout
    )
    r.p.cli(
        "review-request",
        "--run-id",
        prepared["review_run_id"],
        "--instruction-file",
        r.p.note(),
    )
    calls = rows(str(trace))
    assert any(c["tool"] == "git" for c in calls)
    assert not [c for c in calls if c["held"]]
