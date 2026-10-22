"""Judgments with fake paginated GitHub, the real runner and an offline Claude CLI."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs

import pytest

ROOT = Path(__file__).resolve().parents[3]
FAKE_CLI = Path(__file__).with_name("fake_judgement_cli.py")
FAKE_RUNNER = Path(__file__).with_name("fake_judgement_runner.py")


def issue(
    n: int,
    *,
    body: str = "目的: parserを改善する。対象: README。完了条件: 境界を検証する。",
    state: str = "open",
    reason: str | None = None,
    labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "number": n,
        "repository_url": "https://api.github.com/repos/o/r",
        "title": f"作業 {n}",
        "body": body,
        "state": state,
        "state_reason": reason,
        "labels": [{"name": label} for label in labels],
        "assignees": [],
    }


def decision(
    n: int,
    status: str = "startable",
    *,
    target: int | None = None,
    priority: int | None = None,
    deps: tuple[int, ...] = (),
) -> dict[str, Any]:
    ready = status == "startable"
    target = None if status == "out_of_scope" else (n if target is None else target)
    sources = [f"issue:{k}" for k in sorted({n, *deps, *([target] if target else [])})]
    if ready:
        sources.append("code:CODE_COMMIT:README.md")
    return {
        "issue": n,
        "disposition": status,
        "target_parent": target,
        "priority": (priority or n) if ready else None,
        "requirements": {
            "purpose": "parserの境界を明確にする",
            "scope": "READMEとテスト。起動・mergeは対象外",
            "behavior": "不正入力を拒否する",
            "acceptance": "CPUのfake fixtureで境界を検証する",
        }
        if ready
        else None,
        "rationale": f"#{n} の要件と資料を照合した。{status}として扱う。",
        "facts": [f"#{n} の本文を確認した。"],
        "missing_information": ["期待する入力形式を選択する必要がある"]
        if status == "information_needed"
        else [],
        "resume_condition": "阻害条件の解消を確認し、予定回または明示実行で再判断する。",
        "next_action": "判定に従い割当側へ候補または待ち理由を渡す。",
        "sources": sources,
        "dependencies": list(deps),
    }


def agreement_comment(n: int) -> dict[str, Any]:
    policy = importlib.import_module("parent.policy")
    agreement = {
        **decision(n)["requirements"],
        "human_confirmation": "人間がこの目的・範囲・完了条件に合意した。",
    }
    return {
        "id": 900 + n,
        "user": {"login": "operator"},
        "body": policy.marker(
            {
                "parent": n,
                "event": "agreed",
                "revision": 1,
                "agreement": agreement,
                "agreement_sha256": policy.digest(agreement),
            }
        ),
    }


class FakeGitHub:
    repo = "o/r"

    def __init__(self) -> None:
        self.issues = {1: issue(1)}
        self.parents: dict[int, int] = {}
        self.dependencies: dict[int, list[int]] = {}
        self.comment_rows: dict[int, list[dict[str, Any]]] = {}
        self.prs: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.writes: list[dict[str, Any]] = []
        self.fail = ""
        self.on_get: Callable[[str], None] | None = None
        self.on_write: Callable[[], None] | None = None

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        self.calls.append(endpoint)
        if self.on_get:
            self.on_get(endpoint)
        if self.fail and self.fail in endpoint:
            raise ValueError("injected API failure: " + endpoint)
        if endpoint == "user":
            return {"login": "operator"}
        path, _, query = endpoint.partition("?")
        params = parse_qs(query)
        page = int(params.get("page", ["1"])[0])

        def paged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return copy.deepcopy(rows[(page - 1) * 100 : page * 100])

        if path == "repos/o/r/issues":
            return paged(
                [v for _, v in sorted(self.issues.items()) if v["state"] == "open"]
            )
        if path == "repos/o/r/pulls":
            return paged(self.prs)
        parts = path.removeprefix("repos/o/r/issues/").split("/")
        n = int(parts[0])
        if len(parts) == 1:
            return copy.deepcopy(self.issues[n])
        tail = "/".join(parts[1:])
        if tail == "parent":
            assert missing_parent
            return (
                copy.deepcopy(self.issues[self.parents[n]])
                if n in self.parents
                else None
            )
        if tail == "sub_issues":
            return paged(
                [self.issues[k] for k, v in sorted(self.parents.items()) if v == n]
            )
        if tail == "dependencies/blocked_by":
            return paged([self.issues[k] for k in self.dependencies.get(n, [])])
        if tail == "comments":
            return paged(self.comment_rows.get(n, []))
        raise AssertionError(endpoint)

    def rows(self, endpoint: str) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            importlib.import_module("parent.github").GitHub.rows(self, endpoint),
        )

    def identity(self, raw: object) -> int:
        return cast(
            int, importlib.import_module("parent.github").GitHub.identity(self, raw)
        )

    def issue(self, n: int) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            importlib.import_module("parent.github").GitHub.issue(self, n),
        )

    def comments(self, n: int) -> list[dict[str, Any]]:
        return self.rows(f"repos/o/r/issues/{n}/comments")

    def pulls(self) -> list[dict[str, Any]]:
        return self.rows("repos/o/r/pulls?state=all")

    def write(
        self, method: str, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if self.on_write:
            self.on_write()
        self.writes.append({"method": method, "endpoint": endpoint, "payload": payload})
        if method == "PATCH":
            cid = int(endpoint.rsplit("/", 1)[1])
            row = next(
                c for rows in self.comment_rows.values() for c in rows if c["id"] == cid
            )
            row["body"] = payload["body"]
            return copy.deepcopy(row)
        assert method == "POST" and endpoint.endswith("/comments")
        n = int(endpoint.split("/")[-2])
        cid = (
            max(
                (c["id"] for rows in self.comment_rows.values() for c in rows),
                default=0,
            )
            + 1
        )
        row = {"id": cid, "user": {"login": "operator"}, "body": payload["body"]}
        self.comment_rows.setdefault(n, []).append(row)
        return copy.deepcopy(row)


class Harness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
        self.mod = importlib.import_module("dispatcher.judgement")
        self.collector = importlib.import_module("dispatcher.collect")
        self.schema = importlib.import_module("dispatcher.schema")
        self.root, self.work = root, root / "work"
        self.work.mkdir()
        self.git("init", "-q", "-b", "test")
        (self.work / "README.md").write_text("# Parser\n\nStrict inputs.\n")
        self.git("add", "README.md")
        self.git(
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        )
        self.store = importlib.import_module("parent.state").Store(
            root / "private-state"
        )
        self.gh = FakeGitHub()
        self.output = root / "response.json"
        self.calls = root / "agent-calls.jsonl"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "claude"
        fake.write_bytes(FAKE_CLI.read_bytes())
        fake.chmod(0o700)
        self.fake_runner = bin_dir / "fake-judgement-runner"
        self.fake_runner.write_bytes(FAKE_RUNNER.read_bytes())
        self.fake_runner.chmod(0o700)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
        monkeypatch.setenv("FAKE_JUDGEMENT_RESPONSE", str(self.output))
        monkeypatch.setenv("FAKE_JUDGEMENT_CALLS", str(self.calls))
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
        self.count = 0

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=self.work,
            env={
                **os.environ,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            },
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def response(self, *rows: dict[str, Any]) -> None:
        self.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_sha256": "INPUT_DIGEST",
                    "decisions": list(rows),
                },
                ensure_ascii=False,
            )
        )

    def run(self, *, apply: bool = True, runner: Path | None = None) -> dict[str, Any]:
        self.count += 1
        self.run_dir = self.root / f"run-{self.count}"
        return cast(
            dict[str, Any],
            self.mod.run(
                self.gh,
                self.store,
                self.work,
                self.run_dir,
                apply=apply,
                timeout=15,
                runner=runner or self.mod.RUNNER,
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        return cast(
            dict[str, Any], json.loads((self.run_dir / "input.json").read_text())
        )


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_unagreed_standalone_and_existing_family_route_without_new_hierarchy(
    h: Harness,
) -> None:
    h.gh.issues.update(
        {
            2: issue(2, labels=("ai-proposed", "ready")),
            3: issue(3),
            4: issue(4, labels=("issue:parent",)),
            5: issue(5),
            6: issue(6, state="closed", reason="not_planned"),
            7: issue(7),
            8: issue(8),
            9: issue(9, labels=("needs-human",)),
            10: issue(10, labels=("ops-report",)),
        }
    )
    h.gh.parents[3] = 4
    h.gh.dependencies[5] = [6]
    h.gh.prs = [
        {
            "number": 100,
            "body": "Closes #7",
            "state": "closed",
            "merged_at": None,
            "head": {"ref": "codex/unfinished", "sha": "a" * 40},
        }
    ]
    h.gh.issues[8]["assignees"] = [{"login": "human"}]
    h.response(
        decision(1),
        decision(2, "information_needed"),
        decision(3, "existing_parent", target=4),
        decision(4),
        decision(5, "dependency_wait", deps=(6,)),
        decision(7, "existing_owner"),
        decision(8, "existing_owner"),
        decision(9, "human_hold"),
        decision(10, "out_of_scope"),
    )
    result = h.run()
    assert result["result"] == "recorded", result
    assert [(c["parent"], c["route"]) for c in result["candidates"]] == [
        (1, "standalone"),
        (4, "parent"),
    ]
    assert all(
        c["generation"] == 1 and c["requirements"]["acceptance"]
        for c in result["candidates"]
    )
    assert {d["issue"]: d["target_parent"] for d in result["decisions"]}[3] == 4
    assert not (h.store.directory / "registry.json").exists()
    assert all("/comments" in w["endpoint"] for w in h.gh.writes)
    assert 10 not in h.gh.comment_rows
    published = json.dumps(h.gh.comment_rows, ensure_ascii=False)
    assert '"event": "agreed"' not in published
    assert "human_confirmation" not in published and "allow_shared" not in published
    assert "private-judgement-session" not in published
    invocation = json.loads((h.run_dir / "agent-runtime/invocation.json").read_text())
    calls = [json.loads(line) for line in h.calls.read_text().splitlines()]
    args = calls[0]["args"]
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--tools") + 1 : args.index("--permission-mode")] == [
        "Read",
        "Grep",
        "Glob",
    ]
    assert "dontAsk" in args and invocation
    assert args[args.index("--add-dir") + 1] == str(h.run_dir / "agent-input")
    output_schema = ROOT / ".agents/ops/dispatcher/judgement.schema.json"
    assert json.loads(args[args.index("--json-schema") + 1]) == json.loads(
        output_schema.read_text()
    )
    contract = json.loads((h.run_dir / "agent-runtime/result.json").read_text())[
        "contract"
    ]
    assert contract["read_dirs"] == [str(h.run_dir / "agent-input")]
    assert (
        contract["json_schema_sha256"]
        == hashlib.sha256(output_schema.read_bytes()).hexdigest()
    )
    assert json.loads((h.run_dir / "agent-runtime/final.md").read_text())["decisions"]
    assert "I'm writing" in (h.run_dir / "agent-runtime/events.jsonl").read_text()
    assert (h.run_dir / "result.json").stat().st_mode & 0o077 == 0


def test_every_page_and_cross_parent_dependency_are_collected(h: Harness) -> None:
    h.gh.issues = {n: issue(n) for n in range(1, 107)}
    h.gh.parents = {n: 1 for n in range(2, 107)}
    h.gh.issues.update(
        {n: issue(n, state="closed", reason="completed") for n in range(200, 305)}
    )
    h.gh.issues[400] = issue(400, state="closed", reason="completed")
    h.gh.parents[200] = 400
    h.gh.dependencies[1] = list(range(200, 305))
    h.gh.comment_rows[1] = [
        {"id": n, "user": {"login": "human"}, "body": f"指示 {n}"}
        for n in range(1, 107)
    ]
    h.gh.prs = [
        {
            "number": n,
            "body": "",
            "state": "closed",
            "merged_at": None,
            "head": {"ref": f"branch-{n}", "sha": "a" * 40},
        }
        for n in range(500, 606)
    ]
    with h.store.locked():
        snapshot = h.collector.collect(h.gh, h.store, h.work)
    assert len(snapshot["subjects"]) == 106
    assert len(snapshot["issues"]["1"]["children"]) == 105
    assert len(snapshot["issues"]["1"]["dependencies"]) == 105
    assert len(snapshot["issues"]["1"]["comments"]) == 106
    assert len(snapshot["pulls"]) == 106
    assert snapshot["issues"]["200"]["parent"] == 400
    for path in (
        "issues?state=open&sort=created&direction=asc",
        "issues/1/sub_issues",
        "issues/1/dependencies/blocked_by",
        "issues/1/comments",
        "pulls?state=all",
    ):
        assert any(path in c and "page=2" in c for c in h.gh.calls)
    snapshot["input_sha256"] = importlib.import_module("parent.policy").digest(
        h.collector.material(snapshot)
    )
    index_path = importlib.import_module("dispatcher.agent_input").write_agent_input(
        snapshot, h.root / "many-subjects-input"
    )
    index = json.loads(index_path.read_text())
    assert index["subjects"] == snapshot["subjects"]
    assert set(index["issues"]) == set(snapshot["issues"])
    assert index["issues"]["200"]["parent"] == 400
    assert (
        index["issues"]["1"]["dependencies"] == snapshot["issues"]["1"]["dependencies"]
    )
    assert index["pulls"] == {}
    assert len(index_path.read_text().splitlines()) < 1000


def read_agent_text(directory: Path, value: dict[str, Any]) -> str:
    if "segments" in value:
        return "".join(value["segments"])
    return "".join(
        read_agent_text(directory, json.loads((directory / f).read_text()))
        for f in value["files"]
    )


def test_large_agent_input_is_lossless_filters_prs_and_keeps_original_digest(
    h: Harness,
) -> None:
    body = "<!-- ops-report-proposal: R-01 report=#9 option=plan -->\n\n"
    body += ('長文の要件\\"\r\n' + "x" * 5000) * 50 + "\n末尾の重要な停止指示"
    h.gh.issues[1]["body"] = body
    h.gh.issues.update(
        {
            2: issue(2, state="closed", reason="completed"),
            9: issue(9, state="closed", reason="completed", labels=("ops-report",)),
            200: issue(200, state="closed", reason="completed"),
        }
    )
    h.gh.parents[2] = 1
    h.gh.dependencies[1] = [200]
    h.gh.comment_rows[1] = [
        {"id": n, "user": {"login": "human"}, "body": f"指示 {n}: " + "継続する。" * 80}
        for n in range(1, 107)
    ]
    h.gh.comment_rows[1][-1]["body"] = "巨大コメント" * 40000 + "\n最後の取消指示"
    h.gh.prs = [
        {
            "number": n,
            "body": "Refs #1" if n == 500 else f"Closes #{n + 1000}",
            "state": "closed",
            "merged_at": None,
            "head": {"ref": f"branch-{n}", "sha": "a" * 40},
        }
        for n in range(500, 651)
    ]
    h.gh.prs[1]["body"] = "Closes #2\n" + "PR詳細。" * 10000
    h.gh.prs[2]["body"] = "Fixes #200"
    h.gh.prs[2]["merged_at"] = "2026-10-22T00:00:00Z"
    h.gh.prs[3].update(body="Resolves #9", state="open")
    with h.store.locked():
        before = h.collector.collect(h.gh, h.store, h.work)
    digest = importlib.import_module("parent.policy").digest
    expected_digest = digest(h.collector.material(before))
    held = decision(1, "information_needed", deps=(200,))
    held["sources"].append("issue:9")
    h.response(held)
    result = h.run(apply=False, runner=h.fake_runner)
    assert result["result"] == "preview", result
    snapshot = h.snapshot()
    assert result["input_sha256"] == expected_digest == snapshot["input_sha256"]
    assert h.collector.material(snapshot) == h.collector.material(before)
    assert len(snapshot["pulls"]) == 151 and snapshot["code"]["files"]
    directory = h.run_dir / "agent-input"
    index = json.loads((directory / "index.json").read_text())
    assert index["subjects"] == [1]
    assert set(index["issues"]) == {"1", "2", "9", "200"}
    assert set(index["pulls"]) == {"501", "502", "503"}
    assert index["input_sha256"] == expected_digest
    assert index["code_commit"] == snapshot["code"]["commit"] and "code" not in index
    assert index["issue_content_complete"] is True
    assert {o["kind"] for o in index["omissions"]} == {
        "unrelated_pull_requests",
        "code_tree",
        "execution_inputs",
    }
    omitted_prs = next(
        o for o in index["omissions"] if o["kind"] == "unrelated_pull_requests"
    )
    assert omitted_prs["count"] == 148 and all(o["reason"] for o in index["omissions"])
    for n, raw in snapshot["issues"].items():
        detail = json.loads((directory / index["issues"][n]["file"]).read_text())
        assert detail["content_complete"] is True and detail["omissions"] == []
        assert read_agent_text(directory, detail["body"]) == raw["body"]
        comments = [
            {**c, "body": read_agent_text(directory, c["body"])}
            for f in detail["comment_files"]
            for c in json.loads((directory / f).read_text())
        ]
        assert comments == raw["comments"]
        assert all(
            detail[k] == v for k, v in raw.items() if k not in {"body", "comments"}
        )
        if n == "1":
            assert len(detail["body"]["files"]) > 1
            assert len(detail["comment_files"]) > 1
    for p in snapshot["pulls"]:
        if str(p["number"]) not in index["pulls"]:
            continue
        detail = json.loads(
            (directory / index["pulls"][str(p["number"])]["file"]).read_text()
        )
        assert {**detail, "body": read_agent_text(directory, detail["body"])} == p
    for path in directory.rglob("*"):
        assert path.stat().st_mode & 0o077 == 0
        if path.is_file():
            assert path.stat().st_size < 45000
            assert max(map(len, path.read_text().splitlines())) < 2000
    preserved = copy.deepcopy(snapshot)
    importlib.import_module("dispatcher.agent_input").write_agent_input(
        snapshot, h.root / "second-agent-input"
    )
    assert (
        snapshot == preserved
        and digest(h.collector.material(snapshot)) == expected_digest
    )
    assert h.gh.writes == []


def test_runner_receives_only_the_agent_input_directory(h: Harness) -> None:
    with h.store.locked():
        h.store.parents["o/r#200"] = {
            "session": "claude:private-owner",
            "worktree": "/private/worktree",
            "status": "active",
            "generation": 2,
            "tasks": {"1": {"delegated": True}},
            "reviews": [],
        }
        h.store.save()
    h.response(decision(1, "existing_owner"))
    result = h.run(apply=False, runner=h.fake_runner)
    assert result["result"] == "preview", result
    call = json.loads(h.calls.read_text())
    assert call["args"][call["args"].index("--read-dir") + 1] == str(
        h.run_dir / "agent-input"
    )
    assert str(h.run_dir / "agent-input/index.json") in call["prompt"]
    assert str(h.run_dir / "input.json") not in call["prompt"]
    index_text = (h.run_dir / "agent-input/index.json").read_text()
    owner = json.loads(index_text)["owners"]["200"]
    assert owner["status"] == "active" and owner["generation"] == 2
    assert owner["tasks"]["1"]["delegated"] is True
    assert "private-owner" not in index_text and "/private/worktree" not in index_text
    assert h.gh.writes == []


@pytest.mark.parametrize("scenario", ["missing", "wrong"])
def test_read_directory_contract_mismatch_never_publishes(
    h: Harness, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setenv("FAKE_JUDGEMENT_READ_DIRS", scenario)
    h.response(decision(1))
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "failed" and result["candidates"] == []
    assert any("contract" in diagnostic for diagnostic in result["diagnostics"])
    assert h.gh.writes == []


@pytest.mark.parametrize("scenario", ["missing", "wrong"])
def test_output_schema_contract_mismatch_never_publishes(
    h: Harness, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setenv("FAKE_JUDGEMENT_SCHEMA", scenario)
    h.response(decision(1))
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "failed" and result["candidates"] == []
    assert any("contract" in diagnostic for diagnostic in result["diagnostics"])
    assert h.gh.writes == []


def test_missing_structured_output_is_not_recovered_from_prose(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_JUDGEMENT_SCENARIO", "missing_structured_output")
    h.response(decision(1))
    result = h.run()
    evidence = json.loads((h.run_dir / "agent-runtime/result.json").read_text())
    assert evidence["turn_end_reason"] == "invalid_output"
    assert "missing structured_output" in evidence["error"]
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == []


def test_preamble_without_structured_output_still_fails_strict_json(h: Harness) -> None:
    h.response(decision(1))
    h.output.write_text("I'm writing the final JSON now.\n\n" + h.output.read_text())
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "failed" and result["candidates"] == []
    assert any("JSONDecodeError" in message for message in result["diagnostics"])
    assert h.gh.writes == []


def test_output_schema_keeps_decision_fields_and_classifications(h: Harness) -> None:
    schema = json.loads(
        (ROOT / ".agents/ops/dispatcher/judgement.schema.json").read_text()
    )
    decision_schema = schema["definitions"]["decision"]
    assert set(decision_schema["required"]) == h.schema.DECISION_KEYS
    assert set(decision_schema["properties"]) == h.schema.DECISION_KEYS
    assert (
        set(decision_schema["properties"]["disposition"]["enum"])
        == h.schema.DISPOSITIONS
    )
    assert (
        set(schema["definitions"]["requirements"]["required"]) == h.schema.REQUIREMENTS
    )


def test_output_schema_uses_cli_compatible_dialect_and_explicit_array_types() -> None:
    from jsonschema import Draft7Validator  # type: ignore[import-untyped]

    schema = json.loads(
        (ROOT / ".agents/ops/dispatcher/judgement.schema.json").read_text()
    )
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    Draft7Validator.check_schema(schema)

    def check(value: Any) -> None:
        if isinstance(value, dict):
            if {
                "items",
                "minItems",
                "maxItems",
                "contains",
                "uniqueItems",
            } & value.keys():
                assert value.get("type") == "array", value
            if "$ref" in value:
                # Draft-07 ignores validation keywords beside $ref.
                assert set(value) == {"$ref"}, value
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)


@pytest.mark.parametrize(
    ("status", "field", "value"),
    [
        ("information_needed", "facts", []),
        ("information_needed", "sources", []),
        ("information_needed", "missing_information", []),
        ("startable", "missing_information", ["未決事項"]),
        ("startable", "sources", ["issue:1"]),
        ("startable", "facts", [123]),
        ("startable", "facts", ["同じ事実", "同じ事実"]),
    ],
)
def test_output_schema_preserves_array_and_conditional_constraints(
    h: Harness, status: str, field: str, value: Any
) -> None:
    row = decision(1, status)
    row[field] = value
    h.response(row)
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    evidence = json.loads((h.run_dir / "agent-runtime/result.json").read_text())
    assert evidence["turn_end_reason"] == "invalid_output"
    assert "does not match --json-schema" in evidence["error"]
    assert h.gh.writes == []


def test_schema_preflight_refusal_is_diagnosed_without_starting_cli(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = importlib.import_module("dispatcher.agent")
    incompatible = h.root / "incompatible.schema.json"
    schema = json.loads(agent.OUTPUT_SCHEMA.read_text())
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    incompatible.write_text(json.dumps(schema))
    monkeypatch.setattr(agent, "OUTPUT_SCHEMA", incompatible)
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert "failed with exit 2" in result["diagnostics"][0]
    assert "Claude CLI (Ajv default)" in result["diagnostics"][0]
    assert not h.calls.exists() and not (h.run_dir / "agent-runtime").exists()
    assert h.gh.writes == []


def test_cli_schema_refusal_before_first_turn_has_diagnostic_stderr(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_JUDGEMENT_SCENARIO", "schema_cli_error")
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    evidence = json.loads((h.run_dir / "agent-runtime/result.json").read_text())
    assert evidence["turn_end_reason"] == "cli_error"
    assert evidence["runtime"]["session_id"] is None
    assert "turn_end_reason=cli_error" in result["diagnostics"][0]
    assert (
        'no schema with key or ref "https://json-schema.org/draft/2020-12/schema"'
        in result["diagnostics"][0]
    )
    assert h.gh.writes == []


@pytest.mark.parametrize(
    "change",
    [
        "json",
        "duplicate_keys",
        "unknown_version",
        "unknown_id",
        "missing_issue",
        "extra_key",
        "invented_source",
        "missing_code",
        "human_confirmation",
        "wrong_digest",
        "bool_priority",
    ],
)
def test_invalid_agent_output_never_publishes_or_emits_candidates(
    h: Harness, change: str
) -> None:
    h.response(decision(1))
    value = json.loads(h.output.read_text())
    if change == "json":
        h.output.write_text("not JSON")
    elif change == "duplicate_keys":
        h.output.write_text('{"schema_version":1,"schema_version":1}')
    else:
        d = value["decisions"][0]
        if change == "unknown_version":
            value["schema_version"] = 99
        elif change == "unknown_id":
            d["issue"] = 999
        elif change == "missing_issue":
            value["decisions"] = []
        elif change == "extra_key":
            d["launch"] = True
        elif change == "invented_source":
            d["sources"].append("issue:999")
        elif change == "missing_code":
            d["sources"] = ["issue:1"]
        elif change == "human_confirmation":
            d["requirements"]["human_confirmation"] = "fabricated"
        elif change == "wrong_digest":
            value["input_sha256"] = "a" * 64
        else:
            d["priority"] = True
        h.output.write_text(json.dumps(value))
    result = h.run()
    assert (
        result["result"] == "failed"
        and result["candidates"] == []
        and result["diagnostics"]
    )
    assert h.gh.writes == []


@pytest.mark.parametrize(
    "path",
    [
        "issues?state=open&sort=created&direction=asc&per_page=100&page=2",
        "issues/1/comments",
        "issues/1/parent",
        "issues/1/dependencies/blocked_by",
        "pulls?state=all",
    ],
)
def test_api_failure_never_uses_partial_inputs(h: Harness, path: str) -> None:
    h.gh.issues.update({n: issue(n) for n in range(2, 102)})
    h.gh.fail = path
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert result["diagnostics"] and h.gh.writes == []
    assert not h.calls.exists()


@pytest.mark.parametrize("scenario", ["wrong_model", "cli_error", "empty"])
def test_runtime_failure_is_not_task_success(
    h: Harness, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setenv("FAKE_JUDGEMENT_SCENARIO", scenario)
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == []


def test_same_inputs_still_run_agent_and_update_only_changed_comment(
    h: Harness,
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2, "information_needed"))
    first = h.run()
    assert first["result"] == "recorded", first
    writes = len(h.gh.writes)
    second = h.run()
    assert second["result"] == "recorded", second
    assert first["input_sha256"] == second["input_sha256"]
    assert first["candidates"] == second["candidates"]
    assert len(h.gh.writes) == writes
    assert len(h.calls.read_text().splitlines()) == 2
    changed = decision(2, "information_needed")
    changed["rationale"] = "新しい未決事項の理由を具体化した。"
    h.response(decision(1), changed)
    third = h.run()
    assert third["result"] == "recorded", third
    assert third["input_sha256"] == first["input_sha256"]
    assert len(h.gh.writes) == writes + 1
    assert h.gh.writes[-1]["method"] == "PATCH"
    assert len(h.gh.comment_rows[2]) == 1


@pytest.mark.parametrize(
    "change", ["requirements", "human_comment", "dependency", "new_pr", "owner"]
)
def test_changes_immediately_before_publication_defer_the_subject(
    h: Harness, change: str
) -> None:
    h.gh.issues[200] = issue(200, state="closed", reason="completed")
    h.gh.dependencies[1] = [200]
    h.response(decision(1, deps=(200,)))

    def concurrent_edit(endpoint: str) -> None:
        if not h.calls.exists():
            return
        h.gh.on_get = None
        if change == "requirements":
            h.gh.issues[1]["body"] += "\n新しい要件"
        elif change == "human_comment":
            h.gh.comment_rows[1] = [
                {
                    "id": 999,
                    "user": {"login": "human"},
                    "body": "実装を停止してください。",
                }
            ]
        elif change == "dependency":
            h.gh.issues[200]["state_reason"] = "not_planned"
        elif change == "new_pr":
            h.gh.prs.append(
                {
                    "number": 300,
                    "body": "Closes #1",
                    "state": "open",
                    "merged_at": None,
                    "head": {"ref": "worker", "sha": "a" * 40},
                }
            )
        else:
            h.store.parents["o/r#1"] = {
                "status": "active",
                "generation": 1,
                "session": "claude:private-owner",
                "tasks": {},
                "reviews": [],
            }
            h.store.save()

    h.gh.on_get = concurrent_edit
    result = h.run()
    assert result["result"] == "deferred" and result["candidates"] == []
    assert result["decisions"] == []
    assert [(d["issue"], d["stage"]) for d in result["deferred"]] == [(1, "preflight")]
    assert "次の予定回または明示実行で再判断" in result["diagnostics"][0]
    assert h.gh.writes == []


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize(
    "change", ["requirements", "closed", "owner", "delegated", "assignee", "new_pr"]
)
def test_other_subject_changes_do_not_discard_fresh_judgements(
    h: Harness, apply: bool, change: str
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2))

    def concurrent_edit(endpoint: str) -> None:
        if not h.calls.exists():
            return
        h.gh.on_get = None
        if change == "requirements":
            h.gh.issues[2]["body"] += "\n新しい要件"
        elif change == "closed":
            h.gh.issues[2].update(state="closed", state_reason="completed")
        elif change in {"owner", "delegated"}:
            h.store.parents[f"o/r#{2 if change == 'owner' else 300}"] = {
                "status": "active",
                "generation": 1,
                "session": "claude:private-owner",
                "tasks": {} if change == "owner" else {"2": {"delegated": True}},
                "reviews": [],
            }
            h.store.save()
        elif change == "assignee":
            h.gh.issues[2]["assignees"] = [{"login": "human"}]
        else:
            h.gh.prs.append(
                {
                    "number": 300,
                    "body": "Closes #2",
                    "state": "open",
                    "merged_at": None,
                    "head": {"ref": "worker", "sha": "a" * 40},
                }
            )

    h.gh.on_get = concurrent_edit
    result = h.run(apply=apply, runner=h.fake_runner)
    assert result["schema_version"] == 2
    assert result["result"] == ("recorded" if apply else "preview"), result
    assert [d["issue"] for d in result["decisions"]] == [1]
    assert [d["issue"] for d in result["deferred"]] == [2]
    assert [c["parent"] for c in result["candidates"]] == ([1] if apply else [])
    assert [p["issue"] for p in result["publications"]] == ([1] if apply else [])
    assert all(w["endpoint"] == "repos/o/r/issues/1/comments" for w in h.gh.writes)
    assert json.loads((h.run_dir / "result.json").read_text()) == result
    assert len(h.calls.read_text().splitlines()) == 1


@pytest.mark.parametrize("change", ["dependency", "formal_child", "source_report"])
def test_related_input_changes_defer_only_subjects_in_that_closure(
    h: Harness, change: str
) -> None:
    h.gh.issues.update(
        {2: issue(2), 200: issue(200, state="closed", reason="completed")}
    )
    d = decision(2)
    if change == "dependency":
        h.gh.dependencies[2] = [200]
        d = decision(2, deps=(200,))
    elif change == "formal_child":
        h.gh.parents[200] = 2
    else:
        h.gh.issues[2]["body"] = (
            "<!-- ops-report-proposal: R-01 report=#200 option=plan -->\n\n"
            + h.gh.issues[2]["body"]
        )
        h.gh.issues[200]["labels"] = [{"name": "ops-report"}]
        d["sources"].append("issue:200")
    h.response(decision(1), d)

    def concurrent_edit(endpoint: str) -> None:
        if h.calls.exists():
            h.gh.on_get = None
            h.gh.issues[200]["body"] += "\n対象となる判断材料の変更"

    h.gh.on_get = concurrent_edit
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    assert [d["issue"] for d in result["decisions"]] == [1]
    assert [d["issue"] for d in result["deferred"]] == [2]
    assert [c["parent"] for c in result["candidates"]] == [1]
    assert 2 not in h.gh.comment_rows


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("change", ["issues", "code"])
def test_all_subjects_changed_is_deferred_without_automatic_rejudgement(
    h: Harness, apply: bool, change: str
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2))

    def concurrent_edit(endpoint: str) -> None:
        if not h.calls.exists():
            return
        h.gh.on_get = None
        if change == "issues":
            for raw in h.gh.issues.values():
                raw["body"] += "\n新しい要件"
        else:
            (h.work / "README.md").write_text("# Parser\n\nNew requirements.\n")
            h.git("add", "README.md")
            h.git(
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "new code",
            )

    h.gh.on_get = concurrent_edit
    result = h.run(apply=apply, runner=h.fake_runner)
    assert result["result"] == "deferred", result
    assert result["candidates"] == result["decisions"] == result["publications"] == []
    assert [d["issue"] for d in result["deferred"]] == [1, 2]
    assert len(result["diagnostics"]) == 2
    assert h.gh.writes == [] and len(h.calls.read_text().splitlines()) == 1


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize(
    "change", ["new_issue", "unrelated_issue", "unrelated_pr", "unrelated_owner"]
)
def test_unrelated_updates_and_new_open_issues_do_not_invalidate_subjects(
    h: Harness, apply: bool, change: str
) -> None:
    h.gh.issues[200] = issue(200, state="closed", reason="completed")
    h.response(decision(1))

    def concurrent_edit(endpoint: str) -> None:
        if not h.calls.exists():
            return
        h.gh.on_get = None
        if change == "new_issue":
            h.gh.issues[2] = issue(2)
        elif change == "unrelated_issue":
            h.gh.issues[200]["body"] += "\n無関係な変更"
        elif change == "unrelated_pr":
            h.gh.prs.append(
                {
                    "number": 300,
                    "body": "Closes #200",
                    "state": "open",
                    "merged_at": None,
                    "head": {"ref": "unrelated-worker", "sha": "a" * 40},
                }
            )
        else:
            h.store.parents["o/r#200"] = {
                "status": "active",
                "generation": 1,
                "session": "claude:private-owner",
                "tasks": {},
                "reviews": [],
            }
            h.store.save()

    h.gh.on_get = concurrent_edit
    result = h.run(apply=apply, runner=h.fake_runner)
    assert result["result"] == ("recorded" if apply else "preview"), result
    assert [d["issue"] for d in result["decisions"]] == [1]
    assert result["deferred"] == [] and result["diagnostics"] == []
    assert h.snapshot()["subjects"] == [1]
    assert [c["parent"] for c in result["candidates"]] == ([1] if apply else [])
    if apply:
        with h.store.locked():
            h.mod.assert_candidate_fresh(
                h.gh, h.store, h.work, h.snapshot(), result["candidates"][0]
            )
    else:
        assert h.gh.writes == []


def test_new_open_child_defers_only_the_existing_family(h: Harness) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2))

    def concurrent_edit(endpoint: str) -> None:
        if h.calls.exists():
            h.gh.on_get = None
            h.gh.issues[3] = issue(3)
            h.gh.parents[3] = 2

    h.gh.on_get = concurrent_edit
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    assert [c["parent"] for c in result["candidates"]] == [1]
    assert [d["issue"] for d in result["deferred"]] == [2]
    assert h.snapshot()["subjects"] == [1, 2]
    assert 2 not in h.gh.comment_rows and 3 not in h.gh.comment_rows


def test_change_during_comment_write_cannot_record_a_basis(h: Harness) -> None:
    h.gh.issues[200] = issue(200, state="closed", reason="completed")
    h.gh.dependencies[1] = [200]
    h.response(decision(1, deps=(200,)))

    def edit() -> None:
        h.gh.on_write = None
        h.gh.issues[200]["state_reason"] = "not_planned"

    h.gh.on_write = edit
    result = h.run()
    assert result["result"] == "deferred" and result["candidates"] == []
    assert result["decisions"] == []
    assert result["deferred"][0]["stage"] == "publication"
    assert len(h.gh.writes) == 1
    assert not any(
        '"event": "execution_basis"' in c["body"] for c in h.gh.comment_rows[1]
    )


@pytest.mark.parametrize("write_number", [1, 2])
@pytest.mark.parametrize("changed_subject", [1, 2])
def test_subject_changes_during_writes_preserve_other_candidates(
    h: Harness, write_number: int, changed_subject: int
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2))

    def edit() -> None:
        if len(h.gh.writes) + 1 == write_number:
            h.gh.on_write = None
            h.gh.issues[changed_subject]["body"] += "\n新しい要件"

    h.gh.on_write = edit
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    fresh_subject = 3 - changed_subject
    assert [d["issue"] for d in result["decisions"]] == [fresh_subject]
    assert [c["parent"] for c in result["candidates"]] == [fresh_subject]
    assert [(d["issue"], d["stage"]) for d in result["deferred"]] == [
        (changed_subject, "publication")
    ]
    if changed_subject == 2:
        assert 2 not in h.gh.comment_rows
    else:
        bases = [
            c for c in h.gh.comment_rows[1] if '"event": "execution_basis"' in c["body"]
        ]
        assert len(bases) == write_number - 1


def test_dependency_change_during_basis_write_is_caught_after_publication(
    h: Harness,
) -> None:
    h.gh.issues.update(
        {2: issue(2), 200: issue(200, state="closed", reason="completed")}
    )
    h.gh.dependencies[1] = [200]
    h.response(decision(1, deps=(200,)), decision(2))

    def edit() -> None:
        if len(h.gh.writes) == 1:
            h.gh.on_write = None
            h.gh.issues[200]["state_reason"] = "not_planned"

    h.gh.on_write = edit
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    assert [c["parent"] for c in result["candidates"]] == [2]
    assert [d["issue"] for d in result["deferred"]] == [1]
    assert len(h.gh.comment_rows[1]) == 2


def test_final_subject_checks_remove_only_candidates_changed_by_later_writes(
    h: Harness,
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2))

    def edit() -> None:
        if len(h.gh.writes) == 2:
            h.gh.on_write = None
            h.gh.issues[1]["body"] += "\n記録後の変更"

    h.gh.on_write = edit
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    assert [d["issue"] for d in result["decisions"]] == [2]
    assert [c["parent"] for c in result["candidates"]] == [2]
    assert [(d["issue"], d["stage"]) for d in result["deferred"]] == [
        (1, "final_check")
    ]
    assert [p["issue"] for p in result["publications"]] == [1, 2]
    assert len(h.gh.writes) == 4


def test_collection_failure_after_judgement_is_fatal_and_not_a_deferral(
    h: Harness,
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2))

    def fail(endpoint: str) -> None:
        if h.calls.exists():
            h.gh.on_get = None
            h.gh.fail = "issues/2/comments"

    h.gh.on_get = fail
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "failed" and result["candidates"] == []
    assert result["deferred"] == []
    assert "injected API failure" in result["diagnostics"][0]
    assert h.gh.writes == []


def test_parent_record_error_without_input_changes_is_fatal(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    h.response(decision(1))

    def fail(*args: Any) -> None:
        raise ValueError("injected basis record failure")

    monkeypatch.setattr(h.mod, "record_autonomous", fail)
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "failed" and result["candidates"] == []
    assert result["deferred"] == []
    assert "injected basis record failure" in result["diagnostics"][0]


@pytest.mark.parametrize("agent_status", ["startable", "information_needed"])
@pytest.mark.parametrize(
    ("blocker", "disposition", "fact"),
    [
        ("cancelled_dependency", "dependency_wait", "not_planned"),
        ("closed_unmerged_pr", "existing_owner", "PR #100"),
        ("needs_human", "human_hold", "needs-human"),
        ("checking", "human_hold", "phaseはchecking"),
        ("revoked", "human_hold", "phaseはchecking"),
        ("agreed", "human_hold", "phaseはagreed"),
        ("stopped", "human_hold", "停止記録"),
        ("existing_owner", "existing_owner", "担当状態はactive"),
        ("assignee", "existing_owner", "GitHub assignee"),
        ("delegated", "existing_owner", "#1 の委任予約"),
        ("missing_parent", "human_hold", "正式な親関係がない"),
        ("child", "existing_parent", "正式な親は #2"),
        ("report", "out_of_scope", "報告容器"),
    ],
)
def test_mechanical_guard_is_exposed_and_enforced(
    h: Harness, blocker: str, disposition: str, fact: str, agent_status: str
) -> None:
    d = decision(1, agent_status)
    policy = importlib.import_module("parent.policy")
    if blocker == "cancelled_dependency":
        h.gh.issues[200] = issue(200, state="closed", reason="not_planned")
        h.gh.dependencies[1] = [200]
        d = decision(1, agent_status, deps=(200,))
    elif blocker == "closed_unmerged_pr":
        h.gh.prs = [
            {
                "number": 100,
                "body": "Closes #1",
                "state": "closed",
                "merged_at": None,
                "head": {"ref": "worker", "sha": "a" * 40},
            }
        ]
    elif blocker == "needs_human":
        h.gh.issues[1]["labels"] = [{"name": "needs-human"}]
    elif blocker == "agreed":
        h.gh.comment_rows[1] = [agreement_comment(1)]
    elif blocker in {"checking", "revoked"}:
        event = (
            {"parent": 1, "event": "checking"}
            if blocker == "checking"
            else {
                "parent": 1,
                "event": "basis_revoked",
                "revision": 1,
                "generation": 1,
                "reason": "人間の停止指示",
            }
        )
        h.gh.comment_rows[1] = [
            {"id": 9, "user": {"login": "operator"}, "body": policy.marker(event)}
        ]
    elif blocker == "stopped":
        h.gh.issues[1]["body"] += (
            "\n\n## 作業状況・停止事項\n\n停止\n<!-- parent-task-status:v1 state=stopped detail="
            + "a" * 64
            + " -->"
        )
    elif blocker in {"existing_owner", "delegated"}:
        with h.store.locked():
            h.store.parents["o/r#1"] = {
                "session": "claude:private-owner",
                "status": "active" if blocker == "existing_owner" else "ended",
                "generation": 1,
                "tasks": {}
                if blocker == "existing_owner"
                else {"1": {"delegated": True}},
                "reviews": [],
            }
            h.store.save()
    elif blocker == "assignee":
        h.gh.issues[1]["assignees"] = [{"login": "human"}]
    elif blocker == "missing_parent":
        h.gh.issues[1]["labels"] = [{"name": "issue:child"}]
    elif blocker == "report":
        h.gh.issues[1]["labels"] = [{"name": "ops-report"}]
        d["target_parent"] = None
    else:
        h.gh.issues[2] = issue(2, state="closed", reason="completed")
        h.gh.parents[1] = 2
        d["target_parent"] = 2
        d["sources"].append("issue:2")
    h.response(d)
    result = h.run()
    assert result["candidates"] == []
    if agent_status == "startable":
        assert result["result"] == "failed" and h.gh.writes == []
        assert any(
            f"contradicts mechanical guard {disposition}" in item
            for item in result["diagnostics"]
        )
    else:
        assert result["result"] == "recorded", result
        assert result["decisions"][0]["disposition"] == disposition
        assert len(result["diagnostics"]) == 1
        assert "agent_disposition=information_needed" in result["diagnostics"][0]
    index = json.loads((h.run_dir / "agent-input/index.json").read_text())
    required = index["issues"]["1"]["mechanical_guard"]
    assert required == h.schema.guard(h.snapshot(), 1)
    assert required["disposition"] == disposition
    assert any(fact in item for item in required["facts"])


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize(
    "status",
    [
        "startable",
        "information_needed",
        "dependency_wait",
        "existing_owner",
        "existing_parent",
        "out_of_scope",
        "human_hold",
    ],
)
def test_classification_disagreement_only_rejects_startable(
    h: Harness, blocked: bool, status: str
) -> None:
    if blocked:
        h.gh.issues[1]["labels"] = [{"name": "needs-human"}]
    h.gh.issues[2] = issue(2)
    original = decision(1, status)
    original["target_parent"] = 1
    h.response(original, decision(2))
    result = h.run(runner=h.fake_runner)
    saved = json.loads((h.run_dir / "agent-output.json").read_text())
    assert saved["decisions"][0]["disposition"] == status
    assert saved["decisions"][0]["rationale"] == original["rationale"]
    index = json.loads((h.run_dir / "agent-input/index.json").read_text())
    assert (index["issues"]["1"]["mechanical_guard"] is not None) == blocked
    assert index["issues"]["2"]["mechanical_guard"] is None
    if blocked and status == "startable":
        assert result["result"] == "failed" and result["candidates"] == []
        assert result["diagnostics"] and h.gh.writes == []
        return
    assert result["result"] == "recorded", result
    expected = [1, 2] if status == "startable" else [2]
    assert [c["parent"] for c in result["candidates"]] == expected
    recorded = result["decisions"][0]
    assert recorded["disposition"] == ("human_hold" if blocked else status)
    mismatch = blocked and status != "human_hold"
    assert bool(result["diagnostics"]) == mismatch
    if mismatch:
        assert original["rationale"] in recorded["rationale"]
        assert f"agentの分類: {status}" in recorded["rationale"]
        assert recorded["facts"] == [
            *original["facts"],
            "機械的阻害条件: #1 にneeds-humanラベルがある。",
        ]
        assert f"agent_disposition={status}" in result["diagnostics"][0]
        assert "recorded_disposition=human_hold" in result["diagnostics"][0]
    else:
        assert recorded["rationale"] == original["rationale"]
        assert recorded["facts"] == original["facts"]


@pytest.mark.parametrize(
    "change", ["empty_facts", "missing_information", "priority", "source", "extra_key"]
)
def test_classification_correction_does_not_mask_invalid_agent_output(
    h: Harness, change: str
) -> None:
    h.gh.issues[1]["labels"] = [{"name": "needs-human"}]
    d = decision(1, "information_needed")
    if change == "empty_facts":
        d["facts"] = []
    elif change == "missing_information":
        d["missing_information"] = []
    elif change == "priority":
        d["priority"] = 1
    elif change == "source":
        d["sources"].append("issue:999")
    else:
        d["launch"] = True
    h.response(d)
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "failed" and result["candidates"] == []
    assert result["diagnostics"] and h.gh.writes == []


def test_ended_owner_with_agreement_and_work_under_another_parent_is_recorded(
    h: Harness,
) -> None:
    h.gh.issues.update(
        {
            10: issue(10, labels=("issue:parent",)),
            11: issue(11),
            20: issue(20, labels=("issue:parent",)),
            21: issue(21),
        }
    )
    h.gh.parents.update({11: 10, 21: 20})
    h.gh.comment_rows[10] = [agreement_comment(10)]
    h.gh.prs = [
        {
            "number": 100,
            "body": "Closes #21 / Refs #11",
            "state": "open",
            "merged_at": None,
            "head": {"ref": "codex/shared-work", "sha": "a" * 40},
        }
    ]
    with h.store.locked():
        h.store.parents["o/r#10"] = {
            "session": "claude:private-ended-owner",
            "status": "ended",
            "generation": 1,
            "tasks": {},
            "reviews": [],
        }
        h.store.save()
    held = decision(10, "existing_owner")
    held["facts"] = [
        "担当記録はended、generation 1。",
        "残る子 #11 は別親 #20 の子 #21 がPR #100で実装中。",
    ]
    held["sources"].extend(["issue:11", "issue:20", "issue:21", "pr:100"])
    held["rationale"] = "担当はended。残る子の実装と重複するため既存担当へ戻す。"
    h.response(
        decision(1),
        held,
        decision(11, "existing_parent", target=10),
        decision(20, "existing_owner"),
        decision(21, "existing_parent", target=20),
    )
    result = h.run()
    assert result["result"] == "recorded", result
    assert [c["parent"] for c in result["candidates"]] == [1]
    assert len(result["diagnostics"]) == 1 and "#10" in result["diagnostics"][0]
    recorded = next(d for d in result["decisions"] if d["issue"] == 10)
    assert recorded["disposition"] == "human_hold"
    assert held["rationale"] in recorded["rationale"]
    assert recorded["facts"][:2] == held["facts"]
    assert json.loads((h.run_dir / "result.json").read_text()) == result
    comment = h.gh.comment_rows[10][-1]
    assert "判定: human_hold" in comment["body"]
    assert "agentの分類: existing_owner" in comment["body"]
    assert held["rationale"] in comment["body"]
    assert "phaseはagreed" in comment["body"]
    assert importlib.import_module("dispatcher.records").owned_comment(
        comment, "operator", "o/r", 10
    )
    writes = len(h.gh.writes)
    repeated = h.run()
    assert repeated["result"] == "recorded", repeated
    assert repeated["input_sha256"] == result["input_sha256"]
    assert repeated["diagnostics"] == result["diagnostics"]
    assert all(p["action"] == "unchanged" for p in repeated["publications"])
    assert len(h.gh.writes) == writes


def test_assignment_contract_checks_external_context_then_parent_basis_under_one_lock(
    h: Harness,
) -> None:
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "recorded", result
    candidate, snapshot = result["candidates"][0], h.snapshot()
    execution = importlib.import_module("parent.execution")
    session_class = importlib.import_module("parent.session").ParentSession
    with h.store.locked():
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)
        auth = execution.validate_candidate(
            session_class(h.gh, h.store, 1, None),
            candidate["basis_sha256"],
            candidate["generation"],
        )
    assert auth["execution_basis_kind"] == "autonomous_decision"
    assert auth["execution_basis_sha256"] == candidate["basis_sha256"]
    h.gh.issues[1]["body"] += "\n\n## 進捗概要\n\n表示だけを更新"
    with h.store.locked():
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)
    h.gh.issues[1]["body"] += "\n\n## 重要な変更\n\n仕様変更"
    with h.store.locked(), pytest.raises(ValueError, match="inputs changed"):
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)


def test_own_comment_is_not_an_instruction_but_same_marker_by_other_author_is(
    h: Harness,
) -> None:
    h.response(decision(1))
    first = h.run()
    assert first["result"] == "recorded", first
    hold = h.gh.comment_rows[1][0]
    h.gh.comment_rows[1].append({**hold, "id": 500, "user": {"login": "human"}})
    second = h.run()
    assert second["result"] == "recorded", second
    assert first["input_sha256"] != second["input_sha256"]
    assert (
        len(
            [
                c
                for c in h.gh.comment_rows[1]
                if c["body"].startswith("<!-- issue-start-judgement:")
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "change",
    [
        "requirements",
        "source",
        "unknown_key",
        "route",
        "bool_priority",
        "unknown_parent",
        "snapshot_version",
    ],
)
def test_modified_candidate_is_rejected_before_reservation(
    h: Harness, change: str
) -> None:
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "recorded", result
    candidate = copy.deepcopy(result["candidates"][0])
    snapshot = h.snapshot()
    if change == "requirements":
        candidate["requirements"]["scope"] = "変更された別の作業範囲"
    elif change == "source":
        candidate["sources"].append("https://uncollected.invalid/evidence")
    elif change == "unknown_key":
        candidate["human_confirmation"] = "fabricated"
    elif change == "route":
        candidate["route"] = "parent"
    elif change == "bool_priority":
        candidate["priority"] = True
    elif change == "unknown_parent":
        candidate["parent"] = 9999
    else:
        snapshot["schema_version"] = 99
    with h.store.locked(), pytest.raises(ValueError):
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)


def test_report_proposal_work_issue_uses_normal_collection_without_starting_extra_agents(
    h: Harness,
) -> None:
    h.gh.issues[1]["body"] = (
        "<!-- ops-report-proposal: R-01 report=#9 option=plan -->\n\n"
        + h.gh.issues[1]["body"]
    )
    h.gh.issues[9] = issue(9, labels=("ops-report",))
    adopted = decision(1)
    adopted["sources"].append("issue:9")
    h.response(adopted, decision(9, "out_of_scope"))
    result = h.run()
    assert result["result"] == "recorded", result
    assert [c["parent"] for c in result["candidates"]] == [1]
    assert len(h.calls.read_text().splitlines()) == 1
    assert h.snapshot()["issues"]["1"]["source_reports"] == [9]
    h.gh.issues[9]["body"] += "\n採用を取り消す。"
    with h.store.locked(), pytest.raises(ValueError, match="inputs changed"):
        h.mod.assert_candidate_fresh(
            h.gh, h.store, h.work, h.snapshot(), result["candidates"][0]
        )


def test_current_own_display_is_never_an_independent_source(h: Harness) -> None:
    h.response(decision(1))
    assert h.run()["result"] == "recorded"
    d = decision(1)
    d["sources"].append(f"comment:1:{h.gh.comment_rows[1][0]['id']}")
    h.response(d)
    writes = len(h.gh.writes)
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert len(h.gh.writes) == writes


def test_human_explicit_stop_is_returned_with_facts_and_resume_condition(
    h: Harness,
) -> None:
    h.gh.comment_rows[1] = [
        {
            "id": 999,
            "user": {"login": "human"},
            "body": "この案は見送る。次の素材を決めるまで着手しない。",
        }
    ]
    held = decision(1, "human_hold")
    held["rationale"] = "人間が素材決定まで見送りを明示した。"
    held["sources"].append("comment:1:999")
    held["resume_condition"] = "人間が素材を決め、見送りを解除したときに再判断する。"
    h.response(held)
    result = h.run()
    assert result["result"] == "recorded" and result["candidates"] == []
    body = h.gh.comment_rows[1][-1]["body"]
    assert "人間が素材決定まで見送りを明示した。" in body
    assert "再開条件:" in body and "次の対応:" in body and "確認事実:" in body
    assert "#issuecomment-999" in body


def test_empty_input_still_launches_one_judgement_turn(h: Harness) -> None:
    h.gh.issues = {}
    h.response()
    result = h.run()
    assert result["result"] == "recorded" and result["candidates"] == []
    assert len(h.calls.read_text().splitlines()) == 1
    assert h.gh.writes == []


@pytest.mark.parametrize(
    "change", ["duplicate_comment", "cycle", "foreign_dependency", "foreign_subissue"]
)
def test_broken_or_unsupported_input_fails_before_agent(
    h: Harness, change: str
) -> None:
    h.response(decision(1))
    if change == "duplicate_comment":
        c = {"id": 1, "user": {"login": "human"}, "body": "指示"}
        h.gh.comment_rows[1] = [c, c]
    elif change == "cycle":
        h.gh.issues[2] = issue(2)
        h.gh.parents = {1: 2, 2: 1}
    else:
        h.gh.issues[2] = issue(2)
        h.gh.issues[2]["repository_url"] = "https://api.github.com/repos/another/repo"
        if change == "foreign_dependency":
            h.gh.dependencies[1] = [2]
        else:
            h.gh.parents[2] = 1
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == [] and not h.calls.exists()


@pytest.mark.parametrize("changed", [False, True])
def test_preview_and_cli_print_no_candidates_or_local_private_information(
    h: Harness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changed: bool,
) -> None:
    h.response(decision(1))
    monkeypatch.setattr(h.mod, "GitHub", lambda repo: h.gh)

    def concurrent_edit(endpoint: str) -> None:
        if changed and h.calls.exists():
            h.gh.on_get = None
            h.gh.issues[1]["body"] += "\n新しい要件"

    h.gh.on_get = concurrent_edit
    code = h.mod.main(
        [
            "--repo",
            "o/r",
            "--cwd",
            str(h.work),
            "--run-dir",
            str(h.root / "cli"),
            "--parent-state-dir",
            str(h.store.directory),
            "--timeout",
            "15",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert code == 0 and summary["result"] == ("deferred" if changed else "preview")
    assert summary["schema_version"] == 2 and summary["candidate_count"] == 0
    assert summary["decision_count"] == (0 if changed else 1)
    assert summary["deferred_count"] == (1 if changed else 0)
    assert str(h.root) not in output and "session" not in output
    assert h.gh.writes == []


def test_dirty_code_and_broken_registry_fail_before_agent(h: Harness) -> None:
    h.response(decision(1))
    (h.work / "README.md").write_text("uncommitted change")
    result = h.run()
    assert result["result"] == "failed" and not h.calls.exists()
    h.git("restore", "README.md")
    h.store.directory.mkdir(exist_ok=True)
    h.store.path.write_text('{"version":1,"parents":')
    result = h.run()
    assert result["result"] == "failed" and not h.calls.exists()
    assert h.gh.writes == []
