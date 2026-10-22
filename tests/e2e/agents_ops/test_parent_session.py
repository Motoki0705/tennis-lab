"""Parent workflow through fake gh, private temporary state and real throwaway git."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"

FAKE_GH = r"""
import json, os, re, sys
args = sys.argv[1:]
with open(os.environ["PARENT_GH_CALLS"], "a") as f:
    f.write(json.dumps(args) + "\n")
state_path = os.environ["PARENT_GH_STATE"]
with open(state_path) as f:
    state = json.load(f)
def response(value, code=200):
    if "--include" in args:
        print(f"HTTP/1.1 {code} {'OK' if code == 200 else 'Not Found'}\n")
    print(json.dumps(value))
    sys.exit(0 if code < 300 else 1)
def save():
    with open(state_path, "w") as f:
        json.dump(state, f)
def fail():
    print("injected GitHub failure", file=sys.stderr)
    sys.exit(1)
if args[:2] == ["repo", "view"]:
    response({"nameWithOwner": "o/r"})
if args[:2] == ["pr", "view"]:
    pr = next(p for p in state["prs"] if p["number"] == int(args[2]))
    values = {"statusCheckRollup": pr.get("statusCheckRollup", []),
              "headRefOid": pr["head"]["sha"]}
    if os.environ.get("PARENT_VIEW_HEAD_RACE"):
        values["headRefOid"] = os.environ["PARENT_VIEW_HEAD_RACE"]
    response({k: values[k] for k in args[args.index("--json") + 1].split(",")})
if args[:2] == ["pr", "merge"]:
    pr = next(p for p in state["prs"] if p["number"] == int(args[2]))
    assert "--merge" in args and "--match-head-commit" in args
    expected = args[args.index("--match-head-commit") + 1]
    if os.environ.get("PARENT_MERGE_FAIL") or pr["state"] != "open" or pr["head"]["sha"] != expected:
        fail()
    if os.environ.get("PARENT_MERGE_HEAD_RACE"):
        pr["head"]["sha"] = os.environ["PARENT_MERGE_HEAD_RACE"]
        save()
        fail()
    pr["state"] = "closed"
    pr["merged"] = True
    pr["merged_at"] = "2026-10-18T03:00:00Z"
    for child in re.findall(r"Closes #(\d+)", pr["body"]):
        if not os.environ.get("PARENT_MERGE_SKIP_CLOSE"):
            state["issues"][child].update(state="closed", state_reason="completed")
    if os.environ.get("PARENT_MERGE_ADVANCE_BASE"):
        import subprocess
        subprocess.run(["git", "merge", "--no-edit", "--no-ff", expected], check=True, capture_output=True)
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        pr["merge_commit_sha"] = base
        if os.environ.get("PARENT_MERGE_UNEXPLAINED_COMMIT"):
            subprocess.run(["git", "commit", "--allow-empty", "-m", "unexplained base advance"], check=True, capture_output=True)
            base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        for other in state["prs"]:
            other["base"]["sha"] = base
            if other["state"] == "open" and str(other["number"]) == os.environ.get("PARENT_MERGE_FOLLOW_MAIN"):
                subprocess.run(["git", "-C", os.environ["PARENT_MERGE_FOLLOW_WORKTREE"], "merge", "--no-edit", base], check=True, capture_output=True)
                other["head"]["sha"] = subprocess.check_output(["git", "-C", os.environ["PARENT_MERGE_FOLLOW_WORKTREE"], "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("PARENT_MERGE_CI_PENDING"):
        target = int(os.environ["PARENT_MERGE_CI_PENDING"])
        next(p for p in state["prs"] if p["number"] == target)["statusCheckRollup"] = []
    save()
    if os.environ.get("PARENT_MERGE_RESPONSE_LOST"):
        fail()
    response({"merged": True})
assert args[0] == "api", args
method = args[args.index("--method") + 1]
endpoint = next(a for a in args if a == "user" or a.startswith("repos/"))
if os.environ.get("PARENT_FAIL_PATH") == endpoint and method != "GET":
    fail()
payload = json.load(sys.stdin) if method != "GET" else None
path, _, query = endpoint.partition("?")
path = path.casefold()
page = int(re.search(r"(?:^|&)page=(\d+)", query)[1]) if "page=" in query else 1
def paged(rows):
    return rows[(page-1)*100:page*100]
if endpoint == "user":
    response({"login": "operator"})
pull = re.fullmatch(r"repos/o/r/pulls/(\d+)", path)
if pull:
    if os.environ.get("PARENT_READ_HEAD_RACE") and int(pull[1]) == 100:
        state["detail_reads"] = state.get("detail_reads", 0) + 1
        if state["detail_reads"] == 2:
            next(p for p in state["prs"] if p["number"] == 100)["head"]["sha"] = os.environ["PARENT_READ_HEAD_RACE"]
        save()
    response(next(p for p in state["prs"] if p["number"] == int(pull[1])))
comment_edit = re.fullmatch(r"repos/o/r/issues/comments/(\d+)", path)
if comment_edit:
    assert method == "PATCH"
    comment = next(c for rows in state["comments"].values() for c in rows if c["id"] == int(comment_edit[1]))
    comment["body"] = payload["body"]
    save()
    response(comment)
if path == "repos/o/r/pulls":
    if os.environ.get("PARENT_CHANGE_CLOSING_ON_READ"):
        state["closing_reads"] = state.get("closing_reads", 0) + 1
        if state["closing_reads"] == 2:
            state["prs"][0]["body"] = os.environ["PARENT_CHANGE_CLOSING_ON_READ"]
        save()
    if os.environ.get("PARENT_CHANGE_PULLS_ON_READ"):
        state["pull_reads"] = state.get("pull_reads", 0) + 1
        if state["pull_reads"] == 2:
            state["prs"][0]["head"]["sha"] = state["prs"][1]["head"]["sha"]
        save()
    response(paged(state["prs"]))
match = re.fullmatch(r"repos/o/r/issues(?:/(\d+))?(?:/(parent|sub_issues|comments))?", path)
assert match, endpoint
n, tail = match.groups()
if n is None:
    if method == "GET":
        response(paged(list(state["issues"].values())))
    assert method == "POST", method
    n = str(state["next"])
    state["next"] += 1
    row = {"id": int(n)+10000, "number": int(n), "repository_url": "https://api.github.com/repos/o/r",
           "title": payload["title"], "body": payload["body"], "labels": [{"name": x} for x in payload["labels"]],
           "state": "open", "state_reason": None}
    state["issues"][n] = row
    save()
    if os.environ.get("PARENT_FAIL_AFTER_CREATE"):
        fail()
    response(row, 201)
if tail != "comments" and n not in state["issues"]:
    response({"message": "Not Found"}, 404)
row = state["issues"].get(n)
if tail == "parent":
    parent = state["parents"].get(n)
    response(state["issues"][str(parent)] if parent else {"message": "No parent issue found"}, 200 if parent else 404)
if tail == "sub_issues":
    if method == "GET":
        response(paged([i for key, i in state["issues"].items() if state["parents"].get(key) == int(n)]))
    child = next(i for i in state["issues"].values() if i["id"] == payload["sub_issue_id"])
    assert str(child["number"]) not in state["parents"]
    state["parents"][str(child["number"])] = int(n)
    save()
    response(child, 201)
if tail == "comments":
    if method == "GET":
        response(paged(state["comments"].get(n, [])))
    comment = {"id": 1 + max((c["id"] for rows in state["comments"].values() for c in rows), default=0), "user": {"login": "operator"}, "body": payload["body"]}
    state["comments"].setdefault(n, []).append(comment)
    save()
    response(comment, 201)
if method == "PATCH":
    for k, v in payload.items():
        row[k] = [{"name": x} for x in v] if k == "labels" else v
    save()
response(row)
"""


def run_git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=directory, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@dataclass
class ParentEnv:
    root: Path
    repo: Path
    env: dict[str, str]
    state_path: Path
    calls_path: Path
    formats: Any

    def state(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.state_path.read_text()))

    def save(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False))

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.calls_path.read_text().splitlines()]

    def spec(self, value: Any) -> str:
        path = self.root / "input.json"
        path.write_text(json.dumps(value, ensure_ascii=False))
        return str(path)

    def note(self, text: str = "確認済みの証拠・次の操作。") -> str:
        path = self.root / "note.md"
        path.write_text(text)
        return str(path)

    def cli(
        self,
        *args: str,
        parent: str = "1",
        repo: str = "o/r",
        ok: bool | None = True,
        extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "parent",
                "--repo",
                repo,
                "--parent",
                parent,
                *args,
            ],
            cwd=self.repo,
            env={**self.env, **(extra or {})},
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        if ok is True:
            assert result.returncode == 0, result.stderr + result.stdout
        elif ok is False:
            assert result.returncode != 0, result.stdout
        return result

    def agreed(self) -> None:
        self.cli("register")
        self.cli("checking")
        self.cli(
            "agree",
            "--spec",
            self.spec(
                {
                    "purpose": "目的を固定する",
                    "scope": "親と子",
                    "behavior": "合意後に実装する",
                    "acceptance": "fakeで検証する",
                    "human_confirmation": "この目的・範囲・動作・受入条件で進めてください。",
                }
            ),
        )

    def child(
        self,
        key: str,
        deps: list[int] | None = None,
        source: bool = False,
        ok: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        sections = {
            name: name + "について具体的に実装する。"
            for name in self.formats.CHILD_REQUIRED
        }
        sections["依存関係"] = "なし。"
        sections["Acceptance checklist"] = (
            "- [ ] 動作する\n  文脈: 親の目的。\n  下流への影響: 誤委任を防ぐ。\n  検証方法・証拠: fake gh。"
        )
        spec = {
            "role": "child",
            "summary": key,
            "parent": {"identifier": "開発", "number": 1},
            "sections": sections,
            "labels": ["ai-proposed"],
        }
        if source:
            # Report adoption is not the parent's existing requirements agreement.
            if "90" not in self.state()["issues"]:
                report_formats = importlib.import_module("opsreport.proposals")
                proposal = report_formats.Proposal(
                    section="その他の助言",
                    title="合意対象の提案",
                    evidence="fake gh",
                    purpose="目的",
                    scope="変更範囲",
                    expected_effect="効果",
                    acceptance=("検証できる",),
                    related_parent=1,
                )
                markdown, _ = report_formats.render_proposals([proposal], "20261012")
                marker = "<!-- ops-report:v2 slot=20261012 since=2026-10-09T00:00:00+00:00 until=2026-10-12T00:00:00+00:00 proposals=1 status=complete -->"
                state = self.state()
                state["issues"]["90"] = {
                    **state["issues"]["1"],
                    "number": 90,
                    "id": 10090,
                    "title": "運用レポート",
                    "labels": [{"name": "ops-report"}],
                    "body": marker
                    + "\n"
                    + report_formats.PROPOSALS_BEGIN
                    + "\n"
                    + markdown.replace("- [ ] **", "- [x] **")
                    + report_formats.PROPOSALS_END
                    + "\n",
                }
                self.save(state)
                self.cli(
                    "receive-proposal", "--report", "90", "--proposal", "R-20261012-01"
                )
                self.cli(
                    "agree",
                    "--report",
                    "90",
                    "--proposal",
                    "R-20261012-01",
                    "--spec",
                    self.spec(
                        {
                            "purpose": "提案の目的を固定する",
                            "scope": "提案由来の子",
                            "behavior": "合意後に実装する",
                            "acceptance": "fakeで検証する",
                            "human_confirmation": "この提案の要件で合意します。",
                        }
                    ),
                )
            spec["source"] = {
                "proposal_id": "R-20261012-01",
                "report_number": 90,
                "option": None,
            }
        manifest = {"key": key, "spec": spec, "dependencies": deps or []}
        return self.cli("create-child", "--spec", self.spec(manifest), ok=ok)

    def detail(self) -> str:
        return self.spec(
            {
                "reason": "仕様の確認待ち",
                "facts": "再現ログを確認済み",
                "impact": "このタスクと依存先",
                "resume_condition": "判断の合意を記録",
                "next_action": "確認を得て再開する",
            }
        )

    def worktree(self, name: str) -> Path:
        worktree = self.repo / ".claude/worktrees" / name
        run_git(self.repo, "worktree", "add", "-b", "codex/" + name, str(worktree))
        return worktree

    def pr(
        self, n: int, child: int, head: str | None = None, branch: str | None = None
    ) -> dict[str, Any]:
        return {
            "number": n,
            "title": f"子issue #{child} の実装",
            "state": "open",
            "draft": False,
            "body": f"Closes #{child}",
            "merged_at": None,
            "head": {
                "sha": head or run_git(self.repo, "rev-parse", "HEAD"),
                "ref": branch or f"codex/child-{child}",
                "repo": {"full_name": "o/r"},
            },
            "base": {
                "sha": run_git(self.repo, "rev-parse", "main"),
                "ref": "main",
                "repo": {"full_name": "o/r"},
            },
        }


@pytest.fixture
def parent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParentEnv:
    monkeypatch.syspath_prepend(str(OPS))
    monkeypatch.syspath_prepend(str(OPS / "ops-report"))
    formats = importlib.import_module("issues")
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "--initial-branch=main")
    run_git(repo, "config", "user.name", "Test Operator")
    run_git(repo, "config", "user.email", "test@example.invalid")
    (repo / "input.txt").write_text("base\n")
    run_git(repo, "add", "input.txt")
    run_git(repo, "commit", "-m", "baseline")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n" + FAKE_GH)
    gh.chmod(0o755)
    for agent in ("claude", "codex"):
        path = bin_dir / agent
        path.write_text(
            "#!/bin/sh\necho 'a test must never launch an agent' >&2\nexit 97\n"
        )
        path.chmod(0o755)
    sections = {
        name: "未合意。"
        for name in formats.PARENT_REQUIRED
        if name != formats.TASKS_HEADING
    }
    body = formats.render_parent_body(sections, [])
    issues = {
        str(n): {
            "id": 10000 + n,
            "number": n,
            "repository_url": "https://api.github.com/repos/o/r",
            "title": f"[開発] 親{n}",
            "body": body,
            "labels": [{"name": "issue:parent"}],
            "state": "open",
            "state_reason": None,
        }
        for n in (1, 2)
    }
    state_path = tmp_path / "github.json"
    state_path.write_text(
        json.dumps(
            {"issues": issues, "parents": {}, "comments": {}, "prs": [], "next": 10}
        )
    )
    calls_path = tmp_path / "calls.jsonl"
    calls_path.touch()
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "CODEX_THREAD_ID",
            "CODEX_SESSION_ID",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_CODE_PARENT_SESSION_ID",
            "CLAUDE_CODE_AGENT_ID",
        }
        and not k.startswith("HERDR_")
    }
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        PYTHONPATH=str(OPS),
        PYTHONDONTWRITEBYTECODE="1",
        XDG_STATE_HOME=str(tmp_path / "private-state"),
        CLAUDE_CODE_SESSION_ID="private-session-a-unique",
        PARENT_GH_STATE=str(state_path),
        PARENT_GH_CALLS=str(calls_path),
    )
    # Parent operations must never reach the host Herdr executable. Display
    # integration tests explicitly replace this sentinel with their fake CLI.
    herdr = bin_dir / "herdr"
    herdr.write_text(
        "#!/bin/sh\necho 'unexpected Herdr access from parent fixture' >&2\nexit 99\n"
    )
    herdr.chmod(0o755)
    return ParentEnv(tmp_path, repo, env, state_path, calls_path, formats)


def test_assignment_is_not_agreement_and_both_inputs(parent_env: ParentEnv) -> None:
    p = parent_env
    assert json.loads(p.cli("status").stdout)["assignment"] == "unassigned"
    p.cli("register", parent="https://github.com/o/r/issues/1")
    assert json.loads(p.cli("status").stdout)["phase"] == "担当指定のみ"
    p.child("forbidden", ok=False)
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(p.root),
        ok=False,
    )
    assert len(p.state()["issues"]) == 2
    assert not any(
        "POST" in call and call[-3:] == ["repos/o/r/issues", "--input", "-"]
        for call in p.calls()
    )
    p.cli("checking")
    assert json.loads(p.cli("status").stdout)["phase"] == "要件確認中"
    p.child("still-forbidden", ok=False)
    p.agreed()
    assert json.loads(p.cli("status").stdout)["phase"] == "要件合意済み"
    assert (
        p.state()["comments"]["1"][-1]["body"]
        .splitlines()[0]
        .startswith("<!-- parent-session:v1 ")
    )
    assert (
        "> この目的・範囲・動作・受入条件で進めてください。"
        in p.state()["comments"]["1"][-1]["body"]
    )
    # Machine-private IDs never enter comments or issue bodies.
    assert "private-session-a-unique" not in json.dumps(p.state())
    p.cli("register", parent="https://github.com/other/repo/issues/1", ok=False)


def test_identity_errors_precede_io_and_codex_uses_environment(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    for env in (
        {"CLAUDE_CODE_SESSION_ID": ""},
        {"CODEX_SESSION_ID": "also-present"},
        {"CLAUDE_CODE_SESSION_ID": "", "CODEX_THREAD_ID": "thread-only"},
        {
            "CLAUDE_CODE_SESSION_ID": "",
            "CODEX_SESSION_ID": "parent",
            "CODEX_THREAD_ID": "child",
        },
    ):
        p.cli("register", ok=False, extra=env)
    assert p.calls() == []
    p.cli(
        "register",
        extra={
            "CLAUDE_CODE_SESSION_ID": "",
            "CODEX_SESSION_ID": "codex-session-private",
            "CODEX_THREAD_ID": "codex-session-private",
        },
    )
    assert "codex-session-private" not in json.dumps(p.state())


def test_exclusive_assignment_release_ended_and_handoff(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    assert "o/r#1" in p.cli("register", parent="2", ok=False).stderr
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    p.cli("register", ok=False, extra=second)
    p.cli("handoff", "--accept", "--note-file", p.note(), ok=False, extra=second)
    p.cli("handoff", "--note-file", p.note())
    assert json.loads(p.cli("status").stdout)["assignment"] == "handoff_pending"
    p.child("pending-rejected", ok=False)
    p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    p.cli("checking", ok=False)  # previous session lost ownership
    p.cli("release", "--ended", "--note-file", p.note(), extra=second)
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"
    p.cli("register", ok=False)
    p.cli("handoff", "--accept", "--note-file", p.note())
    p.cli("release", "--note-file", p.note())
    assert json.loads(p.cli("status").stdout)["assignment"] == "released"
    p.cli("register", parent="2")


@pytest.mark.parametrize(
    ("first_repo", "second_repo"), [("O/R", "o/r"), ("o/r", "O/R")]
)
def test_repository_case_cannot_bypass_assignment(
    parent_env: ParentEnv, first_repo: str, second_repo: str
) -> None:
    p = parent_env
    p.cli("register", repo=first_repo)
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    before = registry.read_bytes()
    calls_before = len(p.calls())
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    denied = p.cli("register", repo=second_repo, extra=second, ok=False)
    assert "別セッション担当中" in denied.stderr
    assert set(json.loads(before)["parents"]) == {"o/r#1"}
    assert registry.read_bytes() == before
    assert all("GET" in call for call in p.calls()[calls_before:])


@pytest.mark.parametrize("legacy_key", [None, "O/R#1"])
def test_repository_case_preserves_delegation_reservation(
    parent_env: ParentEnv, legacy_key: str | None
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    args = (
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(p.worktree("a")),
    )
    assert json.loads(p.cli(*args).stdout)["action"] == "launch"
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    if legacy_key:
        data = json.loads(registry.read_text())
        data["parents"][legacy_key] = data["parents"].pop("o/r#1")
        registry.write_text(json.dumps(data))
    before = registry.read_bytes()
    calls_before = len(p.calls())
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    for repo in ("O/R", "o/r"):
        denied = p.cli("register", repo=repo, extra=second, ok=False)
        assert "別セッション担当中" in denied.stderr
        denied = p.cli(*args, repo=repo, extra=second, ok=False)
        assert "別セッション" in denied.stderr
        denied = p.cli(*args, repo=repo, ok=False)
        assert "already delegated" in denied.stderr
        status = json.loads(p.cli("status", repo=repo).stdout)
        assert status["current_session_is_owner"]
        assert status["tasks"]["10"]["delegated"]
    assert registry.read_bytes() == before
    assert all("GET" in call for call in p.calls()[calls_before:])


@pytest.mark.parametrize("alias_status", ["active", "released"])
def test_repository_alias_collision_fails_closed(
    parent_env: ParentEnv, alias_status: str
) -> None:
    p = parent_env
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    registry.parent.mkdir(parents=True)
    row = {
        "session": "claude:private-session-a-unique",
        "status": "active",
        "generation": 1,
        "tasks": {"10": {"delegated": True}},
        "reviews": [],
    }
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "parents": {
                    "O/R#1": row,
                    "o/r#1": {
                        **row,
                        "session": "claude:private-session-b-unique",
                        "status": alias_status,
                    },
                },
            }
        )
    )
    before = registry.read_bytes()
    for args in (
        ("status",),
        ("register",),
        (
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/a",
            "--worktree",
            str(p.root),
        ),
    ):
        denied = p.cli(*args, ok=False)
        assert "repository alias collision" in denied.stderr
        assert registry.read_bytes() == before
    assert p.calls() == []


@pytest.mark.parametrize("failure", ["after-create", "relation", "progress"])
def test_child_creation_recovers_partial_failure_without_duplicates(
    parent_env: ParentEnv, failure: str
) -> None:
    p = parent_env
    p.agreed()
    if failure == "after-create":
        p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    else:
        p.env["PARENT_FAIL_PATH"] = (
            "repos/o/r/issues/1/sub_issues"
            if failure == "relation"
            else "repos/o/r/issues/1"
        )
    p.child("idempotent", source=True, ok=False)
    p.env.pop("PARENT_FAIL_PATH", None)
    p.env.pop("PARENT_FAIL_AFTER_CREATE", None)
    result = json.loads(p.child("idempotent", source=True).stdout)
    assert result["reused"] and result["child"] == 10
    p.child("idempotent", source=True)
    state = p.state()
    assert len(state["issues"]) == 4 and state["parents"]["10"] == 1
    child = state["issues"]["10"]
    assert child["body"].splitlines()[0].startswith("<!-- ops-report-proposal:")
    assert {x["name"] for x in child["labels"]} == {"ai-proposed", "issue:child"}
    assert "- [ ] #10 idempotent" in state["issues"]["1"]["body"]
    assert "private-session-a-unique" not in json.dumps(state)


def test_child_key_conflict_and_duplicate_existing_fail_closed(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("one")
    p.child("two")
    p.child("one", deps=[11], ok=False)
    state = p.state()
    state["issues"]["99"] = {**state["issues"]["10"], "number": 99, "id": 10099}
    p.save(state)
    p.child("one", ok=False)
    assert len(p.state()["issues"]) == 5


def test_concurrent_registration_never_owns_two_parents(parent_env: ParentEnv) -> None:
    p = parent_env
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda n: p.cli("register", parent=str(n), ok=None), (1, 2))
        )
    assert sorted(r.returncode for r in results) == [0, 2]
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    data = json.loads(registry.read_text())
    assert len(data["parents"]) == 1


def test_rechecking_revokes_agreement_and_missing_confirmation_fails(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("before-change")
    p.cli("checking")
    assert json.loads(p.cli("status").stdout)["phase"] == "要件確認中"
    p.child("cannot-split", ok=False)
    agreement = {
        "purpose": "目的",
        "scope": "範囲",
        "behavior": "動作",
        "acceptance": "条件",
        "human_confirmation": "",
    }
    p.cli("agree", "--spec", p.spec(agreement), ok=False)
    agreement["human_confirmation"] = "この変更した要件で合意します。"
    p.cli("agree", "--spec", p.spec(agreement))
    p.child("after-change")
    assert len(p.state()["issues"]) == 4


def test_report_and_human_assignment_share_entry_and_never_imply_consent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["2"]["body"] = (
        "<!-- ops-report-proposal: R-opaque-2 report=#90 option=none -->\n"
        + state["issues"]["2"]["body"]
    )
    state["issues"]["2"]["labels"].append({"name": "ai-proposed"})
    p.save(state)
    for n in (1, 2):
        result = json.loads(p.cli("register", parent=str(n)).stdout)
        assert result["phase"] == "担当指定のみ"
        p.cli("release", "--note-file", p.note(), parent=str(n))
    assert len(p.state()["issues"]) == 2


def test_proposal_identity_prevents_duplicate_under_new_task_key(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("original", source=True)
    assert json.loads(p.child("new-key", source=True).stdout)["reused"]
    assert len(p.state()["issues"]) == 4


def test_local_identifiers_are_rejected_from_public_notes(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    p.cli("release", "--note-file", p.note(p.env["CLAUDE_CODE_SESSION_ID"]), ok=False)
    assert "private-session-a-unique" not in json.dumps(p.state())
    assert json.loads(p.cli("status").stdout)["assignment"] == "active"


def test_stop_resume_retries_and_repeated_cycles_keep_distinct_notices(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    for _ in range(2):
        for action in ("stop", "resume"):
            p.env["PARENT_FAIL_PATH"] = "repos/o/r/issues/1/comments"
            p.cli(action, "--child", "10", "--spec", p.detail(), ok=False)
            p.env.pop("PARENT_FAIL_PATH")
            p.cli(action, "--child", "10", "--spec", p.detail())
    comments = p.state()["comments"]["1"]
    for event in ("child_stopped", "child_resumed"):
        assert (
            sum(f'"event": "{event}"' in c["body"].splitlines()[0] for c in comments)
            == 2
        )
    assert p.state()["issues"]["10"]["body"].count("再現ログを確認済み") == 4


def test_handoff_retries_do_not_lose_claim_or_repeat_notice(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.cli("handoff", "--note-file", p.note())
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    p.env["PARENT_FAIL_PATH"] = "repos/o/r/issues/1/comments"
    p.cli("handoff", "--accept", "--note-file", p.note(), ok=False, extra=second)
    p.env.pop("PARENT_FAIL_PATH")
    p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    assert json.loads(p.cli("status", extra=second).stdout)["current_session_is_owner"]
    assert (
        sum(
            '"event": "handoff"' in c["body"].splitlines()[0]
            for c in p.state()["comments"]["1"]
        )
        == 1
    )


def test_partial_stop_dependency_resume_and_improvement_preserve_scope(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    p.child("c")
    p.child("d", deps=[11])
    stopped = json.loads(p.cli("stop", "--child", "10", "--spec", p.detail()).stdout)
    assert stopped["progress"] == {
        "10": "stopped",
        "11": "held",
        "12": "continue",
        "13": "held",
    }
    state = p.state()
    body = state["issues"]["10"]["body"]
    assert all(
        word in body
        for word in ("仕様の確認待ち", "再現ログを確認済み", "判断の合意を記録")
    )
    notification = state["comments"]["1"][-1]["body"]
    assert "#10" in notification and "#11" in notification and "#12" in notification
    assert "再現ログを確認済み" not in notification
    assert "#11: 依存保留" in state["issues"]["1"]["body"]
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/b",
        "--worktree",
        str(p.root),
        ok=False,
    )
    before = p.formats.parse_sections(state["issues"]["12"]["body"])[
        "Acceptance checklist"
    ]
    improvement = {
        "purpose": "将来の改善",
        "evidence": "今回観測した事実",
        "scope": "次回の変更だけ",
    }
    p.cli("improvement", "--child", "12", "--spec", p.spec(improvement))
    assert (
        p.formats.parse_sections(p.state()["issues"]["12"]["body"])[
            "Acceptance checklist"
        ]
        == before
    )
    p.cli("improvement", "--child", "12", "--spec", p.spec(improvement))
    assert p.state()["issues"]["12"]["body"].count("将来の改善") == 1
    resumed = json.loads(p.cli("resume", "--child", "10", "--spec", p.detail()).stdout)
    assert set(resumed["progress"].values()) == {"continue"}


def test_delegation_reconcile_and_completed_tasks_do_not_relaunch(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")
    a, b = p.worktree("a"), p.worktree("b")
    for child, name, path in ((10, "a", a), (11, "b", b)):
        assert (
            json.loads(
                p.cli(
                    "delegate",
                    "--child",
                    str(child),
                    "--agent",
                    "codex",
                    "--branch",
                    "codex/" + name,
                    "--worktree",
                    str(path),
                ).stdout
            )["action"]
            == "launch"
        )
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(a),
        ok=False,
    )
    result = json.loads(p.cli("reconcile").stdout)
    assert {t["action"] for t in result["tasks"]} == {"do_not_launch"}
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/a")]
    p.save(state)
    p.cli(
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("関連テストが成功した。"),
    )
    assert "成果PR: #100" in p.state()["issues"]["10"]["body"]
    assert "関連テストが成功した。" in p.state()["issues"]["10"]["body"]
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(a),
        "--restart-evidence-file",
        p.note(),
        ok=False,
    )
    state = p.state()
    state["issues"]["10"].update(state="closed", state_reason="completed")
    p.save(state)
    p.cli("sync")
    result = json.loads(p.cli("reconcile").stdout)
    assert result["tasks"][0]["state"] == "completed"
    assert result["tasks"][0]["prs"][0]["number"] == 100
    assert "- [x] #10 a" in p.state()["issues"]["1"]["body"]
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(a),
        ok=False,
    )


def test_invalid_worktree_and_reused_worker_branch_are_refused(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(p.repo),
        ok=False,
    )
    worktree = p.worktree("a")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(worktree),
    )
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(worktree),
        ok=False,
    )


def test_issue_body_metadata_and_existing_evidence_survive_updates(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("keep", source=True)
    state = p.state()
    before = state["issues"]["10"]["body"]
    body = before.replace(
        "## 対象外",
        "<!-- retain-this-comment -->\n\n```sh\n## example heading\necho keep\n```\n\n## 対象外",
    )
    state["issues"]["10"]["body"] = body
    p.save(state)
    p.cli("stop", "--child", "10", "--spec", p.detail())
    p.cli("resume", "--child", "10", "--spec", p.detail())
    actual = p.state()["issues"]["10"]["body"]
    assert (
        "<!-- retain-this-comment -->\n\n```sh\n## example heading\necho keep\n```"
        in actual
    )
    assert actual.splitlines()[0] == before.splitlines()[0]
    assert "<!-- parent-task:v1" in actual


def review_setup(
    p: ParentEnv, *, independent: bool = False, standalone: bool = False
) -> list[int]:
    if standalone:
        state = p.state()
        state["issues"]["1"].update(
            title="人間が作成した単独タスク",
            labels=[{"name": "bug"}],
            body="## 目的\n単独タスクを実装する。\n\n## 完了条件\n回帰テストが通る。\n",
        )
        p.save(state)
    p.agreed()
    state = p.state()
    state["prs"] = []
    for i in range(1 if standalone else 3):
        if not standalone:
            p.child(f"child-{i}")
        worktree = p.worktree(f"pr-{i}")
        filename = f"change-{i}.txt" if independent else "change.txt"
        (worktree / filename).write_text(f"new {i}\n")
        run_git(worktree, "add", filename)
        run_git(worktree, "commit", "-m", f"child {i}")
        state["prs"].append(
            p.pr(
                100 + i,
                1 if standalone else 10 + i,
                run_git(worktree, "rev-parse", "HEAD"),
            )
        )
    # Read the current issue/comments state, then replace PRs only.
    current = p.state()
    current["prs"] = state["prs"]
    p.save(current)
    return [pr["number"] for pr in state["prs"]]


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "closed", "draft", "unrelated", "fork", "quoted"]
)
def test_invalid_review_set_has_no_public_request(
    parent_env: ParentEnv, case: str
) -> None:
    p = parent_env
    selected = review_setup(p)
    state = p.state()
    if case == "missing":
        selected.pop()
    elif case == "duplicate":
        selected.append(100)
    elif case in {"closed", "draft"}:
        state["prs"][0]["state" if case == "closed" else "draft"] = (
            "closed" if case == "closed" else True
        )
    elif case == "unrelated":
        state["prs"][0]["body"] = "Closes #999"
    elif case == "quoted":
        state["prs"][0]["body"] = "```text\nCloses #10\n```"
    else:
        state["prs"][0]["head"]["repo"]["full_name"] = "other/r"
    p.save(state)
    p.cli("review-prepare", "--prs", *map(str, selected), ok=False)
    assert not any('"event": "review"' in c["body"] for c in p.state()["comments"]["1"])


def test_review_snapshot_explicit_once_handoff_and_states(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    selected = review_setup(p)
    assert json.loads(p.cli("status").stdout)["review"] == "未指示"
    request = json.loads(p.cli("review-prepare", "--prs", *map(str, selected)).stdout)
    assert request["parent_issue"] == 1 and len(request["prs"]) == 3
    assert "private-session" not in json.dumps(request)
    assert (
        json.loads(p.cli("review-prepare", "--prs", *map(str, selected)).stdout)
        == request
    )
    for pr in request["prs"]:
        expected = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                "--full-index",
                "--no-renames",
                "--no-color",
                pr["merge_base_sha"],
                pr["head_sha"],
                "--",
            ],
            cwd=p.repo,
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(expected).hexdigest() == pr["diff_sha256"]
        assert pr["base_sha"] == run_git(p.repo, "rev-parse", "main")
    run_id = request["review_run_id"]
    args = (
        "review-request",
        "--run-id",
        run_id,
        "--instruction-file",
        p.note("この3PRをまとめてレビューしてください。"),
    )
    handed = json.loads(p.cli(*args).stdout)
    assert handed["review_state"] == "instructed"
    p.cli(*args)
    comments = p.state()["comments"]["1"]
    assert (
        sum('"state": "instructed"' in c["body"].splitlines()[0] for c in comments) == 1
    )
    p.cli("review-prepare", "--prs", *map(str, selected), ok=False)
    p.cli(
        "review-state",
        "--run-id",
        run_id,
        "--state",
        "complete",
        "--evidence-file",
        p.note(),
        ok=False,
    )
    for target in ("result_received", "fixes_verified", "complete"):
        p.cli(
            "review-state",
            "--run-id",
            run_id,
            "--state",
            target,
            "--evidence-file",
            p.note(),
        )
    for call in p.calls():
        assert not any(word in call for word in ("merge", "review"))


def test_changed_review_target_refuses_instruction(parent_env: ParentEnv) -> None:
    p = parent_env
    selected = review_setup(p)
    request = json.loads(p.cli("review-prepare", "--prs", *map(str, selected)).stdout)
    state = p.state()
    state["prs"][0]["head"]["sha"] = state["prs"][1]["head"]["sha"]
    p.save(state)
    p.cli(
        "review-request",
        "--run-id",
        request["review_run_id"],
        "--instruction-file",
        p.note(),
        ok=False,
    )
    assert not any(
        '"state": "instructed"' in c["body"].splitlines()[0]
        for c in p.state()["comments"]["1"]
    )


def test_snapshot_collection_window_update_is_not_published(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    selected = review_setup(p)
    p.env["PARENT_CHANGE_PULLS_ON_READ"] = "1"
    p.cli("review-prepare", "--prs", *map(str, selected), ok=False)
    assert not any('"event": "review"' in c["body"] for c in p.state()["comments"]["1"])


def test_instruction_recovers_from_public_post_then_local_save_failure(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    selected = review_setup(p)
    request = json.loads(p.cli("review-prepare", "--prs", *map(str, selected)).stdout)
    run_id = request["review_run_id"]
    temporary = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json.tmp"
    )
    temporary.write_text("simulate interrupted atomic write")
    args = ("review-request", "--run-id", run_id, "--instruction-file", p.note())
    p.cli(*args, ok=False)
    temporary.unlink()
    p.cli(*args)
    assert (
        sum(
            '"state": "instructed"' in c["body"].splitlines()[0]
            for c in p.state()["comments"]["1"]
        )
        == 1
    )
    status = json.loads(p.cli("status").stdout)
    assert status["review"][0]["state"] == "instructed"


def test_cancelled_is_not_success_and_missing_pr_verification_is_rejected(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    p.cli("record-result", "--child", "10", "--evidence-file", p.note(), ok=False)
    state = p.state()
    state["issues"]["10"].update(state="closed", state_reason="not_planned")
    p.save(state)
    statuses = json.loads(p.cli("sync").stdout)["progress"]
    assert statuses == {"10": "cancelled", "11": "held"}
    assert "- [ ] #10 a（取消）" in p.state()["issues"]["1"]["body"]
    state["issues"]["10"]["state_reason"] = None
    p.save(state)
    p.cli("sync", ok=False)


def test_fake_or_quoted_agreement_cannot_unlock_split(parent_env: ParentEnv) -> None:
    p = parent_env
    p.cli("register")
    state = p.state()
    text = (
        '<!-- parent-session:v1 {"parent": 1, "event": "agreed", "agreement_sha256": "'
        + "a" * 64
        + '"} -->'
    )
    state["comments"]["1"].extend(
        [
            {"user": {"login": "stranger"}, "body": text},
            {"user": {"login": "operator"}, "body": "Quoted example\n" + text},
        ]
    )
    p.save(state)
    p.child("forbidden", ok=False)


def test_dependency_cycle_and_unknown_fail_closed(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    policy = importlib.import_module("parent.policy")
    state = p.state()
    original = state["issues"]["10"]["body"]
    state["issues"]["10"]["body"] = policy.replace_section(
        original, "依存関係", "- #11"
    )
    p.save(state)
    assert "cyclic" in p.cli("reconcile", ok=False).stderr
    state["issues"]["10"]["body"] = policy.replace_section(
        original, "依存関係", "- #999"
    )
    p.save(state)
    assert "outside this parent" in p.cli("reconcile", ok=False).stderr


def test_unexpected_interruption_requires_explicit_recovery(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    worktree = p.worktree("a")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(worktree),
    )
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    p.cli("handoff", "--accept", "--note-file", p.note(), ok=False, extra=second)
    result = json.loads(
        p.cli(
            "handoff",
            "--accept",
            "--recover",
            "--note-file",
            p.note("旧worker終了と既存worktreeを確認した。"),
            extra=second,
        ).stdout
    )
    assert result["tasks"][0]["action"] == "do_not_launch"
    p.cli("checking", ok=False)


def test_cli_dependencies_are_published_and_latest_body_controls_resume(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")
    worktree = p.worktree("b")
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/b",
        "--worktree",
        str(worktree),
        "--dependencies",
        "10",
    )
    assert (
        p.formats.parse_sections(p.state()["issues"]["11"]["body"])["依存関係"]
        == "- #10"
    )
    p.cli("checking")
    p.cli("stop", "--child", "10", "--spec", p.detail())
    assert json.loads(p.cli("reconcile").stdout)["tasks"][1]["state"] == "held"
    # Human correction of dependencies takes precedence over the old local cache.
    state = p.state()
    policy = importlib.import_module("parent.policy")
    state["issues"]["11"]["body"] = policy.replace_section(
        state["issues"]["11"]["body"], "依存関係", "なし。"
    )
    p.save(state)
    result = json.loads(p.cli("reconcile").stdout)
    assert result["tasks"][1]["state"] == "continue"
    assert result["tasks"][1]["assignment"]["dependencies"] == []


# The consumer is tested against the producer's actual public/local contract above.
REVIEW_SCRIPT = OPS / "review/parent_review.py"
FAKE_PARENT_REVIEWER = r"""
import json, os, pathlib, sys, time
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
cwd = pathlib.Path(opt("--cwd"))
context = json.loads((cwd / "context.json").read_text())
request = json.loads((cwd / "request.json").read_text())
call = {"args": args, "cwd": str(cwd), "context": context,
        "prompt": pathlib.Path(opt("--prompt-file")).read_text(),
        "diffs": {str(p["number"]): (cwd / f'pr-{p["number"]}.diff').read_text() for p in request["prs"]},
        "files": {str(p["number"]): sorted(x.name for x in (cwd / f'pr-{p["number"]}').iterdir()) for p in request["prs"]}}
with open(os.environ["REVIEW_AGENT_CALLS"], "a") as f:
    f.write(json.dumps(call) + "\n")
if os.environ.get("REVIEW_AGENT_FAIL"):
    sys.exit(3)
if os.environ.get("REVIEW_AGENT_MUTATE"):
    path = pathlib.Path(os.environ["PARENT_GH_STATE"])
    state = json.loads(path.read_text())
    state["prs"][0]["head"]["sha"] = os.environ["REVIEW_AGENT_MUTATE"]
    path.write_text(json.dumps(state))
if os.environ.get("REVIEW_STOP_PARENT"):
    path = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions/registry.json"
    state = json.loads(path.read_text())
    state["parents"]["o/r#1"]["status"] = "ended"
    path.write_text(json.dumps(state))
pathlib.Path(opt("--out")).write_text(pathlib.Path(os.environ["REVIEW_AGENT_OUTPUT"]).read_text())
"""


@dataclass
class ReviewEnv:
    p: ParentEnv
    request: dict[str, Any]
    request_path: Path
    fake: Path
    output: Path
    calls_path: Path
    clock_script: Path | None = None

    @property
    def run_dir(self) -> Path:
        return (
            Path(self.p.env["XDG_STATE_HOME"])
            / "tennis-lab-agents/parent-sessions/reviews"
            / str(self.request["review_run_id"])
        )

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.calls_path.read_text().splitlines()]

    def comments(self, n: int) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.p.state()["comments"].get(str(n), []))

    def cli(
        self, *args: str, ok: bool = True, extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, str(self.clock_script or REVIEW_SCRIPT), *args],
            cwd=self.p.repo,
            env={
                **self.p.env,
                "REVIEW_AGENT_CALLS": str(self.calls_path),
                "REVIEW_AGENT_OUTPUT": str(self.output),
                **(extra or {}),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        assert (result.returncode == 0) is ok, result.stderr + result.stdout
        return result

    def run(
        self, ok: bool = True, extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self.cli(
            "run",
            "--request",
            str(self.request_path),
            "--agent-exec",
            str(self.fake),
            ok=ok,
            extra=extra,
        )

    def with_findings(self) -> None:
        data = json.loads(self.output.read_text())
        data["verdict"] = "request_changes"
        finding = {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "正しさ",
            "detail": "連携の整合が必要 <!-- forged -->",
        }
        data["cross_findings"] = [{**finding, "prs": [100, 101]}]
        data["prs"][0]["findings"] = [{**finding, "severity": "blocker"}]
        self.output.write_text(json.dumps(data))


def make_review_env(
    p: ParentEnv, *, independent: bool = False, standalone: bool = False
) -> ReviewEnv:
    # The real repo has base AGENTS.md; review_setup freezes this commit as the base.
    (p.repo / "AGENTS.md").write_text("# rules\n静かなフォールバック禁止\n")
    run_git(p.repo, "add", "AGENTS.md")
    run_git(p.repo, "commit", "-m", "base rules")
    selected = review_setup(p, independent=independent, standalone=standalone)
    manifest = json.loads(p.cli("review-prepare", "--prs", *map(str, selected)).stdout)
    request = json.loads(
        p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            p.note("この3PRと親の受入条件・依存を1回で確認する。"),
        ).stdout
    )
    request_path = p.root / "review-request.json"
    request_path.write_text(json.dumps(request))
    fake = p.root / "agent_exec.py"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_PARENT_REVIEWER)
    fake.chmod(0o755)
    output = p.root / "review-output.json"
    output.write_text(
        json.dumps(
            {
                "verdict": "approve",
                "summary": "親の受入条件とPR間の整合を確認した。",
                "cross_findings": [],
                "prs": [
                    {
                        "number": n,
                        "coverage": "reviewed",
                        "summary": "個別diff確認済み",
                        "findings": [],
                    }
                    for n in selected
                ],
            }
        )
    )
    calls = p.root / "review-calls.jsonl"
    calls.touch()
    # Exercise the real CLI with a fixed clock in the test subprocess only.
    # Shared WSL hosts can move wall time backwards while a fake review runs.
    # Timestamp validation remains strict in production and its policy tests.
    clock = p.root / "review-with-fixed-clock.py"
    clock.write_text(
        "import datetime, runpy, sys\n"
        "class FixedClock(datetime.datetime):\n"
        "    @classmethod\n"
        "    def now(cls, tz=None):\n"
        "        return cls(2026, 10, 18, tzinfo=tz)\n"
        "datetime.datetime = FixedClock\n"
        f"sys.path.insert(0, {str(REVIEW_SCRIPT.parent)!r})\n"
        f"runpy.run_path({str(REVIEW_SCRIPT)!r}, run_name='__main__')\n"
    )
    return ReviewEnv(p, request, request_path, fake, output, calls, clock)


@pytest.fixture
def review_env(parent_env: ParentEnv) -> ReviewEnv:
    return make_review_env(parent_env)


def review_policy_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS / "review"))
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("parent_review_policy")


def test_explicit_parent_review_runs_codex_exactly_once_and_records_all_prs(
    review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = review_env
    r.with_findings()
    result = json.loads(r.run().stdout)["result"]
    assert result["status"] == "succeeded"
    assert result["verdict"] == "request_changes"
    assert result["request"]["parent_issue"] == 1
    assert result["request"]["prs"] == r.request["prs"]
    assert result["cross_findings"][0]["id"] == "cross-1"
    assert [p["child_issues"] for p in result["prs"]] == [[10], [11], [12]]
    assert len(r.calls()) == 1
    call = r.calls()[0]
    args = call["args"]
    assert args[args.index("--agent") + 1] == "codex"
    assert args[args.index("--profile") + 1] == "review"
    assert args[args.index("--mode") + 1] == "read-only"
    assert all(str(n) in call["diffs"] for n in (100, 101, 102))
    assert all("change.txt" in files for files in call["files"].values())
    assert "全体の完了条件" in call["context"]["parent"]["body"]
    assert any(
        '"event": "agreed"' in c["body"] for c in call["context"]["parent_comments"]
    )
    assert len(call["context"]["children"]) == 3
    assert "インターフェース" in call["prompt"]
    policy = review_policy_module(monkeypatch)
    parent_results = [
        x for x in policy.records(r.comments(1), "operator") if x["kind"] == "result"
    ]
    assert parent_results == [result]
    for n in (100, 101, 102):
        [projection] = policy.records(r.comments(n), "operator")
        assert projection["pr"] == n and projection["coverage"] == "reviewed"
        assert projection["snapshot"] == next(
            s for s in r.request["prs"] if s["number"] == n
        )
        assert "#issuecomment-" in projection["parent_run_url"]
    assert "&lt;!-- forged --&gt;" in r.comments(100)[0]["body"]
    assert not Path(call["cwd"]).exists()
    duplicate = json.loads(r.run().stdout)
    assert duplicate["reused"] and duplicate["result"] == result
    assert len(r.calls()) == 1
    assert (
        len(
            [
                x
                for x in policy.records(r.comments(1), "operator")
                if x["kind"] == "result"
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "case",
    [
        "unassigned",
        "ended",
        "handoff",
        "checking",
        "no_instruction",
        "foreign_instruction",
        "quoted_instruction",
        "old_generation",
        "wrong_session",
        "duplicate_pr",
        "changed_request",
    ],
)
def test_invalid_or_uninstructed_review_never_launches(
    review_env: ReviewEnv, case: str
) -> None:
    r = review_env
    p = r.p
    registry_path = r.run_dir.parents[1] / "registry.json"
    registry = json.loads(registry_path.read_text())
    row = registry["parents"]["o/r#1"]
    state = p.state()
    extra: dict[str, str] = {}
    if case == "unassigned":
        registry["parents"] = {}
    elif case in {"ended", "handoff"}:
        row["status"] = "ended" if case == "ended" else "handoff_pending"
    elif case == "checking":
        p.cli("checking")
        state = p.state()
    elif case in {"no_instruction", "foreign_instruction", "quoted_instruction"}:
        instruction = state["comments"]["1"][-1]
        if case == "no_instruction":
            state["comments"]["1"].pop()
        elif case == "foreign_instruction":
            instruction["user"]["login"] = "attacker"
        else:
            instruction["body"] = "引用:\n" + instruction["body"]
    elif case == "old_generation":
        row["generation"] += 1
    elif case == "wrong_session":
        extra["CLAUDE_CODE_SESSION_ID"] = "different-session"
    elif case == "duplicate_pr":
        r.request["prs"].append(r.request["prs"][0])
        r.request_path.write_text(json.dumps(r.request))
    else:
        r.request["reason"] = "改変した要求"
        r.request_path.write_text(json.dumps(r.request))
    registry_path.write_text(json.dumps(registry))
    p.save(state)
    r.run(ok=False, extra=extra)
    assert r.calls() == []
    assert not (r.run_dir / "execution.json").exists()


@pytest.mark.parametrize(
    "case",
    [
        "head",
        "base",
        "base_ref",
        "removed",
        "added",
        "closed",
        "draft",
        "fork",
        "unrelated",
        "ci_failed_collection",
        "diff",
    ],
)
def test_changed_or_incomplete_review_targets_fail_closed(
    review_env: ReviewEnv, case: str
) -> None:
    r = review_env
    state = r.p.state()
    pr = state["prs"][0]
    extra: dict[str, str] = {}
    if case in {"head", "base"}:
        pr[case]["sha"] = state["prs"][1]["head"]["sha"]
    elif case == "base_ref":
        pr["base"]["ref"] = "retarget"
    elif case == "removed":
        state["prs"].pop()
    elif case == "added":
        state["prs"].append({**pr, "number": 103})
    elif case in {"closed", "draft", "unrelated", "fork"}:
        if case == "closed":
            pr["state"] = "closed"
        elif case == "draft":
            pr["draft"] = True
        elif case == "unrelated":
            pr["body"] = "Closes #999"
        else:
            pr["head"]["repo"]["full_name"] = "fork/r"
    elif case == "ci_failed_collection":
        pr["statusCheckRollup"] = None
    else:
        (r.run_dir / "pr-100.diff").write_bytes(b"changed bytes")
    r.p.save(state)
    value = r.run(ok=False, extra=extra)
    assert r.calls() == []
    if case != "diff":
        result = json.loads(value.stdout)["result"]
        assert result["status"] == "failed" and result["verdict"] == "incomplete"
        assert all(p["coverage"] == "not_reviewed" for p in result["prs"])
        r.run(ok=False)
        assert r.calls() == []


@pytest.mark.parametrize(
    "case", ["agent", "parse", "partial", "post", "head_during_review"]
)
def test_failed_run_is_durable_and_never_automatically_retried(
    review_env: ReviewEnv, case: str
) -> None:
    r = review_env
    extra = {}
    if case == "agent":
        extra["REVIEW_AGENT_FAIL"] = "1"
    elif case == "parse":
        r.output.write_text("invalid JSON")
    elif case == "partial":
        data = json.loads(r.output.read_text())
        data["verdict"] = "incomplete"
        data["prs"][1]["coverage"] = "not_reviewed"
        r.output.write_text(json.dumps(data))
    elif case == "post":
        extra["PARENT_FAIL_PATH"] = "repos/o/r/issues/101/comments"
    else:
        extra["REVIEW_AGENT_MUTATE"] = r.request["prs"][1]["head_sha"]
    value = json.loads(r.run(ok=False, extra=extra).stdout)
    assert value["result"]["status"] == "failed"
    assert len(r.calls()) == 1
    r.run(ok=False)
    assert len(r.calls()) == 1
    result = json.loads((r.run_dir / "result.json").read_text())
    assert result["status"] == "failed"
    assert all(
        '"status":"succeeded"' not in c["body"].splitlines()[0] for c in r.comments(1)
    )


def test_interrupted_consumed_run_is_not_restarted(review_env: ReviewEnv) -> None:
    r = review_env
    r.run()
    execution = json.loads((r.run_dir / "execution.json").read_text())
    execution["status"] = "running"
    (r.run_dir / "execution.json").write_text(json.dumps(execution))
    result = r.run(ok=False)
    assert "running/interrupted" in result.stderr
    assert len(r.calls()) == 1


def test_new_explicit_run_requires_reason_and_is_visible_as_exception(
    review_env: ReviewEnv,
) -> None:
    r = review_env
    r.run()
    r.p.cli("review-prepare", "--prs", "100", "101", "102", ok=False)
    manifest = json.loads(
        r.p.cli(
            "review-prepare",
            "--prs",
            "100",
            "101",
            "102",
            "--reason-file",
            r.p.note("追加レビューが必要な理由を親が明示。"),
        ).stdout
    )
    new_request = json.loads(
        r.p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            r.p.note("例外として新しいレビューを実行。"),
        ).stdout
    )
    r.request_path.write_text(json.dumps(new_request))
    result = json.loads(r.run().stdout)["result"]
    assert len(r.calls()) == 2
    assert result["exception"]["previous_runs"] == [r.request["review_run_id"]]
    assert "追加レビュー" in result["exception"]["reason"]
    r.request_path.write_text(json.dumps(r.request))
    r.run(ok=False)
    assert len(r.calls()) == 2


def test_fix_and_regression_test_push_are_resolved_without_codex_again(
    review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = review_env
    r.with_findings()
    result = json.loads(r.run().stdout)["result"]
    worktree = r.p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("fixed\n")
    (worktree / "tests").mkdir()
    (worktree / "tests/test_change.py").write_text(
        "# Regression for the boundary condition\n"
    )
    paths = ["change.txt", "tests/test_change.py"]
    run_git(worktree, "add", ".")
    run_git(worktree, "commit", "-m", "fix blocker and cross finding")
    head = run_git(worktree, "rev-parse", "HEAD")
    state = r.p.state()
    state["prs"][0]["head"]["sha"] = head
    r.p.save(state)
    proof = r.p.spec(
        {
            "ci": "必須CIすべて通過、head=" + head,
            "regression_tests": "回帰テスト4件成功",
            "resolution": "指摘の再現条件が解消したことを差分とテストで確認。",
            "reviewed_paths": paths,
        }
    )
    args = (
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--fixed-head",
        head,
        "--evidence",
        proof,
    )
    first = json.loads(r.cli(*args, "--finding", "pr-100-1").stdout)
    assert first["coverage"]["covered"]
    r.cli(*args, "--finding", "cross-1")
    r.cli(*args, "--finding", "cross-1")
    assert len(r.calls()) == 1
    policy = review_policy_module(monkeypatch)
    resolutions = [
        x
        for x in policy.records(r.comments(1), "operator")
        if x["kind"] == "resolution"
    ]
    assert len(resolutions) == 2
    assert {x["finding"] for x in resolutions} == {"pr-100-1", "cross-1"}
    assert all(
        x["fixed_snapshot"]["head_sha"] == head
        and x["changed_paths"] == paths
        and x["evidence"]["reviewed_paths"] == paths
        for x in resolutions
    )
    latest = [
        {**p, "head_sha": head} if p["number"] == 100 else p for p in r.request["prs"]
    ]
    # Use the recomputed diff, not a hand-edited snapshot, to exercise the gate boundary.
    latest[0] = resolutions[0]["fixed_snapshot"]
    monkeypatch.chdir(r.p.repo)
    inputs = importlib.import_module("review_workspace")
    fix_digest = hashlib.sha256(
        inputs.full_diff(r.request["prs"][0]["head_sha"], head)
    ).hexdigest()
    assert fix_digest == resolutions[0]["fix_diff_sha256"]
    coverage = policy.evaluate_coverage(
        result,
        latest,
        resolutions,
        {100: paths},
        fix_diff_digests={100: fix_digest},
    )
    assert coverage.covered
    assert coverage.unresolved == ((101, "cross-1"),)
    assert all((100, f) not in coverage.unresolved for f in ("cross-1", "pr-100-1"))
    r.run(ok=False)  # historical run is not approval for the pushed head
    assert len(r.calls()) == 1


@pytest.mark.parametrize(
    "case", ["missing_path", "missing_declaration", "empty_reason"]
)
def test_resolve_rejects_unconfirmed_fix_scope_before_publication(
    review_env: ReviewEnv, case: str
) -> None:
    r = review_env
    r.with_findings()
    r.run()
    worktree = r.p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("fixed\n")
    (worktree / "unconfirmed.txt").write_text("extra change\n")
    run_git(worktree, "add", ".")
    run_git(worktree, "commit", "-m", "fix with extra file")
    head = run_git(worktree, "rev-parse", "HEAD")
    state = r.p.state()
    state["prs"][0]["head"]["sha"] = head
    r.p.save(state)
    proof: dict[str, Any] = {
        "ci": "CI成功 head=" + head,
        "regression_tests": "回帰テスト成功",
        "resolution": "修正差分を確認",
        "reviewed_paths": ["change.txt", "unconfirmed.txt"],
    }
    if case == "missing_path":
        proof["reviewed_paths"] = ["change.txt"]
    elif case == "missing_declaration":
        del proof["reviewed_paths"]
    else:
        proof["resolution"] = " "
    before = r.comments(1), r.comments(100)
    r.cli(
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "pr-100-1",
        "--fixed-head",
        head,
        "--evidence",
        r.p.spec(proof),
        ok=False,
    )
    assert (r.comments(1), r.comments(100)) == before
    assert not list(r.run_dir.glob("resolution-*.json"))
    assert len(r.calls()) == 1


def test_sequential_peer_merges_reuse_one_review_run(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = make_review_env(parent_env, independent=True)
    result = json.loads(r.run().stdout)["result"]
    assert result["verdict"] == "approve"
    policy = review_policy_module(monkeypatch)
    inputs = importlib.import_module("review_workspace")
    snapshot = importlib.import_module("parent.snapshot").snapshot
    monkeypatch.chdir(r.p.repo)
    state = r.p.state()
    merged = []
    for index in (0, 1):
        pr = state["prs"][index]
        run_git(
            r.p.repo,
            "merge",
            "--no-ff",
            "-m",
            f"merge PR {pr['number']}",
            pr["head"]["sha"],
        )
        tip = run_git(r.p.repo, "rev-parse", "HEAD")
        pr.update(state="closed", merged=True, merge_commit_sha=tip)
        merged.append(pr)
        remaining = []
        current = []
        for i, raw in enumerate(state["prs"]):
            if i <= index:
                current.append(r.request["prs"][i])
            else:
                raw["base"]["sha"] = tip
                entry, _ = snapshot(raw)
                remaining.append(entry)
                current.append(entry)
                assert all(
                    entry[k] == r.request["prs"][i][k]
                    for k in ("head_sha", "merge_base_sha", "diff_sha256")
                )
        advances = inputs.verify_base_advances(result, remaining, merged)
        assert len(advances) == index + 1
        coverage = policy.evaluate_coverage(
            result, current, [], {}, fix_diff_digests={}, base_advances=advances
        )
        assert coverage.covered and not coverage.unresolved
        assert not policy.evaluate_coverage(
            result, current, [], {}, fix_diff_digests={}
        ).covered
    assert len(r.calls()) == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_merge",
        "unmerged",
        "foreign_pr",
        "wrong_base",
        "wrong_repo",
        "wrong_branch",
        "unknown_commit",
        "rewind",
    ],
)
def test_base_advance_requires_github_merge_and_complete_git_history(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    r = make_review_env(parent_env, independent=True)
    result = json.loads(r.run().stdout)["result"]
    review_policy_module(monkeypatch)
    inputs = importlib.import_module("review_workspace")
    snapshot = importlib.import_module("parent.snapshot").snapshot
    monkeypatch.chdir(r.p.repo)
    state = r.p.state()
    peer, target = state["prs"][:2]
    run_git(r.p.repo, "merge", "--no-ff", "-m", "merge PR 100", peer["head"]["sha"])
    tip = run_git(r.p.repo, "rev-parse", "HEAD")
    peer.update(state="closed", merged=True, merge_commit_sha=tip)
    target["base"]["sha"] = tip
    evidence = [peer]
    if case == "missing_merge":
        evidence = []
    elif case == "unmerged":
        peer["merged"] = False  # merge_commit_sha alone may be a GitHub test merge.
    elif case == "foreign_pr":
        peer["number"] = 999
    elif case == "wrong_base":
        peer["base"]["ref"] = "other"
    elif case == "wrong_repo":
        peer["base"]["repo"]["full_name"] = "other/repo"
    elif case == "wrong_branch":
        peer["head"]["ref"] = "codex/other"
    elif case == "unknown_commit":
        (r.p.repo / "unreviewed.txt").write_text("unexpected base change\n")
        run_git(r.p.repo, "add", "unreviewed.txt")
        run_git(r.p.repo, "commit", "-m", "unexplained commit after the merge")
        target["base"]["sha"] = run_git(r.p.repo, "rev-parse", "HEAD")
    else:
        target["base"]["sha"] = run_git(
            r.p.repo, "rev-parse", r.request["prs"][1]["base_sha"] + "^"
        )
    current, _ = snapshot(target)
    with pytest.raises(ValueError):
        inputs.verify_base_advances(result, [current], evidence)
    assert len(r.calls()) == 1


def test_resolve_after_peer_merge_keeps_explicit_fix_coverage(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = make_review_env(parent_env, independent=True)
    output = json.loads(r.output.read_text())
    output["verdict"] = "request_changes"
    output["prs"][1]["findings"] = [
        {
            "severity": "major",
            "path": "change-1.txt",
            "line": 1,
            "rule": "境界条件",
            "detail": "回帰テストを伴う修正が必要",
        }
    ]
    r.output.write_text(json.dumps(output))
    result = json.loads(r.run().stdout)["result"]
    state = r.p.state()
    peer, target = state["prs"][:2]
    run_git(r.p.repo, "merge", "--no-ff", "-m", "merge PR 100", peer["head"]["sha"])
    tip = run_git(r.p.repo, "rev-parse", "HEAD")
    peer.update(state="closed", merged=True, merge_commit_sha=tip)
    for raw in state["prs"][1:]:
        raw["base"]["sha"] = tip
    worktree = r.p.repo / ".claude/worktrees/pr-1"
    (worktree / "change-1.txt").write_text("fixed boundary\n")
    (worktree / "test_change.py").write_text("# Boundary regression\n")
    run_git(worktree, "add", ".")
    run_git(worktree, "commit", "-m", "fix with regression")
    head = run_git(worktree, "rev-parse", "HEAD")
    target["head"]["sha"] = head
    r.p.save(state)
    paths = ["change-1.txt", "test_change.py"]
    proof = r.p.spec(
        {
            "ci": "CI成功 head=" + head,
            "regression_tests": "境界回帰テスト成功",
            "resolution": "修正と境界回帰テストの差分を確認",
            "reviewed_paths": paths,
        }
    )
    fixed = json.loads(
        r.cli(
            "resolve",
            "--run",
            r.request["review_run_id"],
            "--pr",
            "101",
            "--finding",
            "pr-101-1",
            "--fixed-head",
            head,
            "--evidence",
            proof,
        ).stdout
    )
    assert fixed["coverage"]["covered"] and not fixed["coverage"]["unresolved"]
    policy = review_policy_module(monkeypatch)
    inputs = importlib.import_module("review_workspace")
    snapshot = importlib.import_module("parent.snapshot").snapshot
    monkeypatch.chdir(r.p.repo)
    latest = [r.request["prs"][0], *(snapshot(raw)[0] for raw in state["prs"][1:])]
    advances = inputs.verify_base_advances(result, latest[1:], [peer])
    record = fixed["resolution"]
    actual_digest = hashlib.sha256(
        inputs.full_diff(record["reviewed_head"], head)
    ).hexdigest()
    coverage = policy.evaluate_coverage(
        result,
        latest,
        [record],
        {101: paths},
        fix_diff_digests={101: actual_digest},
        base_advances=advances,
    )
    assert coverage.covered and not coverage.unresolved
    assert len(r.calls()) == 1


def test_merging_main_into_a_reviewed_pr_requires_a_new_explicit_review(
    parent_env: ParentEnv,
) -> None:
    r = make_review_env(parent_env, independent=True)
    r.with_findings()
    r.run()
    state = r.p.state()
    peer, target = state["prs"][:2]
    run_git(r.p.repo, "merge", "--no-ff", "-m", "merge peer", target["head"]["sha"])
    tip = run_git(r.p.repo, "rev-parse", "HEAD")
    target.update(state="closed", merged=True, merge_commit_sha=tip)
    worktree = r.p.repo / ".claude/worktrees/pr-0"
    run_git(worktree, "merge", "--no-ff", "-m", "follow main", tip)
    head = run_git(worktree, "rev-parse", "HEAD")
    peer["head"]["sha"] = head
    peer["base"]["sha"] = tip
    r.p.save(state)
    failure = r.cli(
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "pr-100-1",
        "--fixed-head",
        head,
        "--evidence",
        r.p.spec(
            {
                "ci": "成功",
                "regression_tests": "成功",
                "resolution": "main追従を確認",
                "reviewed_paths": ["change-1.txt"],
            }
        ),
        ok=False,
    )
    assert "a new explicit review is required" in failure.stderr
    assert not list(r.run_dir.glob("resolution-*.json"))
    assert len(r.calls()) == 1


def test_no_poll_or_timer_entrypoint_and_events_never_launch(
    review_env: ReviewEnv,
) -> None:
    r = review_env
    state = r.p.state()
    state["prs"][0]["statusCheckRollup"] = [
        {"__typename": "StatusContext", "context": "CI", "state": "SUCCESS"}
    ]
    state["prs"][0]["head"]["sha"] = state["prs"][1]["head"]["sha"]
    state["prs"].append({**state["prs"][0], "number": 103})
    r.p.save(state)
    r.p.cli("reconcile")
    r.p.cli("sync")
    r.cli(ok=False)  # no default command, PR enumeration, or scheduling entrypoint
    assert r.calls() == []
    assert not (OPS / "review/cross_review.py").exists()
    assert not list((OPS / "review").glob("systemd/*"))
    assert list(OPS.glob("*/systemd/*report.timer"))


def test_large_diff_is_available_in_full_without_extra_review(
    review_env: ReviewEnv,
) -> None:
    r = review_env
    # Prepare a new explicit snapshot after expanding a real PR diff.
    worktree = r.p.repo / ".claude/worktrees/pr-0"
    text = "full-diff-content\n" * 15_000
    (worktree / "change.txt").write_text(text)
    run_git(worktree, "add", "change.txt")
    run_git(worktree, "commit", "-m", "large diff")
    state = r.p.state()
    state["prs"][0]["head"]["sha"] = run_git(worktree, "rev-parse", "HEAD")
    r.p.save(state)
    manifest = json.loads(
        r.p.cli(
            "review-prepare",
            "--prs",
            "100",
            "101",
            "102",
            "--reason-file",
            r.p.note("対象更新、大きなdiffを全文確認する。"),
        ).stdout
    )
    request = json.loads(
        r.p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            r.p.note(),
        ).stdout
    )
    r.request_path.write_text(json.dumps(request))
    r.run()
    assert len(r.calls()) == 1
    call = r.calls()[0]
    assert call["diffs"]["100"].count("full-diff-content") == 15_000
    assert call["prompt"].count("full-diff-content") < 15_000
    assert "一部のみ。全文ファイルを必ず読む" in call["prompt"]


def test_concurrent_resends_consume_the_run_only_once(review_env: ReviewEnv) -> None:
    r = review_env

    def invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(REVIEW_SCRIPT),
                "run",
                "--request",
                str(r.request_path),
                "--agent-exec",
                str(r.fake),
            ],
            cwd=r.p.repo,
            env={
                **r.p.env,
                "REVIEW_AGENT_CALLS": str(r.calls_path),
                "REVIEW_AGENT_OUTPUT": str(r.output),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(lambda _: invoke(), range(2)))
    assert any(p.returncode == 0 for p in completed), [p.stderr for p in completed]
    for proc in completed:
        if proc.returncode:
            assert "running/interrupted" in proc.stderr
    assert len(r.calls()) == 1
    assert (
        json.loads((r.run_dir / "execution.json").read_text())["agent_invocations"] == 1
    )


def test_parent_stopping_during_review_invalidates_the_result(
    review_env: ReviewEnv,
) -> None:
    r = review_env
    value = json.loads(r.run(ok=False, extra={"REVIEW_STOP_PARENT": "1"}).stdout)
    assert value["result"]["status"] == "failed"
    assert all(p["coverage"] == "not_reviewed" for p in value["result"]["prs"])
    r.run(ok=False)
    assert len(r.calls()) == 1
