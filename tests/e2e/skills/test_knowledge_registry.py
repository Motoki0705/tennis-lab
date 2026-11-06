"""CPU contracts of the registries, change rules, Colab import, papers and queries.

Each test builds a disposable library (``KNOWLEDGE_DIR``); change rules use a
throwaway Git checkout as the merge base. No GPU, Drive, network or live queue.
"""
from __future__ import annotations

import base64
import fnmatch
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/knowledge-control/scripts"
PROSE = "\n合成入力で登録経路を確認した。精度は評価していない。\n"


def run(base: Path, script: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT,
                          env={**os.environ, "KNOWLEDGE_DIR": str(base)}, text=True, capture_output=True, check=check)


def seed_registry(base: Path, tasks: dict[str, dict[str, Any]], deployments: dict[str, Any] | None = None) -> None:
    registry = base / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "tasks.yaml").write_text(yaml.safe_dump({"schema_version": 1, "tasks": tasks}, allow_unicode=True))
    shutil.copyfile(ROOT / "knowledge/registry/relations.yaml", registry / "relations.yaml")
    document = deployments or {"schema_version": 1, "configs": [], "deployments": []}
    (registry / "deployments.yaml").write_text(yaml.safe_dump(document, allow_unicode=True))


def active(*names: str) -> dict[str, dict[str, Any]]:
    return {name: {"status": "active", "note": "isolated test topic"} for name in names}


def node(base: Path, node_id: str, task: str = "topic", *extra: str) -> Path:
    run(base, "kg_new.py", "--type", "run", "--task", task, "--id", node_id, "--title", "試験", *extra)
    path = next((base / "nodes").rglob(f"*-{node_id}.md"))
    path.write_text(path.read_text() + PROSE)
    return path


def update(path: Path, **changes: Any) -> None:
    _, frontmatter, body = path.read_text().split("---", 2)
    values = yaml.safe_load(frontmatter)
    values.update(changes)
    values = {k: v for k, v in values.items() if v is not None}
    path.write_text("---\n" + yaml.safe_dump(values, allow_unicode=True) + "---" + body)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.org", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def commit(root: Path) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


def test_tasks_and_relations_are_controlled_vocabularies(tmp_path: Path) -> None:
    seed_registry(tmp_path, active("topic"))
    first = node(tmp_path, "run-one")
    second = node(tmp_path, "run-two")
    rejected = run(tmp_path, "kg_new.py", "--type", "run", "--task", "unregistered", "--id", "run-x", "--title", "試験", check=False)
    assert rejected.returncode == 2 and "not registered" in rejected.stderr
    assert not (tmp_path / "nodes/unregistered").exists()
    run(tmp_path, "kg_validate.py")
    stray = tmp_path / "nodes/other_topic"
    stray.mkdir()
    (stray / ".sequence").write_text("1\n")
    assert "nodes/other_topic: task is not registered" in run(tmp_path, "kg_validate.py", check=False).stdout
    shutil.rmtree(stray)
    for relation, message in (({"to": "run-two", "rel": "superseded_by"}, "unknown rel 'superseded_by'"),
                              ({"to": "run-two", "rel": "related"}, "requires a note"),
                              ({"to": "run-two", "rel": "compares", "weight": 1}, "relations allow only"),
                              ({"to": "run-two", "rel": "compares", "note": " "}, "note must be nonempty")):
        update(first, relations=[relation])
        result = run(tmp_path, "kg_validate.py", check=False)
        assert result.returncode == 1 and message in result.stdout, relation
    update(first, relations=[{"to": "run-two", "rel": "related", "note": "同じ入力を使った"}])
    update(second, relations=[{"to": "run-one", "rel": "compares"}])
    run(tmp_path, "kg_validate.py")
    (tmp_path / "registry/relations.yaml").unlink()
    assert "missing registry file" in run(tmp_path, "kg_validate.py", check=False).stdout


def test_retired_task_keeps_history_but_takes_no_new_nodes(tmp_path: Path) -> None:
    seed_registry(tmp_path, active("topic", "successor"))
    existing = node(tmp_path, "run-old")
    seed_registry(tmp_path, {"topic": {"status": "retired", "successor": "successor", "note": "廃止した方式"}, **active("successor")})
    result = run(tmp_path, "kg_new.py", "--type", "run", "--task", "topic", "--id", "run-new", "--title", "試験", check=False)
    assert result.returncode == 2 and "retired; record new evidence under 'successor'" in result.stderr
    run(tmp_path, "kg_new.py", "--type", "run", "--task", "topic", "--id", "run-old", "--title", "修正", "--force")
    existing.write_text(existing.read_text() + PROSE)
    run(tmp_path, "kg_validate.py")
    seed_registry(tmp_path, {"topic": {"status": "retired", "successor": "missing", "note": "廃止"}})
    assert "successor must name another registered task" in run(tmp_path, "kg_validate.py", check=False).stdout


def test_deployments_must_match_tracked_config_and_cite_evidence(tmp_path: Path) -> None:
    library = tmp_path / "knowledge"
    config = tmp_path / "configs/pipeline.yaml"
    config.parent.mkdir()
    config.write_text("court:\n  checkpoint: court/a.ckpt\n  checkpoint_strict: true\npeople:\n  pose_checkpoint: pose/b.pth\n")
    (tmp_path / "LICENSES.md").write_text("pose weights `b.pth` come from the upstream release.\n")
    seed_registry(library, active("topic"))
    node(library, "run-court")
    entries: list[dict[str, Any]] = [
        {"config": "configs/pipeline.yaml", "key": "court.checkpoint", "value": "court/a.ckpt", "task": "topic", "evidence": ["run-court"]},
        {"config": "configs/pipeline.yaml", "key": "people.pose_checkpoint", "value": "pose/b.pth", "task": "topic", "source": "LICENSES.md"},
    ]
    deployments: dict[str, Any] = {"schema_version": 1, "configs": ["configs/pipeline.yaml"], "deployments": entries}
    seed_registry(library, active("topic"), deployments)
    run(library, "kg_validate.py")
    config.write_text(config.read_text().replace("court/a.ckpt", "court/c.ckpt"))
    result = run(library, "kg_validate.py", check=False)
    assert "config selects 'court/c.ckpt' but the registry records 'court/a.ckpt'" in result.stdout
    config.write_text(config.read_text().replace("court/c.ckpt", "court/a.ckpt") + "ball:\n  checkpoint: ball/new.ckpt\n")
    assert "ball.checkpoint selects 'ball/new.ckpt' without a deployment entry" in run(library, "kg_validate.py", check=False).stdout
    config.write_text(config.read_text().replace("ball:\n  checkpoint: ball/new.ckpt\n", ""))
    cases: list[tuple[dict[str, Any], str]] = [
        ({"evidence": ["run-missing"]}, "evidence node 'run-missing' does not exist"),
        ({"evidence": [], "source": None}, "cite evidence nodes, or a source document"),
        ({"source": "README.md"}, "source must be an existing"),
        ({"task": "unknown"}, "task 'unknown' is not registered"),
        ({"papers": ["paper-2024-missing"]}, "papers must be a list of unique registered paper IDs"),
    ]
    for change, message in cases:
        entry = {k: v for k, v in {**entries[0], **change}.items() if v is not None}
        seed_registry(library, active("topic"), {**deployments, "deployments": [entry, entries[1]]})
        result = run(library, "kg_validate.py", check=False)
        assert result.returncode == 1 and message in result.stdout, change
    (tmp_path / "LICENSES.md").write_text("no weights listed\n")
    seed_registry(library, active("topic"), deployments)
    assert "does not mention b.pth" in run(library, "kg_validate.py", check=False).stdout


def test_change_rules_judge_only_what_the_branch_adds(tmp_path: Path) -> None:
    library = tmp_path / "knowledge"
    seed_registry(library, active("topic"))
    old = node(library, "run-old")
    update(old, status="running")
    summary = library / "summary.md"
    summary.write_text("# Summary\n\n## topic\n\n現行の判断はrun-oldの結果待ち。\n")
    git(tmp_path, "init", "-q")
    base = commit(tmp_path)
    checked = ("--check-summary", "--base-ref", base)

    no_base = run(library, "kg_validate.py")
    assert "no --base-ref" in no_base.stdout and "WARN" not in no_base.stdout  # Pre-existing nodes never warn.
    new = node(library, "run-new")
    result = run(library, "kg_validate.py", *checked, check=False)
    assert result.returncode == 1 and "000002-run-new.md: added or status-changed node must declare summary_impact" in result.stdout
    assert "WARN: 000002-run-new.md: new run node has no 'issue'" in result.stdout
    assert "run-old" not in result.stdout
    assert run(library, "kg_validate.py", "--base-ref", base).returncode == 0  # Review declarations belong to --check-summary.
    update(new, summary_impact="none", issue=1)
    run(library, "kg_validate.py", *checked)
    update(new, summary_impact="updated")
    assert "summary.md is unchanged relative to" in run(library, "kg_validate.py", *checked, check=False).stdout
    summary.write_text(summary.read_text() + "run-newで判断を更新した。\n")
    run(library, "kg_validate.py", *checked)
    update(old, status="done")
    assert "000001-run-old.md: added or status-changed node" in run(library, "kg_validate.py", *checked, check=False).stdout
    update(old, summary_impact="updated")
    run(library, "kg_validate.py", *checked)
    update(new, artifacts={"log": "/home/someone/projects/tennis-lab/.training_queue/logs/x.log"})
    assert "must not record a home-directory path" in run(library, "kg_validate.py", *checked, check=False).stdout
    update(new, artifacts={"log": ".training_queue/logs/x.log", "drive_record": "operations/colab/s/j"})
    run(library, "kg_validate.py", *checked)
    seed_registry(library, {"topic": {"status": "retired", "note": "廃止"}})
    assert "task 'topic' is retired" in run(library, "kg_validate.py", *checked, check=False).stdout
    seed_registry(library, active("topic"))
    summary.write_text("# Summary\n\n## topic\n\n現行の判断はrun-oldの結果待ち。\n")
    update(new, summary_impact="none")
    update(old, summary_impact="none")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/p.yaml").write_text("model:\n  checkpoint: m.ckpt\n")
    entry = {"config": "configs/p.yaml", "key": "model.checkpoint", "value": "m.ckpt", "task": "topic", "evidence": ["run-new"]}
    seed_registry(library, active("topic"), {"schema_version": 1, "configs": ["configs/p.yaml"], "deployments": [entry]})
    assert "registry/deployments.yaml changed" in run(library, "kg_validate.py", *checked, check=False).stdout
    summary.write_text(summary.read_text() + "標準設定のmodelはrun-newのcheckpointを使う。\n")
    run(library, "kg_validate.py", *checked)


def colab_record(directory: Path, commit_sha: str, **changes: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    request = {"schema_version": 1, "job_id": "job-1", "argv": ["python", "-m", "tool", "model=small"], "cwd": "/content/tennis-lab/sub",
               "persist": ["outputs/x"], "runner_outputs": [], "created_at": "2026-10-27T01:02:03Z", "source_commit": commit_sha, "helpers": {}}
    status = {"schema_version": 1, "job_id": "job-1", "status": "completed", "returncode": 0, "drive_saved_at": "2026-10-27T05:00:00Z"}
    source = {"commit": commit_sha, "patch": "", "untracked": {}, "excluded": [], "storage_excluded": [], "exclusion_reasons": {}}
    for name, payload in (("request", request), ("status", status), ("source", source)):
        payload.update(changes.get(name, {}))
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "status.json").write_text(json.dumps(status))
    (directory / "source_changes.json").write_text(json.dumps(source))
    (directory / "stdout.log").write_text(changes.get("log", "| test/err_px | 1.5 |\n"))
    return directory


def colab(library: Path, record: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(library, "kg_from_colab.py", str(record), "--drive-record", "operations/colab/s/job-1", "--task", "topic",
               "--id", "run-colab", "--title", "Colab試験", *extra, check=check)


def test_colab_bundle_reproduces_the_recorded_source(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    (source_repo / "sub").mkdir(parents=True)
    (source_repo / "sub/tracked.py").write_text("value = 1\n")
    git(source_repo, "init", "-q")
    sha = commit(source_repo)
    (source_repo / "sub/tracked.py").write_text("value = 2\n")
    tracked = subprocess.run(["git", "-C", str(source_repo), "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
    untracked = {"sub/new.py": b"print('a')\nprint('b')", "sub/empty.txt": b"", "notes.md": "日本語\n".encode()}
    record = colab_record(tmp_path / "record", sha, source={"patch": base64.b64encode(tracked).decode(),
                                                            "untracked": {k: base64.b64encode(v).decode() for k, v in untracked.items()}})
    library = tmp_path / "knowledge"
    seed_registry(library, active("topic"))
    printed = colab(library, record, "--provider", "codex").stdout
    assert "outputs stay on Drive: outputs/x" in printed
    bundle = library / "runs/run-colab"
    script = (bundle / "repro.sh").read_text()
    assert f"git checkout {sha}" in script and "cd sub\n# --- original training command ---\npython -m tool model=small\n" in script
    values = yaml.safe_load(next((library / "nodes").rglob("*.md")).read_text().split("---", 2)[1])
    assert values["status"] == "done" and values["metrics"] == {"err_px": 1.5} and values["config"] == {"model": "small"}
    assert values["artifacts"]["drive_record"] == "operations/colab/s/job-1" and values["date"] == "2026-10-27"
    assert json.loads((bundle / "colab-source.json").read_text())["untracked"] == sorted(untracked)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(source_repo), str(clone)], check=True)
    git(clone, "checkout", "-q", sha)
    git(clone, "apply", str(bundle / "uncommitted.patch"))
    assert (clone / "sub/tracked.py").read_text() == "value = 2\n"
    assert {name: (clone / name).read_bytes() for name in untracked} == untracked
    run(library, "kg_repro_paths.py")


@pytest.mark.parametrize(("changes", "message"), [
    ({"status": {"status": "running"}}, "register only after a terminal status"),
    ({"status": {"job_id": "job-2"}}, "job IDs differ"),
    ({"status": {"drive_saved_at": None}}, "must record drive_saved_at"),
    ({"request": {"cwd": "/tmp/elsewhere"}}, "outside the VM checkout"),
    ({"request": {"schema_version": 2}}, "schema_version 1"),
    ({"source": {"untracked": {"x.bin": base64.b64encode(b"\xff\xfe").decode()}}}, "is not UTF-8 text"),
    ({"source": {"untracked": {"../escape.py": base64.b64encode(b"x").decode()}}}, "leaves the repository"),
    ({"log": "x" * (256 * 1024 + 1)}, "pass --skip-log"),
])
def test_colab_import_rejects_incomplete_records_before_writing(tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    record = colab_record(tmp_path / "record", "b" * 40, **changes)
    library = tmp_path / "knowledge"
    seed_registry(library, active("topic"))
    result = colab(library, record, check=False)
    assert result.returncode == 2 and message in result.stderr
    assert not (library / "runs").exists() and not (library / "nodes").exists()


def test_colab_import_keeps_oversized_log_on_drive_only_when_asked(tmp_path: Path) -> None:
    record = colab_record(tmp_path / "record", "b" * 40, log="| test/err_px | 2 |\n" + "x" * (256 * 1024))
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"err_px": 2.25}')
    library = tmp_path / "knowledge"
    seed_registry(library, active("topic"))
    colab(library, record, "--skip-log", "--metrics", str(metrics))
    assert not (library / "runs/run-colab/stdout.log").exists()
    assert json.loads((library / "runs/run-colab/metrics.json").read_text()) == {"err_px": 2.25}
    metrics.write_text('{"err_px": "high"}')
    assert "finite numbers" in colab(library, record, "--skip-log", "--metrics", str(metrics), "--force", check=False).stderr


def test_colab_attach_adds_the_bundle_without_touching_findings(tmp_path: Path) -> None:
    record = colab_record(tmp_path / "record", "c" * 40)
    library = tmp_path / "knowledge"
    seed_registry(library, active("topic"))
    path = node(library, "run-colab", "topic", "--status", "failed")
    update(path, metrics={"err_px": 9.0})
    before = path.read_text()
    attach = ("kg_from_colab.py", str(record), "--drive-record", "operations/colab/s/job-1", "--task", "topic", "--id", "run-colab", "--attach")

    assert "drop --title" in run(library, *attach, "--title", "x", check=False).stderr
    assert "exactly one existing run node" in run(library, *attach[:-2], "run-missing", "--attach", check=False).stderr
    assert path.read_text() == before

    run(library, *attach)
    values = yaml.safe_load(path.read_text().split("---", 2)[1])
    assert values["status"] == "failed" and values["metrics"] == {"err_px": 9.0}  # Findings stay as recorded.
    assert values["repro"]["commit"] == "c" * 40 and values["artifacts"]["drive_record"] == "operations/colab/s/job-1"
    assert path.read_text().split("---", 2)[2] == before.split("---", 2)[2]
    assert {p.name for p in (library / "runs/run-colab").iterdir()} == {
        "repro.sh", "colab-request.json", "colab-status.json", "colab-source.json", "stdout.log"}
    assert "already has" in run(library, *attach, check=False).stderr  # Never overwrites a bundle.

    other = colab_record(tmp_path / "other", "d" * 40)
    for bundled in (library / "runs/run-colab").iterdir():
        bundled.unlink()
    result = run(library, "kg_from_colab.py", str(other), *attach[2:], check=False)
    assert "already records different commit" in result.stderr


def test_metadata_only_papers_redistribute_nothing(tmp_path: Path) -> None:
    seed_registry(tmp_path, active("topic"))
    common = ("--title", "Paper", "--authors", "Author", "--tasks", "topic", "--source", "https://arxiv.org/abs/2401.00001v1")
    run(tmp_path, "kg_papers.py", "--id", "paper-2024-meta", *common)
    note = tmp_path / "Papers/paper-2024-meta/paper.md"
    assert sorted(p.name for p in note.parent.iterdir()) == ["paper.md"]
    assert {"pdf", "sha256", "license"}.isdisjoint(yaml.safe_load(note.read_text().split("---", 2)[1]))
    note.write_text(note.read_text() + "\n手法の背景としてだけ参照する。再現は主張しない。\n")
    node(tmp_path, "run-cite", "topic", "--papers", "paper-2024-meta")
    run(tmp_path, "kg_validate.py")
    (note.parent / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    assert "paper.pdf exists but the note does not record" in run(tmp_path, "kg_validate.py", check=False).stdout
    (note.parent / "paper.pdf").unlink()
    update(note, sha256="0" * 64)
    assert "sha256 describes a stored PDF" in run(tmp_path, "kg_validate.py", check=False).stdout
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF")
    without_license = run(tmp_path, "kg_papers.py", "--id", "paper-2024-pdf", *common, "--pdf", str(pdf), check=False)
    assert without_license.returncode == 2 and "--pdf requires --license" in without_license.stderr
    unknown = run(tmp_path, "kg_papers.py", "--id", "paper-2024-other", *common[:-4], "--tasks", "nowhere", "--source", common[-1], check=False)
    assert unknown.returncode == 2 and "not registered" in unknown.stderr


def test_query_cli_answers_from_the_whole_library(tmp_path: Path) -> None:
    seed_registry(tmp_path, active("topic", "other"))
    first = node(tmp_path, "run-first", "topic", "--date", "2026-10-01", "--issue", "7")
    second = node(tmp_path, "run-second", "topic", "--date", "2026-10-05", "--parents", "run-first")
    node(tmp_path, "run-elsewhere", "other", "--date", "2026-10-03")
    update(first, metrics={"err_px": 3.0}, tags=["baseline"])
    update(second, metrics={"err_px": 2.0}, relations=[{"to": "run-first", "rel": "supersedes"}])
    run(tmp_path, "kg_new.py", "--type", "group", "--task", "topic", "--id", "group-pair", "--title", "比較", "--members", "run-first", "run-second")
    group = next((tmp_path / "nodes").rglob("*-group-pair.md"))
    group.write_text(group.read_text() + PROSE)

    def query(*args: str) -> Any:
        return json.loads(run(tmp_path, "kg_query.py", *args, "--json").stdout)

    found = query("find", "--task", "topic", "--type", "run")
    assert [n["id"] for n in found["nodes"]] == ["run-second", "run-first"] and found["total"] == 2
    assert [n["id"] for n in query("find", "--since", "2026-10-02", "--until", "2026-10-04")["nodes"]] == ["run-elsewhere"]
    assert [n["id"] for n in query("find", "--tag", "baseline")["nodes"]] == ["run-first"]
    assert [n["id"] for n in query("find", "--issue", "7")["nodes"]] == ["run-first"]
    assert query("find", "--text", "登録経路", "--limit", "1")["total"] == 4
    shown = query("show", "run-first")
    assert shown["children"] == ["run-second"] and shown["groups"] == ["group-pair"]
    assert shown["relations_in"] == [{"from": "run-second", "rel": "supersedes"}]
    table = query("metrics", "--task", "topic", "--key", "err_px", "--oldest")
    assert [(r["id"], r["err_px"]) for r in table["rows"]] == [("run-first", 3.0), ("run-second", 2.0)]
    assert query("metrics", "--task", "topic") == {"err_px": 2}
    tasks = {t["task"]: t for t in query("tasks")}
    assert tasks["topic"]["nodes"] == 3 and tasks["other"]["latest"] == "2026-10-03"
    assert query("deployments") == []
    assert run(tmp_path, "kg_query.py", "show", "run-missing", check=False).returncode == 1
    assert run(tmp_path, "kg_query.py", "find", "--since", "2026/10/01", check=False).returncode == 2


def test_portable_paths_drop_machine_prefixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    lib = importlib.import_module("kg_lib")
    worktree, main, home = tmp_path / "main/.claude/worktrees/task", tmp_path / "main", tmp_path / "home"
    for directory in (worktree, home):
        directory.mkdir(parents=True)
    monkeypatch.setattr(lib, "repo_root", lambda: worktree)
    monkeypatch.setattr(lib, "main_checkout", lambda: main.resolve())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert lib.portable_path(worktree / "outputs/run") == "outputs/run"
    assert lib.portable_path(main / ".training_queue/logs/x.log") == ".training_queue/logs/x.log"
    assert lib.portable_path(home / "artifacts/run") == "~/artifacts/run"
    assert lib.portable_path(tmp_path / "elsewhere") == str((tmp_path / "elsewhere").resolve())


def test_ci_runs_when_a_tracked_config_or_source_changes() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/knowledge.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads the bare key `on` as True.
    registry = yaml.safe_load((ROOT / "knowledge/registry/deployments.yaml").read_text())
    watched = set(registry["configs"]) | {e["source"] for e in registry["deployments"] if "source" in e}
    for event in ("pull_request", "push"):
        patterns = triggers[event]["paths"]
        assert all(any(fnmatch.fnmatch(path, pattern) for pattern in patterns) for path in watched), (event, watched)
