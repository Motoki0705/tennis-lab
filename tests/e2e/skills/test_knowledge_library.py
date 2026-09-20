"""CPU-only storage contracts: concurrency, integrity and summary review lifecycle."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/knowledge-control/scripts"


def command(base: Path, script: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT, env={**os.environ, "KNOWLEDGE_DIR": str(base)}, text=True, capture_output=True, check=check)


def meta(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text().split("---", 2)[1])  # type: ignore[no-any-return]


def create(base: Path, node_id: str, task: str = "new_topic", *extra: str) -> subprocess.CompletedProcess[str]:
    return command(base, "kg_new.py", "--type", "run", "--task", task, "--id", node_id, "--title", "試験", *extra)


def test_task_scoped_sequences_force_and_invalid_task(tmp_path: Path) -> None:
    create(tmp_path, "run-one")
    create(tmp_path, "run-two")
    create(tmp_path, "run-other", "synthetic_data_generation")
    create(tmp_path, "run-one", "new_topic", "--force")
    assert len(list((tmp_path / "nodes").rglob("*.md"))) == 3
    assert (tmp_path / "nodes/new_topic/000002-run-two.md").is_file()
    assert (tmp_path / "nodes/synthetic_data_generation/000001-run-other.md").is_file()
    with pytest.raises(subprocess.CalledProcessError):
        create(tmp_path, "run-one", "different_task", "--force")
    with pytest.raises(subprocess.CalledProcessError):
        create(tmp_path, "run-bad", "../escape")
    assert not (tmp_path / "escape").exists()


def test_concurrent_registration_allocates_unique_numbers(tmp_path: Path) -> None:
    procs = [subprocess.Popen([sys.executable, str(SCRIPTS / "kg_new.py"), "--type", "run", "--task", "plcs", "--id", f"run-{i}", "--title", "並列登録"], cwd=ROOT, env={**os.environ, "KNOWLEDGE_DIR": str(tmp_path)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(8)]
    for proc in procs:
        _, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0, stderr.decode()
    numbers = [meta(path)["sequence"] for path in (tmp_path / "nodes/plcs").glob("*.md")]
    assert sorted(numbers) == list(range(1, 9))


def test_paper_references_checksum_and_summary_staleness(tmp_path: Path) -> None:
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF")
    command(tmp_path, "kg_papers.py", "--id", "paper-2024-fixture", "--title", "Paper", "--authors", "Author", "--tasks", "new_topic", "--source", "https://example.org/paper/v1", "--license", "https://creativecommons.org/licenses/by/4.0/", "--pdf", str(pdf))
    create(tmp_path, "run-one", "new_topic", "--papers", "paper-2024-fixture")
    (tmp_path / "summary.md").write_text("# Summary\n試験の限界を確認した。\n")
    stale = command(tmp_path, "kg_validate.py", "--check-summary", check=False)
    assert stale.returncode == 1 and "unreviewed" in stale.stdout
    command(tmp_path, "kg_summary.py", "--mark-reviewed")
    command(tmp_path, "kg_validate.py", "--check-summary")
    node = next((tmp_path / "nodes").rglob("*.md"))
    node.write_text(node.read_text() + "\n追加の考察\n")
    assert command(tmp_path, "kg_summary.py", check=False).returncode == 1
    paper = tmp_path / "Papers/paper-2024-fixture/paper.pdf"
    paper.write_bytes(b"%PDF-modified")
    result = command(tmp_path, "kg_validate.py", check=False)
    assert result.returncode == 1 and "sha256 mismatch" in result.stdout
    node.write_text(node.read_text().replace("paper-2024-fixture", "paper-2024-missing"))
    assert "unknown paper" in command(tmp_path, "kg_validate.py", check=False).stdout


def test_invalid_schema_reports_errors_instead_of_crashing(tmp_path: Path) -> None:
    create(tmp_path, "run-one")
    path = next((tmp_path / "nodes").rglob("*.md"))
    path.write_text(path.read_text().replace("sequence: 1", "sequence: wrong").replace("parents: []", "parents: 3"))
    result = command(tmp_path, "kg_validate.py", check=False)
    assert result.returncode == 1
    assert "sequence must be" in result.stdout and "parents must be" in result.stdout
    assert "Traceback" not in result.stderr


def test_duplicate_sequence_and_broken_edges_are_rejected(tmp_path: Path) -> None:
    create(tmp_path, "run-one")
    create(tmp_path, "run-two", "new_topic", "--parents", "run-missing")
    path = tmp_path / "nodes/new_topic/000002-run-two.md"
    path.write_text(path.read_text().replace("sequence: 2", "sequence: 1"))
    result = command(tmp_path, "kg_validate.py", check=False)
    assert "duplicate task sequence" in result.stdout
    assert "parent 'run-missing' does not exist" in result.stdout


def test_log_import_respects_explicit_date_and_task(tmp_path: Path) -> None:
    job = tmp_path / "done/20260920_fixture.job"
    job.parent.mkdir()
    job.write_text("# name: fixture\npython -m src.tasks.blcs.scripts.train model=test loss=baseline\n")
    log = tmp_path / "fixture.log"
    log.write_text("| test/position_error_m | 0.25 |\n")
    command(tmp_path, "kg_from_run.py", "--task", "blcs", "--job", str(job), "--log", str(log), "--date", "2020-01-01", "--write")
    node = next((tmp_path / "nodes").rglob("*.md"))
    assert meta(node)["date"] == "2020-01-01"
    assert meta(node)["metrics"] == {"position_error_m": .25}


def test_migration_preserves_metadata_and_rebases_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    migration = importlib.import_module("kg_migrate")
    base = tmp_path / "knowledge/nodes"
    base.mkdir(parents=True)
    for number, day in ((1, "2024-01-02"), (2, "2024-01-01")):
        (base / f"run-plcs-{number}.md").write_text(f"---\nid: run-plcs-{number}\ntype: run\ntitle: fixture\ndate: '{day}'\ntags: [plcs]\nmetrics: {{value: {number}}}\n---\n[other](run-plcs-{3-number}.md)\n")
    (tmp_path / "knowledge/summary.md").write_text("[baseline](nodes/run-plcs-2.md)")
    bundle = tmp_path / "knowledge/runs/run-plcs-1"
    bundle.mkdir(parents=True)
    snapshot = bundle / "snapshot.md"
    snapshot.write_text("historical nodes/run-plcs-1.md")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    monkeypatch.setattr(migration, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(migration, "nodes_dir", lambda: base)
    plan = migration.migrate(False, {})
    assert len(plan) == 2 and (base / "run-plcs-1.md").exists()
    migration.migrate(True, {})
    first = base / "plcs/000001-run-plcs-2.md"
    second = base / "plcs/000002-run-plcs-1.md"
    assert meta(first)["metrics"] == {"value": 2}
    assert "000002-run-plcs-1.md" in first.read_text()
    assert "000001-run-plcs-2.md" in second.read_text()
    assert "nodes/plcs/000001-run-plcs-2.md" in (tmp_path / "knowledge/summary.md").read_text()
    assert snapshot.read_text() == "historical nodes/run-plcs-1.md"
    assert migration.migrate(True, {}) == {}
