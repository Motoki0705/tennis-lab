"""CPU-only storage contracts: concurrency, integrity and summary review lifecycle."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from datetime import date
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
    result = command(base, "kg_new.py", "--type", "run", "--task", task, "--id", node_id, "--title", "試験", *extra)
    path = next((base / "nodes").rglob(f"*-{node_id}.md"))
    path.write_text(path.read_text() + "\n合成入力の登録動作を確認した。精度の評価は行っていない。\n")
    return result


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
    note = tmp_path / "Papers/paper-2024-fixture/paper.md"
    note.write_text(note.read_text() + "\n検証用のPDF fixture。実在研究の主張は含まない。\n")
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


def test_highest_sequence_is_never_reused_after_deletion(tmp_path: Path) -> None:
    create(tmp_path, "run-one")
    create(tmp_path, "run-two")
    directory = tmp_path / "nodes/new_topic"
    (directory / "000002-run-two.md").unlink()
    create(tmp_path, "run-three")
    assert (directory / "000003-run-three.md").is_file()
    for node in directory.glob("*.md"):
        node.unlink()
    create(tmp_path, "run-four")
    assert (directory / "000004-run-four.md").is_file()
    (directory / ".sequence").write_text("2\n")
    result = command(tmp_path, "kg_validate.py", check=False)
    assert "below existing node" in result.stdout
    with pytest.raises(subprocess.CalledProcessError):
        create(tmp_path, "run-five")
    (directory / ".sequence").unlink()
    result = command(tmp_path, "kg_validate.py", check=False)
    assert "missing .sequence" in result.stdout


def update_meta(path: Path, **changes: Any) -> None:
    _, frontmatter, body = path.read_text().split('---', 2)
    values = yaml.safe_load(frontmatter)
    values.update(changes)
    path.write_text('---\n' + yaml.safe_dump(values, allow_unicode=True) + '---' + body)


@pytest.mark.parametrize(('changes', 'message'), [
    ({'id': 'run--one'}, 'invalid id'),
    ({'title': ['not', 'text']}, 'title must'),
    ({'issue': True}, 'issue must'),
    ({'issue': [1, '2']}, 'issue must'),
    ({'provider': 'typo'}, 'provider must'),
    ({'status': {}}, 'status must'),
    ({'config': []}, 'config must'),
    ({'metrics': {'loss': float('nan')}}, 'metrics must'),
    ({'artifacts': 'output'}, 'artifacts must'),
    ({'artifacts': {'run_dir': '.'}}, 'artifacts.run_dir'),
    ({'artifacts': {'run_dir': False}}, 'artifacts.run_dir'),
    ({'date': '20260101'}, 'date must'),
    ({'recorded_at': '2026-02-30'}, 'recorded_at must'),
    ({'date_source': 'guessed'}, 'date_source must'),
    ({'parents': ['run-one']}, 'self parent'),
    ({'tags': ['one', 'one']}, 'duplicate tags'),
    ({'relations': [{'to': 'run-one', 'rel': []}]}, 'relation requires'),
    ({'members': ['run-one']}, 'members are only allowed'),
])
def test_schema_contract_rejects_invalid_values(tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    create(tmp_path, 'run-one')
    path = next((tmp_path / 'nodes').rglob('*.md'))
    update_meta(path, **changes)
    result = command(tmp_path, 'kg_validate.py', check=False)
    assert result.returncode == 1 and message in result.stdout
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('frontmatter', [
    'id: run-one\nid: run-two\n',
    'id: [unterminated\n',
    'config: &config {model: test}\nmetrics: *config\n',
    'metrics: {loss: 1, loss: 2}\n',
])
def test_bad_yaml_is_an_actionable_error(tmp_path: Path, frontmatter: str) -> None:
    create(tmp_path, 'run-one')
    path = next((tmp_path / 'nodes').rglob('*.md'))
    path.write_text('---\n' + frontmatter + '---\nFindings\n')
    result = command(tmp_path, 'kg_validate.py', check=False)
    assert result.returncode == 1 and 'invalid YAML' in result.stdout
    assert 'Traceback' not in result.stderr


def test_cycles_rejected_but_mutual_comparisons_allowed(tmp_path: Path) -> None:
    create(tmp_path, 'run-one')
    create(tmp_path, 'run-two')
    first = tmp_path / 'nodes/new_topic/000001-run-one.md'
    second = tmp_path / 'nodes/new_topic/000002-run-two.md'
    update_meta(first, relations=[{'to': 'run-two', 'rel': 'compares'}])
    update_meta(second, parents=['run-one'], relations=[{'to': 'run-one', 'rel': 'compares'}])
    command(tmp_path, 'kg_validate.py')
    update_meta(first, parents=['run-two'])
    result = command(tmp_path, 'kg_validate.py', check=False)
    assert 'parents: cycle detected' in result.stdout
    update_meta(first, parents=[])
    for name in ('a', 'b'):
        command(tmp_path, 'kg_new.py', '--task', 'new_topic', '--type', 'group', '--id', f'group-{name}', '--title', '比較', '--members', 'run-one')
    groups = sorted((tmp_path / 'nodes').rglob('*-group-*.md'))
    for i, group in enumerate(groups):
        group.write_text(group.read_text() + '\n比較条件は同一。精度の主張はしない。\n')
        update_meta(group, members=[f'group-{("b", "a")[i]}'])
    assert 'members: cycle detected' in command(tmp_path, 'kg_validate.py', check=False).stdout


def test_unfinished_scaffold_and_summary_edits_require_completion(tmp_path: Path) -> None:
    command(tmp_path, 'kg_new.py', '--type', 'run', '--task', 'new_topic', '--id', 'run-draft', '--title', '下書き')
    node = next((tmp_path / 'nodes').rglob('*.md'))
    assert 'date' not in meta(node) and 'provider' not in meta(node)
    assert 'unfinished' in command(tmp_path, 'kg_validate.py', check=False).stdout
    node.write_text(node.read_text() + '\nデータ不足で測定できなかった。曲線も未取得。次に入力を確認する。\n')
    update_meta(node, status='failed', metrics={}, issue=None)
    summary = tmp_path / 'summary.md'
    summary.write_text('# Summary\n')
    assert command(tmp_path, 'kg_summary.py', '--mark-reviewed', check=False).returncode == 1
    summary.write_text('# Summary\n実行失敗のためbaselineは維持する。\n')
    command(tmp_path, 'kg_summary.py', '--mark-reviewed')
    command(tmp_path, 'kg_validate.py', '--check-summary')
    summary.write_text(summary.read_text() + '\n追加の解釈は未レビュー。\n')
    assert command(tmp_path, 'kg_validate.py', '--check-summary', check=False).returncode == 1
    command(tmp_path, 'kg_summary.py', '--mark-reviewed')
    command(tmp_path, 'kg_validate.py', '--check-summary')
    summary.write_text(summary.read_text().replace(' on ', ' on 2026-02-30 --><!-- on ', 1))
    assert command(tmp_path, 'kg_validate.py', '--check-summary', check=False).returncode == 1


def test_missing_library_or_orphan_papers_are_not_silently_ignored(tmp_path: Path) -> None:
    result = command(tmp_path, 'kg_validate.py', check=False)
    assert result.returncode == 1 and 'missing nodes directory' in result.stdout
    create(tmp_path, 'run-one')
    orphan = tmp_path / 'Papers/paper-2024-orphan'
    orphan.mkdir(parents=True)
    (orphan / 'paper.pdf').write_bytes(b'%PDF-fixture')
    assert 'missing paper.md' in command(tmp_path, 'kg_validate.py', check=False).stdout
    (orphan / 'paper.md').write_text('---\nid: [bad\n---\nNote\n')
    result = command(tmp_path, 'kg_validate.py', check=False)
    assert result.returncode == 1 and 'invalid YAML' in result.stdout
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('quoted_base', [True, False])
def test_base_comparison_preserves_identity_and_deleted_allocations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quoted_base: bool) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    history = importlib.import_module('kg_history')
    base = tmp_path / 'knowledge'
    create(base, 'run-one')
    create(base, 'run-two')
    first = base / 'nodes/new_topic/000001-run-one.md'
    update_meta(first, recorded_at='2026-06-26' if quoted_base else date(2026, 6, 26))
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.org', 'commit', '-qm', 'base'], check=True)
    monkeypatch.setattr(history, 'repo_root', lambda: tmp_path)
    monkeypatch.setattr(history, 'nodes_dir', lambda: base / 'nodes')
    lib = importlib.import_module('kg_lib')
    directory = base / 'nodes/new_topic'
    assert history.validate_history(lib.load_nodes(base / 'nodes'), 'HEAD') == []
    update_meta(first, recorded_at=date(2026, 6, 26) if quoted_base else '2026-06-26')
    assert history.validate_history(lib.load_nodes(base / 'nodes'), 'HEAD') == []
    create(base, 'run-three')
    assert history.validate_history(lib.load_nodes(base / 'nodes'), 'HEAD') == []
    first = directory / '000001-run-one.md'
    update_meta(first, sequence=4, recorded_at='2000-01-01')
    errors = history.validate_history(lib.load_nodes(base / 'nodes'), 'HEAD')
    assert any('sequence is immutable' in e for e in errors)
    assert any('recorded_at is immutable' in e for e in errors)
    first.unlink()
    (directory / '000002-run-two.md').unlink()
    (directory / '.sequence').write_text('1\n')
    errors = history.validate_history(lib.load_nodes(base / 'nodes'), 'HEAD')
    assert any('retain at least base allocation 2' in e for e in errors)
    update_meta(directory / '000003-run-three.md', sequence=2)
    errors = history.validate_history(lib.load_nodes(base / 'nodes'), 'HEAD')
    assert any('new sequence must exceed base allocation 2' in e for e in errors)
    with pytest.raises(ValueError, match='cannot read history base'):
        history.validate_history([], 'unknown-ref')
