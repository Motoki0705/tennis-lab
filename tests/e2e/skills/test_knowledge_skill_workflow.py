"""Execute the SKILL's documented commands, then validate their completed artifacts.

No model API, GPU, Drive, downloads, or live queue: fixtures provide the evidence and
the human/agent-authored prose. The fixture library lives in its own Git checkout so
the documented base-ref review runs exactly as it does on a branch. This tests
operational consistency, not research quality.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / '.agents/skills/knowledge-control'
SOURCE_COMMIT = 'a' * 40


def examples() -> dict[str, str]:
    return dict(re.findall(r'<!-- example:([a-z]+) -->\s*```bash\n(.*?)```',
                          (SKILL / 'SKILL.md').read_text() + '\n' + (SKILL / 'references/record.md').read_text(), re.DOTALL))


def metadata(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text().split('---', 2)[1])  # type: ignore[no-any-return]


def test_skill_metadata_and_local_references_are_discoverable() -> None:
    frontmatter = metadata(SKILL / 'SKILL.md')
    assert frontmatter['name'] == SKILL.name
    assert isinstance(frontmatter['description'], str) and frontmatter['description'].strip()
    for path in (SKILL / 'SKILL.md', SKILL / 'references/record.md'):
        for link in re.findall(r'\[[^]]+\]\(([^)]+)\)', path.read_text()):
            if '://' not in link:
                assert (path.parent / link.split('#')[0]).exists(), (path, link)
    assert {'setup', 'query', 'queue', 'colab', 'log', 'manual', 'group', 'paper', 'review'} <= examples().keys()


def seed_registry(base: Path, *tasks: str) -> None:
    """Write the registry an isolated library needs: tasks, relation types, no deployments."""
    registry = base / 'registry'
    registry.mkdir(parents=True, exist_ok=True)
    entries = {task: {'status': 'active', 'note': 'isolated test topic'} for task in tasks}
    (registry / 'tasks.yaml').write_text(yaml.safe_dump({'schema_version': 1, 'tasks': entries}, allow_unicode=True))
    shutil.copyfile(ROOT / 'knowledge/registry/relations.yaml', registry / 'relations.yaml')
    (registry / 'deployments.yaml').write_text('schema_version: 1\nconfigs: []\ndeployments: []\n')


def git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.org', *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def colab_record(directory: Path, job: str) -> None:
    """The files the tennis-colab worker publishes for one completed job."""
    directory.mkdir(parents=True)
    request = {'schema_version': 1, 'job_id': job, 'argv': ['python', '-m', 'fixture', 'model=baseline'],
               'cwd': '/content/tennis-lab', 'persist': [], 'runner_outputs': ['outputs/fixture/train/colab'],
               'execution_backend': 'notebook', 'created_at': '2020-01-03T04:05:06Z', 'source_commit': SOURCE_COMMIT, 'helpers': {}}
    (directory / 'request.json').write_text(json.dumps(request))
    (directory / 'status.json').write_text(json.dumps({'schema_version': 1, 'job_id': job, 'status': 'completed',
                                                       'returncode': 0, 'drive_saved_at': '2020-01-03T05:00:00Z'}))
    untracked = base64.b64encode(b'print("fixture")\n').decode()
    (directory / 'source_changes.json').write_text(json.dumps({'commit': SOURCE_COMMIT, 'patch': '', 'untracked': {'scripts/colab_fixture.py': untracked},
                                                               'excluded': [], 'storage_excluded': [], 'exclusion_reasons': {}}))
    (directory / 'stdout.log').write_text('| test/position_error_m | 0.25 |\n')


@pytest.mark.parametrize(('route', 'status'), [('queue', 'done'), ('log', 'failed'), ('manual', 'failed'), ('colab', 'done')])
def test_documented_workflow_passes_the_ci_gate(tmp_path: Path, route: str, status: str) -> None:
    library = tmp_path / 'knowledge'
    seed_registry(library, 'novel_research_topic')
    summary = library / 'summary.md'
    summary.write_text('# Summary\n\n新しい研究トピックは準備中で、判断はまだない。\n')
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'base library')
    base_ref = git(tmp_path, 'rev-parse', 'HEAD')

    queue = tmp_path / 'shared queue'
    job_name = f'{route}_fixture'
    job_id = f'20260920_{job_name}'
    job = queue / ('done' if status == 'done' else 'failed') / f'{job_id}.job'
    job.parent.mkdir(parents=True)
    job.write_text(f'# name: {job_name}\npython -m fixture model=baseline loss=mse data=fixture\n')
    log = queue / 'logs' / f'{job_id}.log'
    log.parent.mkdir()
    log.write_text('| test/position_error_m | 0.25 |\n' if status == 'done' else 'Failed before measurement: missing input\n')
    repro = queue / 'repro' / job_id
    (repro / 'predictions').mkdir(parents=True)
    captured = {'name': job_name, 'command': 'python -m fixture model=baseline', 'captured_at': '2020-01-02T01:00:00Z'}
    (repro / 'run.json').write_text(json.dumps(captured))
    (repro / 'repro.sh').write_text('python -m fixture\n')
    (repro / 'predictions/metrics.json').write_text('{"position_error_m": 0.25}')
    drive_record = 'operations/colab/fixture-session/job-fixture'
    colab_record(tmp_path / 'record', 'job-fixture')
    env = {**os.environ, 'KNOWLEDGE_DIR': str(library), 'TRAINING_QUEUE_DIR': str(queue),
           'TASK': 'novel_research_topic', 'PROVIDER': 'codex', 'STATUS': status,
           'JOB': job_name, 'JOB_FILE': str(job), 'LOG_FILE': str(log),
           'RECORD_DIR': str(tmp_path / 'record'), 'DRIVE_RECORD': drive_record, 'BASE_REF': base_ref,
           'RUN_ID': f'run-{job_name.replace("_", "-")}', 'GROUP_ID': f'group-{route}',
           'TITLE': '隔離fixtureによる登録確認', 'GROUP_TITLE': '登録検証のまとめ',
           'PAPER_ID': 'paper-2024-fixture', 'PAPER_TITLE': 'Disposable Fixture', 'AUTHOR': 'Fixture Author',
           'SOURCE_URL': 'https://example.org/fixture/v1', 'PAPER_SUMMARY': 'Fictional research contribution',
           'PAPER_REASON': 'Does this fixture preserve experiment-to-paper references?'}
    commands = examples()

    def run_example(name: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', '-euc', commands['setup'] + '\n' + commands[name]], cwd=ROOT,
                              env=env, check=check, text=True, capture_output=True)

    run_example('paper')
    run_example(route)
    run_example('group')
    node = next((library / 'nodes').rglob('*-run-*.md'))
    values = metadata(node)
    assert values['status'] == status and values['provider'] == 'codex'
    if route == 'queue':
        assert values['date'] == '2020-01-02'
        assert values['metrics']['position_error_m'] == 0.25
        assert (library / f'runs/{env["RUN_ID"]}/run.json').read_text() == json.dumps(captured)
    elif route == 'colab':
        assert values['date'] == '2020-01-03' and values['metrics'] == {'position_error_m': 0.25}
        assert values['artifacts']['drive_record'] == drive_record
        assert values['artifacts']['drive_outputs'] == ['outputs/fixture/train/colab']
        bundle = library / f'runs/{env["RUN_ID"]}'
        assert f'git checkout {SOURCE_COMMIT}' in (bundle / 'repro.sh').read_text()
        assert '+++ b/scripts/colab_fixture.py' in (bundle / 'uncommitted.patch').read_text()
    else:
        assert 'date' not in values  # Unknown experiment date must stay unknown.
        assert values['metrics'] == {}
    # Finish the drafts following the prose guidance (no required heading template).
    values['papers'] = [env['PAPER_ID']]
    node.write_text('---\n' + yaml.safe_dump(values, allow_unicode=True) +
                    '---\n\n登録契約を検証した。背景はpaper-2024-fixture。合成fixtureであり実精度は評価していない。\n'
                    '未測定値やTensorBoardは存在しない。次は実際の入力で比較する。\n')
    group = next((library / 'nodes').rglob('*-group-*.md'))
    group.write_text(group.read_text() + '\nメンバーの保存形式を確認。モデル精度の改善は主張しない。\n')
    note = library / f'Papers/{env["PAPER_ID"]}/paper.md'
    note.write_text(note.read_text() + '\n試験専用の自作fixture。外部研究の再現ではなく背景リンクの検証に使う。\n')

    # Added nodes must declare their summary impact before the review passes.
    result = run_example('review', check=False)
    assert result.returncode == 1 and 'declare summary_impact' in result.stdout
    for path, impact in ((node, 'updated'), (group, 'none')):
        text = path.read_text()
        path.write_text(text.replace('\ntitle:', f'\nsummary_impact: {impact}\ntitle:', 1))
    result = run_example('review', check=False)
    assert result.returncode == 1 and 'summary.md is unchanged' in result.stdout
    summary.write_text(f'# Summary\n\n## novel_research_topic\n\n登録手順を確認した（[根拠](nodes/novel_research_topic/{node.name})）。'
                       '未測定の結果でbaselineを決めない。次は入力を確保する。\n')
    run_example('review')  # The exact base-ref review command from the SKILL.
    # The gate is read-only and stable after review.
    before = summary.read_bytes()
    subprocess.run([sys.executable, str(SKILL / 'scripts/kg_validate.py'), '--check-summary', '--base-ref', base_ref],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert summary.read_bytes() == before
    # The documented queries find the new evidence.
    queried = run_example('query').stdout
    assert env['RUN_ID'] in queried
    if status == 'done':
        assert 'position_error_m' in queried
