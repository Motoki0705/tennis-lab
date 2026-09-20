"""Execute the SKILL's documented commands, then validate their completed artifacts.

No model API, GPU, downloads, or live queue: fixtures provide the evidence and the
human/agent-authored prose. This tests operational consistency, not research quality.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / '.agents/skills/knowledge-control'


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


@pytest.mark.parametrize(('route', 'status'), [('queue', 'done'), ('log', 'failed'), ('manual', 'failed')])
def test_documented_workflow_passes_the_ci_gate(tmp_path: Path, route: str, status: str) -> None:
    library = tmp_path / 'knowledge'
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
    pdf = tmp_path / 'source paper.pdf'
    pdf.write_bytes(b'%PDF-1.4\nfixture, not a research paper\n%%EOF')
    env = {**os.environ, 'KNOWLEDGE_DIR': str(library), 'TRAINING_QUEUE_DIR': str(queue),
           'TASK': 'novel_research_topic', 'PROVIDER': 'codex', 'STATUS': status,
           'JOB': job_name, 'JOB_FILE': str(job), 'LOG_FILE': str(log),
           'RUN_ID': f'run-{job_name.replace("_", "-")}', 'GROUP_ID': f'group-{route}',
           'TITLE': '隔離fixtureによる登録確認', 'GROUP_TITLE': '登録検証のまとめ',
           'PAPER_ID': 'paper-2024-fixture', 'PAPER_TITLE': 'Disposable Fixture', 'AUTHOR': 'Fixture Author',
           'SOURCE_URL': 'https://example.org/fixture/v1',
           'LICENSE_URL': 'https://creativecommons.org/licenses/by/4.0/', 'PDF_PATH': str(pdf)}
    commands = examples()

    def run_example(name: str) -> None:
        subprocess.run(['bash', '-euc', commands['setup'] + '\n' + commands[name]], cwd=ROOT,
                       env=env, check=True, text=True, capture_output=True)

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
    summary = library / 'summary.md'
    summary.write_text('# Summary\n\n登録手順を確認した。未測定の結果でbaselineを変更しない。次は入力を確保する。\n')
    run_example('review')  # Exact mark-reviewed + --check-summary commands from SKILL.
    # A second check demonstrates the gate is read-only and stable after review.
    before = summary.read_bytes()
    subprocess.run([sys.executable, str(SKILL / 'scripts/kg_validate.py'), '--check-summary'],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert summary.read_bytes() == before
    note.write_text(note.read_text() + '\n後から追加した未レビューの限界。\n')
    result = subprocess.run([sys.executable, str(SKILL / 'scripts/kg_validate.py'), '--check-summary'],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 1 and 'unreviewed' in result.stdout
