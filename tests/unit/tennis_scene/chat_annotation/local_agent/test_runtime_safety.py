"""Temporary process groups and negative contracts; no model or real Codex run."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.chat_annotation.local_agent import (
    dispatcher,
    efficiency,
    worker_session,
)
from src.tennis_scene.chat_annotation.local_agent.__main__ import main
from src.tennis_scene.chat_annotation.local_agent.campaign_state import read_state
from src.tennis_scene.chat_annotation.local_agent.common import atomic_write_json
from src.tennis_scene.chat_annotation.local_agent.configuration import (
    CampaignConfig,
    ControlConfig,
)
from src.tennis_scene.chat_annotation.local_agent.path_contracts import (
    output_name,
    validate_command_paths,
)
from src.tennis_scene.chat_annotation.local_agent.processes import (
    capture_supervisor,
    owned_members,
    signal_owned,
)
from src.tennis_scene.chat_annotation.local_agent.worker_context import Ctx

from .conftest import CampaignFixture


@contextmanager
def process_tree(tmp_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    ready = tmp_path / 'ready.json'
    code = '''
import json, pathlib, signal, subprocess, sys, time
child_ready = pathlib.Path(sys.argv[1] + '.child')
child = subprocess.Popen([sys.executable, '-c',
    'import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text("ready"); time.sleep(120)', str(child_ready)])
while not child_ready.exists():
    time.sleep(.01)
pathlib.Path(sys.argv[1]).write_text(json.dumps({'child':child.pid}))
time.sleep(120)
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(ready)], cwd=tmp_path,
                               start_new_session=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owner = capture_supervisor(process.pid)
    try:
        until = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert ready.exists()
        yield process
    finally:
        signal_owned(owner, signal.SIGKILL)
        process.wait(timeout=5)


def test_pid_reuse_or_another_boot_cannot_signal_a_worker(tmp_path: Path) -> None:
    with process_tree(tmp_path) as process:
        owner = capture_supervisor(process.pid)
        with pytest.raises(RuntimeError, match='PID was reused'):
            signal_owned(replace(owner, start_ticks=owner.start_ticks + 1), signal.SIGTERM)
        assert process.poll() is None
        assert signal_owned(replace(owner, boot_id='previous-boot'), signal.SIGKILL) == 0
        assert process.poll() is None


def test_timeout_escalates_after_grace_and_waits_for_descendants(tmp_path: Path) -> None:
    with process_tree(tmp_path) as process:
        owner = capture_supervisor(process.pid)
        record: dict[str, Any] = {'dir': str(tmp_path), 'process_identity': owner.document()}
        task = {'attempts': [record]}
        # A stale/early receipt must not turn live descendants into a finished job.
        (tmp_path / 'exit_code').write_text('0\n')
        dispatcher.terminate_attempt(record, 'timeout', .05)
        process.wait(timeout=3)
        assert dispatcher.running_state(task) == 'alive'
        assert owned_members(owner)
        time.sleep(.06)
        dispatcher.terminate_attempt(record, 'timeout', .05)
        until = time.monotonic() + 3
        while owned_members(owner) and time.monotonic() < until:
            time.sleep(.01)
        assert not owned_members(owner)
        assert dispatcher.running_state(task) == 'finished'
        assert record['termination_reason'] == 'timeout' and 'kill_requested_at' in record


@pytest.mark.parametrize('exit_code,completed,failed,expected', [
    (0, True, False, 'review'), (1, True, False, 'continue'),
    (0, False, False, 'continue'), (0, True, True, 'continue'),
])
def test_success_requires_the_actual_exit_and_completion_contract(
    campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch,
    exit_code: int, completed: bool, failed: bool, expected: str,
) -> None:
    monkeypatch.delenv('CODEX_HOME', raising=False)
    task_id, directory = campaign.finished_task(campaign.annotation())
    (directory / 'exit_code').write_text(str(exit_code))
    events: list[dict[str, Any]] = [{'type': 'thread.started', 'thread_id': 'fixture-session'}]
    if completed:
        events.append({'type': 'turn.completed'})
    if failed:
        events.append({'type': 'turn.failed', 'error': {'message': 'fixture failure'}})
    (directory / 'events.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in events))
    state = read_state()
    dispatcher.finalize(state, task_id)
    assert state['tasks'][task_id]['status'] == expected
    assert 'exit_inferred' not in state['tasks'][task_id]['attempts'][-1]


def test_timeout_never_accepts_a_completed_annotation(campaign: CampaignFixture) -> None:
    task_id, directory = campaign.finished_task(campaign.annotation())
    (directory / 'exit_code').write_text('0')
    (directory / 'events.jsonl').write_text('{"type":"turn.completed"}\n')
    state = read_state()
    state['tasks'][task_id]['attempts'][-1]['termination_reason'] = 'timeout'
    dispatcher.finalize(state, task_id)
    assert state['tasks'][task_id]['status'] != 'review'
    assert state['tasks'][task_id]['attempts'][-1]['exit_code'] == 124


@pytest.mark.parametrize('error', [None, {}, {'message': ''}])
def test_failure_events_without_messages_never_become_success(
    campaign: CampaignFixture, error: dict[str, str] | None,
) -> None:
    task_id, directory = campaign.finished_task(campaign.annotation())
    (directory / 'exit_code').write_text('0')
    events = [{'type': 'turn.completed'}, {'type': 'turn.failed', 'error': error}]
    (directory / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    state = read_state()
    dispatcher.finalize(state, task_id)
    assert state['tasks'][task_id]['status'] != 'review'


def test_configured_codex_home_is_the_only_session_authority(
    campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, directory = campaign.finished_task(campaign.annotation())
    thread = 'fixture-configured-home'
    (directory / 'events.jsonl').write_text(json.dumps({'thread_id': thread}) + '\n')
    rollout = campaign.config.codex_home / 'sessions/2026/10/01' / f'rollout-{thread}.jsonl'
    rollout.parent.mkdir(parents=True)
    payload = {'type': 'token_count', 'info': {'last_token_usage': {'total_tokens': 25},
               'model_context_window': 100}, 'rate_limits': {'primary': {'used_percent': 17}}}
    rollout.write_text(json.dumps({'payload': payload}) + '\n')
    monkeypatch.setenv('CODEX_HOME', str(campaign.config.campaign_dir / 'wrong-home'))
    monkeypatch.delenv('CODEX_THREAD_ID', raising=False)
    assert efficiency.sessions_directory() == campaign.config.codex_home / 'sessions'
    assert dispatcher.rollout_rate_limits(thread) == ({'primary': {'used_percent': 17}}, .25)
    assert worker_session.context_fraction(Ctx(directory))['fraction'] == .25


@pytest.mark.parametrize('field', ['poll_seconds', 'timeout_seconds', 'termination_grace_seconds'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0])
def test_nonfinite_or_negative_timing_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ControlConfig.model_validate({field: value})


def test_missing_worker_executable_becomes_an_explicit_failure(
    campaign: CampaignFixture,
) -> None:
    from src.tennis_scene.chat_annotation.local_agent.configuration import (
        campaign_context,
    )
    from src.tennis_scene.chat_annotation.local_agent.launcher import supervise
    _, directory = campaign.finished_task(campaign.annotation())
    atomic_write_json(directory / 'launch.json', {'model': 'unused', 'effort': 'max', 'codex_config': []})
    config = campaign.config.model_copy(update={'codex_binary': '/fixture/definitely-missing-codex'})
    with campaign_context(config):
        assert supervise(directory) == 127
    assert json.loads((directory / 'exit.json').read_text())['exit_code'] == 127


def test_output_scope_rejects_escape_and_directory_symlinks(campaign: CampaignFixture, tmp_path: Path) -> None:
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (campaign.config.campaign_dir / 'escape').symlink_to(foreign, target_is_directory=True)
    for path in (foreign / 'result.json', campaign.config.campaign_dir / 'escape/result.json'):
        with pytest.raises(ValueError, match='outside its root'):
            validate_command_paths(output=path)
    assert not list(foreign.iterdir())


def test_campaign_cannot_alias_protected_annotation_storage(campaign: CampaignFixture) -> None:
    annotated = campaign.config.annotated
    annotated.mkdir(exist_ok=True)
    alias = campaign.config.campaign_dir / 'annotation-alias'
    alias.symlink_to(annotated, target_is_directory=True)
    value = campaign.config.model_dump()
    value['campaign_dir'] = alias
    with pytest.raises(ValueError, match='separate from annotation'):
        CampaignConfig.model_validate(value)


@pytest.mark.parametrize('name', ['../escape', '/tmp/escape', 'a/b', 'a\\b', '.', ''])
def test_image_names_are_not_alternate_paths(name: str) -> None:
    with pytest.raises(ValueError, match='filename component'):
        output_name(name)


def test_worker_cli_rejects_reading_edits_outside_the_campaign(campaign: CampaignFixture, tmp_path: Path) -> None:
    _, directory = campaign.finished_task(campaign.annotation())
    foreign = tmp_path / 'foreign-edits.json'
    foreign.write_text('{}')
    assert main(['--campaign', str(campaign.config.campaign_dir), 'worker', 'apply', str(directory), '--edits', str(foreign)]) == 1


def test_failed_idle_campaign_returns_nonzero(campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tennis_scene.chat_annotation.local_agent.campaign_state import locked_state
    task_id, _ = campaign.finished_task(campaign.annotation())
    with locked_state() as state:
        state['tasks'][task_id]['status'] = 'failed'
    monkeypatch.setattr(dispatcher, 'launch', lambda *args: pytest.fail('Failed tasks must not be relaunched'))
    assert main(['--campaign', str(campaign.config.campaign_dir), 'run', '--exit-when-idle']) == 1


def test_unsupported_process_backend_fails_before_spawning(campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported() -> None:
        raise OSError('pidfd unavailable')
    monkeypatch.setattr(dispatcher, 'require_process_backend', unsupported)
    monkeypatch.setattr(dispatcher.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not start an unmanageable worker'))
    state = read_state()
    task_id = next(iter(state['tasks']))
    with pytest.raises(OSError, match='pidfd'):
        dispatcher.launch(state, task_id, {})
    assert not list(campaign.config.tasks.iterdir())


def test_timeline_cache_detects_same_size_mtime_content_change(campaign: CampaignFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tennis_scene.chat_annotation.local_agent import common
    common.verified_timeline(campaign.video, campaign.manifest)
    before = campaign.video.stat()
    contents = bytearray(campaign.video.read_bytes())
    contents[-1] ^= 1
    campaign.video.write_bytes(contents)
    os.utime(campaign.video, ns=(before.st_atime_ns, before.st_mtime_ns))
    def recheck(*args: Any) -> Any:
        raise ValueError('changed video must be verified again')
    monkeypatch.setattr(common, 'check_clip', recheck)
    with pytest.raises(ValueError, match='verified again'):
        common.verified_timeline(campaign.video, campaign.manifest)


@pytest.mark.parametrize('frames', ['', ',', '4:2', '2:2', '-1:2', '0:20'])
def test_invalid_frame_ranges_are_not_silently_empty(frames: str) -> None:
    from src.tennis_scene.chat_annotation.local_agent.worker_context import parse_frames
    with pytest.raises(ValueError):
        parse_frames(frames, 12)
