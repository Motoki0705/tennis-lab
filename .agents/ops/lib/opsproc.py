"""Subprocess helpers shared by the ``.agents/ops`` jobs.

Every external command fails loudly with its stderr. Long-running agent commands run in
their own process group so a timeout or crash never leaves claude/codex (or the tools
they spawned) running after the caller removes their worktree.

Import with ``sys.path.insert(0, <.agents/ops/lib>)``; the ops jobs are scripts, not
an installed package.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

LOG = logging.getLogger("opsproc")
DEFAULT_TERM_GRACE_SECONDS = 5.0


class CommandError(RuntimeError):
    """An external command (gh, git, agent) failed or returned unusable output."""


def run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    ok_codes: tuple[int, ...] = (0,),
    env: Mapping[str, str] | None = None,
) -> str:
    """Run a short-lived ``cmd`` (gh/git) and return stdout; raise
    :class:`CommandError` on any failure.

    ``ok_codes`` lists exit statuses that are not failures (``git grep`` uses 1
    for "no match"). Long-running commands use :func:`run_process_group`.
    """
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            env=None if env is None else dict(env),
        )
    except FileNotFoundError as exc:
        raise CommandError(f"{cmd[0]}: command not found") from exc
    if proc.returncode not in ok_codes:
        raise CommandError(
            f"{shlex.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def _live_group_members(pgid: int) -> list[int]:
    """PIDs of non-zombie processes in process group ``pgid`` (Linux ``/proc``)."""
    members: list[int] = []
    for stat_file in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_file.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited while scanning
        # Fields after "(comm)": state ppid pgrp ...; comm itself may contain ")".
        fields = stat[stat.rindex(")") + 2 :].split()
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(stat_file.parent.name))
    return members


def _wait_group_gone(proc: subprocess.Popen[str], pgid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        proc.poll()  # reap the leader so it does not linger as a zombie
        if not _live_group_members(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _terminate_group(
    proc: subprocess.Popen[str], pgid: int, grace: float
) -> list[int]:
    """Stop every process in ``pgid`` (TERM, then KILL), reap the leader, and return
    the PIDs that were still alive. Raises if anything survives SIGKILL."""
    alive = _live_group_members(pgid)
    if alive:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        if not _wait_group_gone(proc, pgid, grace):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            if not _wait_group_gone(proc, pgid, grace):
                raise CommandError(
                    f"process group {pgid} survived SIGKILL: {_live_group_members(pgid)}"
                )
    proc.wait()
    return alive


def run_process_group(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    term_grace: float = DEFAULT_TERM_GRACE_SECONDS,
) -> str:
    """Run ``cmd`` in its own process group and return stdout.

    Agent CLIs spawn descendants (``agent_exec.sh`` -> claude/codex -> tools);
    killing only the direct child on timeout would leave them running. On timeout,
    error or exit, the whole group is terminated (SIGTERM, then SIGKILL after
    ``term_grace`` seconds) and reaped.
    """
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=None if env is None else dict(env),
        )
    except FileNotFoundError as exc:
        raise CommandError(f"{cmd[0]}: command not found") from exc
    pgid = proc.pid  # session leader: its pid is the process group id
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        killed = _terminate_group(proc, pgid, term_grace)
        proc.communicate()
        raise CommandError(
            f"{shlex.join(cmd)} timed out after {timeout}s (killed {killed})"
        ) from exc
    except BaseException:
        _terminate_group(proc, pgid, term_grace)
        raise
    leftover = _terminate_group(proc, pgid, term_grace)
    if leftover:
        LOG.warning("killed processes left behind by %s: %s", cmd[0], leftover)
    if proc.returncode != 0:
        raise CommandError(
            f"{shlex.join(cmd)} exited {proc.returncode}: {stderr.strip()}"
        )
    return stdout
