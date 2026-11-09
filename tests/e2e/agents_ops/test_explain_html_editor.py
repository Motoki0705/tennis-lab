"""Public editor cancellation/timeout and actual trusted hook boundaries."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.agents_ops.explain_editor import (
    editor_env as editor_env,
)
from tests.support.agents_ops.explain_editor import (
    opsproc,
    schema,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("termination", ["cancelled", "timeout"])
def test_cancel_and_timeout_reap_the_group_and_keep_current(
    editor_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    env = editor_env
    monkeypatch.setenv("FAKE_EXPLAIN_CASE", "hang")
    os.mkfifo(env.root / "ready")
    cancel = Event()
    offset = [0.0]
    clock = SimpleNamespace(
        monotonic=lambda: time.monotonic() + offset[0], sleep=time.sleep
    )
    monkeypatch.setattr(opsproc, "time", clock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            env.adapter.edit,
            env.request,
            env.current,
            env.context,
            timeout=3600,
            cancel=cancel,
        )
        with (env.root / "ready").open() as ready:
            pids = json.load(ready)
        assert opsproc.process_identity(pids["leader"]) is not None
        assert opsproc.process_identity(pids["tool"]) is not None
        if termination == "cancelled":
            cancel.set()
        else:
            offset[0] = (
                7200  # Advance the deadline only after both processes are ready.
            )
        with pytest.raises(schema.ContractError, match=termination):
            pending.result(timeout=10)
    assert opsproc.process_identity(pids["leader"]) is None
    assert opsproc.process_identity(pids["tool"]) is None
    assert env.store.current("geometry") == env.current
    assert env.store.history("geometry")[0].revision_id == env.current.revision_id
    # Service shutdown uses this same Event for its active editor calls.
    before = (env.root / "calls.jsonl").read_bytes()
    cancel.set()
    with pytest.raises(schema.ContractError, match="cancelled"):
        env.adapter.edit(env.request, env.current, env.context, cancel=cancel)
    assert (env.root / "calls.jsonl").read_bytes() == before


def test_actual_no_tools_hook_denies_public_writes(
    editor_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = editor_env
    monkeypatch.setenv("FAKE_EXPLAIN_CASE", "tool")
    with pytest.raises(schema.ContractError, match="invalid_model_output"):
        env.adapter.edit(env.request, env.current, env.context)
    assert not (env.root / "forbidden-public-file").exists()
    assert env.store.current("geometry") == env.current
    records = list((env.root / "private").glob("*/hooks.jsonl"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["decision"] == "deny"
