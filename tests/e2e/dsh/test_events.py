"""Pinned headless truncation contracts, without API requests or credentials."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.dsh.events import EventError, event_prefix, parse_events

SOURCE = Path(__file__).resolve().parents[3] / "tools/dsh"


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event) + "\n").encode()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "tool_call", "callId": "c", "tool": "read", "truncated": True},
        {
            "type": "tool_call",
            "callId": "c",
            "tool": "read",
            "input": {"nested": "[truncated: depth]"},
            "truncated": True,
        },
        {
            "type": "tool_result",
            "callId": "c",
            "status": "completed",
            "result": "shortened",
            "truncated": True,
        },
        {"type": "text", "text": "shortened", "truncated": True},
        {"type": "thinking", "text": "shortened", "truncated": True},
        {"type": "error", "message": "shortened", "truncated": True},
        {
            "type": "session",
            "sessionId": "session-shortened",
            "cwd": "/shortened",
            "truncated": True,
        },
        {"type": "status", "phase": "turn_end", "turn": 1, "truncated": True},
        {
            "type": "status",
            "phase": "step_end",
            "turn": 1,
            "step": 1,
            "truncated": True,
        },
        *[
            {"type": kind, "truncated": True}
            for kind in (
                "session",
                "status",
                "tool_call",
                "tool_result",
                "text",
                "thinking",
                "error",
            )
        ],
    ],
)
def test_pinned_shortened_shapes_are_readable_but_rejected_by_smoke(
    event: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "events.ndjson"
    path.write_bytes(_line(event))
    assert event_prefix(path.read_bytes()) == ([event], True)
    with pytest.raises(EventError):
        parse_events(path)


@pytest.mark.parametrize(
    "input_value", [None, True, 42, "raw arguments", [], ["x"], {}]
)
@pytest.mark.parametrize("truncated", [False, True])
def test_tool_input_preserves_json_values_without_confusing_null_with_absence(
    input_value: Any, truncated: bool
) -> None:
    # Pinned parseArguments preserves scalars, arrays and invalid-JSON raw text.
    event = {"type": "tool_call", "callId": "c", "tool": "read", "input": input_value}
    if truncated:
        event["truncated"] = True
    assert event_prefix(_line(event)) == ([event], True)
    if not isinstance(input_value, dict) or truncated:
        assert event_prefix(_line(event), allow_truncated=False) == ([], False)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "tool_call", "callId": "c", "tool": "read"},
        {"type": "tool_result", "callId": "c", "status": "completed"},
        {"type": "text"},
        {"type": "thinking"},
        {"type": "error"},
        {"type": "session", "sessionId": "s"},
        {"type": "status", "phase": "turn_end", "turn": 1},
        {"type": "final"},
    ],
)
def test_unmarked_missing_fields_are_corruption(event: dict[str, Any]) -> None:
    assert event_prefix(_line(event)) == ([], False)


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize(
    "event",
    [
        {"type": "tool_call", "callId": "c", "input": {}},
        {"type": "tool_call", "callId": 5, "tool": "read", "input": {}},
        {"type": "tool_call", "callId": "c", "tool": []},
        {"type": "tool_result", "callId": "c", "status": "completed", "result": []},
        {"type": "tool_result", "callId": "c", "status": "other", "result": "x"},
        {"type": "tool_result", "status": "completed", "result": "x"},
        {"type": "text", "text": 5},
        {"type": "thinking", "text": None},
        {"type": "error", "message": {}},
        {"type": "session", "sessionId": "s", "cwd": []},
        {"type": "status", "phase": "turn_end", "turn": 1, "reason": []},
        {"type": "status", "phase": "turn_end"},
        {"type": "status", "phase": "step_end", "turn": 1},
        {"type": "status", "phase": "step_end", "turn": True, "step": 1},
        {"type": "status", "phase": "other", "turn": 1},
        {"type": "status", "phase": "step_end", "turn": 1, "step": 1, "usage": None},
        {
            "type": "status",
            "phase": "step_end",
            "turn": 1,
            "step": 1,
            "usage": {"inputTokens": "5", "outputTokens": 1},
        },
        {"type": "final", "text": None},
    ],
)
def test_marker_does_not_excuse_scalar_loss_or_wrong_types(
    event: dict[str, Any], truncated: bool
) -> None:
    if truncated:
        event = {**event, "truncated": True}
    assert event_prefix(_line(event)) == ([], False)


@pytest.mark.parametrize("marker", [False, 1, "true", None])
def test_truncation_marker_must_be_literal_true(marker: Any) -> None:
    assert event_prefix(_line({"type": "text", "text": "x", "truncated": marker})) == (
        [],
        False,
    )


def test_final_is_lossless_and_cannot_claim_truncation() -> None:
    event = {"type": "final", "text": "最終回答\n" * 20000}
    assert event_prefix(_line(event)) == ([event], True)
    assert event_prefix(_line({**event, "truncated": True})) == ([], False)


def test_real_pinned_serializer_shapes_without_api() -> None:
    node = shutil.which("node")
    modules = list(
        (SOURCE / "node_modules/@deepseek-ai/dsh-headless/lib").glob("json-stream-*.js")
    )
    if node is None or len(modules) != 1:
        pytest.skip("Pinned dsh is not installed; offline shape contracts still run")
    assert node is not None
    script = r"""
import {pathToFileURL} from 'node:url';
const {t: boundJsonLine, n: projectJsonRun} = await import(pathToFileURL(process.argv[1]));
const long = 'あ'.repeat(9000);
const large = Object.fromEntries(Array.from({length:5}, (_,i) => [`k${i}`, long]));
let deep = {};
for (let i=0; i<70; i++) deep = {nested:deep};
const events = [
  {type:'tool_call',callId:'c',tool:'read',input:large},
  {type:'tool_call',callId:'c',tool:'read',input:Object.values(large)},
  {type:'tool_call',callId:'c',tool:'read',input:{[long]:long}},
  {type:'tool_call',callId:'c',tool:'read',input:deep},
  {type:'tool_call',callId:'c',tool:'read',input:long},
  {type:'tool_call',callId:'\0'.repeat(9000),tool:'read',input:{}},
  {type:'tool_result',callId:'c',status:'completed',result:long},
  {type:'tool_result',callId:'c',status:'completed',result:'\0'.repeat(9000)},
  {type:'text',text:long},
  {type:'text',text:'\0'.repeat(9000)},
  {type:'thinking',text:long},
  {type:'thinking',text:'\0'.repeat(9000)},
  {type:'status',phase:'turn_end',turn:1,reason:{kind:'error',error:{message:long}}},
  {type:'status',phase:'turn_end',turn:1,reason:{kind:'error',error:large}},
  {type:'status',phase:'step_end',turn:1,step:1,usage:{inputTokens:1,outputTokens:1,...large}},
  {type:'session',sessionId:'session-fixture',cwd:long},
  {type:'session',sessionId:'session-fixture',cwd:'\0'.repeat(9000)},
  {type:'error',message:long},
  {type:'error',message:'\0'.repeat(9000)},
];
for (const event of events) process.stdout.write(boundJsonLine(event) + '\n');
const projection = projectJsonRun({on:()=>()=>{}}, {session:{},id:'session-final'}, process.stdout, {cwd:'/fixture'});
projection.finish(long.repeat(5));
projection.dispose();
"""
    result = subprocess.run(
        [node, "--input-type=module", "--eval", script, str(modules[0])],
        capture_output=True,
        check=True,
        timeout=20,
    )
    events, complete = event_prefix(result.stdout)
    assert complete, result.stderr
    assert len(events) == 21
    assert all(event.get("truncated") is True for event in events[:-2])
    assert events[0] == {
        "type": "tool_call",
        "callId": "c",
        "tool": "read",
        "truncated": True,
    }
    assert events[1] == events[0]
    assert events[2]["input"]
    assert len(next(iter(events[2]["input"])).encode()) <= 8192
    assert "[truncated: depth]" in json.dumps(events[3])
    assert isinstance(events[4]["input"], str)
    for index in (5, 7, 9, 11, 16, 18):
        assert events[index] == {"type": events[index]["type"], "truncated": True}
    assert "reason" in events[12] and "reason" not in events[13]
    assert "usage" not in events[14]
    assert all(len(line) + 1 <= 32768 for line in result.stdout.splitlines()[:-1])
    assert events[-1] == {"type": "final", "text": "あ" * 9000 * 5}
