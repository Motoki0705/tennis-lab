"""Small rollout-trace bundle builder shared by analyzer tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_sample_trace_bundle(root: Path) -> Path:
    """Write a two-inference trace with patch and terminal tool calls."""

    bundle = root / "trace-fixture"
    payload_dir = bundle / "payloads"
    payload_dir.mkdir(parents=True)

    patch_text = """*** Begin Patch
*** Update File: src/example.py
@@
-old_value = 1
+new_value = 2
*** End Patch"""
    request_one = {
        "model": "gpt-test",
        "instructions": "Follow repository instructions.",
        "input": [_message("user", "Inspect and update src/example.py")],
        "tools": [
            {"type": "custom", "name": "apply_patch"},
            {"type": "function", "name": "exec_command"},
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    request_two_items = [
        _message("user", "Inspect and update src/example.py"),
        {
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "Inspect the source, then implement the patch.",
                }
            ],
            "encrypted_content": "opaque",
        },
        {"type": "custom_tool_call", "call_id": "call-patch", "input": patch_text},
        {
            "type": "custom_tool_call_output",
            "call_id": "call-patch",
            "output": "Done!",
        },
        {
            "type": "function_call",
            "call_id": "call-exec",
            "name": "exec_command",
            "arguments": '{"cmd":"sed -n 1,80p src/example.py"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-exec",
            "output": "new_value = 2",
        },
    ]
    request_two = {
        **request_one,
        "input": request_two_items,
    }
    payloads: dict[str, Any] = {
        "request-1": request_one,
        "response-1": {"output_items": [], "token_usage": {}},
        "request-2": request_two,
        "response-2": {"output_items": [], "token_usage": {}},
        "patch-invocation": {"input": patch_text},
        "patch-result": {"output": "Done!"},
        "exec-invocation": {"cmd": "sed -n '1,80p' src/example.py"},
        "exec-result": {
            "output": "new_value = 2",
            "original_token_count": 200,
        },
    }
    raw_payloads: dict[str, Any] = {}
    for payload_id, payload in payloads.items():
        relative = f"payloads/{payload_id}.json"
        (bundle / relative).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        raw_payloads[payload_id] = {
            "raw_payload_id": payload_id,
            "kind": {"type": _payload_kind(payload_id)},
            "path": relative,
        }

    conversation_items = {
        "user-1": _conversation_item(
            "user-1",
            "user",
            "message",
            [{"type": "text", "text": "Inspect and update src/example.py"}],
        ),
        "reasoning-1": _conversation_item(
            "reasoning-1",
            "assistant",
            "reasoning",
            [
                {
                    "type": "summary",
                    "text": "Inspect the source, then implement the patch.",
                },
                {"type": "encoded", "label": "reasoning", "value": "opaque"},
            ],
        ),
        "patch-call": _conversation_item(
            "patch-call",
            "assistant",
            "custom_tool_call",
            [
                {
                    "type": "payload_ref",
                    "label": "patch",
                    "raw_payload_id": "patch-invocation",
                }
            ],
            call_id="call-patch",
        ),
        "patch-output": _conversation_item(
            "patch-output",
            "tool",
            "custom_tool_call_output",
            [
                {
                    "type": "payload_ref",
                    "label": "result",
                    "raw_payload_id": "patch-result",
                }
            ],
            call_id="call-patch",
        ),
        "exec-call": _conversation_item(
            "exec-call",
            "assistant",
            "function_call",
            [
                {
                    "type": "payload_ref",
                    "label": "command",
                    "raw_payload_id": "exec-invocation",
                }
            ],
            call_id="call-exec",
        ),
        "exec-output": _conversation_item(
            "exec-output",
            "tool",
            "function_call_output",
            [
                {
                    "type": "payload_ref",
                    "label": "result",
                    "raw_payload_id": "exec-result",
                }
            ],
            call_id="call-exec",
        ),
        "assistant-2": _conversation_item(
            "assistant-2",
            "assistant",
            "message",
            [{"type": "text", "text": "Implemented and verified."}],
        ),
    }
    state = {
        "schema_version": 1,
        "trace_id": "trace-test",
        "rollout_id": "rollout-test",
        "started_at_unix_ms": 1_000,
        "ended_at_unix_ms": 1_500,
        "status": "completed",
        "root_thread_id": "thread-root",
        "threads": {},
        "codex_turns": {},
        "conversation_items": conversation_items,
        "inference_calls": {
            "inference-1": _inference(
                "inference-1",
                10,
                "request-1",
                "response-1",
                ["user-1"],
                ["reasoning-1", "patch-call", "exec-call"],
                ["tool-patch", "tool-exec"],
                input_tokens=100,
                cached_tokens=20,
                output_tokens=30,
                reasoning_tokens=12,
            ),
            "inference-2": _inference(
                "inference-2",
                50,
                "request-2",
                "response-2",
                [
                    "user-1",
                    "reasoning-1",
                    "patch-call",
                    "patch-output",
                    "exec-call",
                    "exec-output",
                ],
                ["assistant-2"],
                [],
                input_tokens=140,
                cached_tokens=80,
                output_tokens=20,
                reasoning_tokens=4,
            ),
        },
        "code_cells": {},
        "tool_calls": {
            "tool-patch": _tool(
                "tool-patch",
                20,
                "apply_patch",
                "patch-call",
                "patch-output",
                "patch-invocation",
                "patch-result",
            ),
            "tool-exec": {
                **_tool(
                    "tool-exec",
                    30,
                    "exec_command",
                    "exec-call",
                    "exec-output",
                    "exec-invocation",
                    "exec-result",
                ),
                "terminal_operation_id": "terminal-op-1",
                "summary": {"type": "terminal", "operation_id": "terminal-op-1"},
            },
        },
        "terminal_sessions": {},
        "terminal_operations": {
            "terminal-op-1": {
                "operation_id": "terminal-op-1",
                "terminal_id": "terminal-1",
                "tool_call_id": "tool-exec",
                "kind": "exec_command",
                "execution": _execution(31),
                "request": {
                    "type": "exec_command",
                    "command": ["sed", "-n", "1,80p", "src/example.py"],
                    "display_command": "sed -n 1,80p src/example.py",
                    "cwd": "/repo",
                    "yield_time_ms": 10_000,
                    "max_output_tokens": 10_000,
                },
                "result": {
                    "exit_code": 0,
                    "stdout": "new_value = 2\n",
                    "stderr": "",
                    "formatted_output": "new_value = 2",
                    "original_token_count": 200,
                    "chunk_id": None,
                },
                "model_observations": [],
                "raw_payload_ids": ["exec-result"],
            }
        },
        "compactions": {},
        "compaction_requests": {},
        "interaction_edges": {},
        "raw_payloads": raw_payloads,
    }
    (bundle / "state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )
    return bundle


def _message(role: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _conversation_item(
    item_id: str,
    role: str,
    kind: str,
    parts: list[dict[str, Any]],
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "thread_id": "thread-root",
        "codex_turn_id": "turn-1",
        "first_seen_at_unix_ms": 1_000,
        "role": role,
        "channel": "analysis" if kind == "reasoning" else None,
        "kind": kind,
        "agent_message": None,
        "body": {"parts": parts},
        "call_id": call_id,
        "produced_by": [],
    }


def _inference(
    inference_id: str,
    seq: int,
    request_id: str,
    response_id: str,
    request_items: list[str],
    response_items: list[str],
    tools: list[str],
    *,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
) -> dict[str, Any]:
    return {
        "inference_call_id": inference_id,
        "thread_id": "thread-root",
        "codex_turn_id": "turn-1",
        "execution": _execution(seq),
        "model": "gpt-test",
        "provider_name": "test-provider",
        "response_id": f"response-{inference_id}",
        "upstream_request_id": f"request-{inference_id}",
        "request_item_ids": request_items,
        "response_item_ids": response_items,
        "tool_call_ids_started_by_response": tools,
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_tokens,
        },
        "raw_request_payload_id": request_id,
        "raw_response_payload_id": response_id,
    }


def _tool(
    tool_id: str,
    seq: int,
    kind: str,
    call_item: str,
    output_item: str,
    invocation_id: str,
    result_id: str,
) -> dict[str, Any]:
    return {
        "tool_call_id": tool_id,
        "mcp_call_id": None,
        "model_visible_call_id": f"call-{tool_id}",
        "code_mode_runtime_tool_id": None,
        "thread_id": "thread-root",
        "started_by_codex_turn_id": "turn-1",
        "execution": _execution(seq),
        "requester": {"type": "model"},
        "kind": {"type": kind},
        "model_visible_call_item_ids": [call_item],
        "model_visible_output_item_ids": [output_item],
        "terminal_operation_id": None,
        "summary": {"type": "generic", "label": kind},
        "raw_invocation_payload_id": invocation_id,
        "raw_result_payload_id": result_id,
        "raw_runtime_payload_ids": [],
    }


def _execution(seq: int) -> dict[str, Any]:
    return {
        "started_at_unix_ms": 1_000 + seq,
        "started_seq": seq,
        "ended_at_unix_ms": 1_010 + seq,
        "ended_seq": seq + 1,
        "status": "completed",
    }


def _payload_kind(payload_id: str) -> str:
    if payload_id.startswith("request"):
        return "inference_request"
    if payload_id.startswith("response"):
        return "inference_response"
    if payload_id.endswith("invocation"):
        return "tool_invocation"
    return "tool_result"
