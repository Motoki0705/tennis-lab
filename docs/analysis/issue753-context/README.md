# Issue #753 context-pressure snapshot

This directory contains a content-free aggregate and a PNG derived from the
Issue #753 parent-session JSONL transcript. The visualization compares
serialized tool-output volume across the four context windows separated by
three compactions. Its timeline overlays wait/process-poll calls, subagent
control calls, and workflow re-read/contract output.

## Reproduce from the raw parent transcript

The raw JSONL remains outside the repository. Pass its path explicitly:

```bash
.venv/bin/python docs/analysis/issue753-context/generate_context_pressure.py \
  --transcript "$TRANSCRIPT" \
  --snapshot docs/analysis/issue753-context/context_snapshot.json \
  --image docs/analysis/issue753-context/issue753_context_pressure.png
```

The generator retains no message, command, reasoning, or tool-output body in
the snapshot. It records only counts, compact-JSON character lengths, event
positions, compaction timestamps, source file metadata, and a SHA-256 digest.

## Reproduce from the committed aggregate

No transcript access is needed to render the committed snapshot:

```bash
.venv/bin/python docs/analysis/issue753-context/generate_context_pressure.py \
  --from-snapshot docs/analysis/issue753-context/context_snapshot.json \
  --image docs/analysis/issue753-context/issue753_context_pressure.png
```

## Counting and classification

For each parent-session `custom_tool_call_output` or `function_call_output`,
the measured unit is:

```text
len(json.dumps(payload.output, ensure_ascii=False, separators=(",", ":")))
```

The initiating call assigns one of four deterministic categories. Formal gates
include canonical checks and direct pytest/Ruff/mypy/CI verification. Workflow,
design, and source loading covers read-oriented inspection. Workflow re-read
and contract replay covers repeated workflow documents plus Issue-state,
artifact, spawn, and verification contracts. Remaining results are grouped as
other tool output. Async cell waits and PTY continuations inherit the category
of the call they continue.

The lower timeline counts wait primitives (`wait`, `wait_agent`, and
`write_stdin`) separately from subagent controls (`spawn_agent`,
`followup_task`, `send_message`, and `interrupt_agent`). The alignment with
compaction markers is a temporal association, not proof that orchestration
caused compaction.

Encrypted reasoning, message bodies, compaction replacement history, and
child-agent transcript files are excluded. Parent-visible collaboration result
envelopes remain part of the parent's tool-output pressure.
