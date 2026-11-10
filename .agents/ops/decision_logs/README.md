# Parent decision records

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This package defines the shared version 1 contract for captured results, parent
decisions, actions and corrections. It stores private snapshots and prepares safe
publication units. Entry-point capture is implemented in #1815; Git publication
and generated navigation READMEs belong to #1816. This package performs no agent
launches, Git/HTTP operations, scheduling or acceptance decisions.

## Contract and ownership

Code is authoritative; the field sets, enums and validators are not duplicated
here. Existing [worker logs](../worker_logs/README.md) retain their three-file
format. Their schema helpers, `Guard`, strict JSON decoder, flock and durable
atomic writer are reused directly.

| Module | Consumer contract |
|---|---|
| [schema.py](schema.py) | `new_id()`, `validate_record()`, `json_bytes()`; exact versioned fields and tagged missing values |
| [graph.py](graph.py) | `validate_graph()`, `references()`, `states()`; typed references, DAG, action history and explicit verification |
| [journal.py](journal.py) | `Journal`; immutable captures and append-only preparation/publication history |
| [projection.py](projection.py) | `render()`, `validate_public()`; the publication envelope `{record, sanitization}` |
| [aggregation.py](aggregation.py) | `aggregate()`; display groups referencing all original records |

Each record has a random UUID4 hex ID, kind, scope, three distinct times and typed
data. Scope carries the repository, parent, nullable child issue, public activity
key, existing worker delegation key/turn and nullable public review key. Direct
parent work needs no invented child issue. New IDs and keys must use `new_id()`;
private session/run IDs and delivery keys never determine public IDs. Their
association exists only in capture entries' private context.

`validate_record()` checks exact keys and types, including nested objects;
serialized input uses the shared duplicate-key-rejecting decoder.
`validate_graph()` additionally rejects missing/wrong-kind references, cross-parent
references, cycles, conflicting action histories and mismatched decision plans.
References express causality independently of timestamps. A graph can branch
from one result to several decisions/actions and join several results in a decision.

## Results, decisions and lifecycle

Results distinguish machine observation, worker report and parent verification.
Runtime completion, artifact quality, CI, final-report availability and publication
are separate observations. No file's existence proves parent receipt or acceptance.
Explicit parent verification is a result whose `verifies` references the checked
results. `states()` derives undecided/unverified states from those references.

Each result compares code, requirements, instruction, execution conditions and
verified scope against `previous_result_id`. Without a baseline all changes must
be `unconfirmed`. Evidence has a fixed revision or content fingerprint; a mutable
URL plus fingerprint identifies a snapshot but does not promise to reconstruct
its full text. Private/unavailable evidence retains a reason without a private URL.

Decisions contain input result IDs, a short reason, concrete next-action plans,
reuse/recheck scopes, unresolved work, resume conditions and a typed source:
settings, human instruction or parent choice. An initial decision instead uses
fixed request/agreement evidence in `basis`, with no fabricated predecessor result.
Plans have their own random IDs; each branch uses a separate plan/action.

Actions reference the decision and plan. If no decision was supplied, both IDs
are null and `decision_missing_reason` is mandatory; no decision is invented.
State updates append action snapshots linked by `previous_action_id`, preserving
their scope, decision, plan and operation. An earlier requested/running action
snapshot is the source of a result; a later terminal snapshot links that result
in `result_ids`. Pointing both sides at the same snapshot would create a cycle.
Results from actions marked only `decided_not_executed` are rejected.

Corrections append a target reference and reason, optionally referencing a separate
replacement record of the same kind and scope. Original records remain present;
the graph does not silently replace them or treat corrections as new observations.

## Missing values, times and usage

Availability is either `{status: observed, value: ...}` or
`{status: unavailable, reason: ...}`. Unknown times, durations, usage and effective
settings are never replaced with zero, the selected model or the current time.
Observed zero remains valid. Occurrence/receipt times use availability values;
`recorded_at` is the actual recording time. All observed times require a timezone.

Selected model/effort retains the specification source; observed effective values
are independent. Measurements retain their evidence. Usage retains per-turn versus
cumulative scope and a public counter key; neither grouping nor this package adds
cumulative counters, estimates cost or interprets cached tokens as billed usage.

Missing final reports retain a reason such as `missing_final`. A pending decision,
unrecorded parent verification, decided-but-unexecuted action, pre-start rejection
and pending publication are distinct fields/states. Their meaning is not inferred
from runtime exit status.

## Private journal and publication boundary

Construct `Journal(parent_store.directory, repo, parent)` using the existing
[parent state](../parent/README.md). The package creates one parent journal below
`decision-logs/`, separated by repository and parent. Directories are private and
files/locks are owner-only. Symlinks are rejected. Entry lists only grow; flock,
file/directory fsync and atomic replacement protect each append. Corrupt JSON,
duplicate/unknown keys, unknown versions and identity changes fail without resetting
the file. This is a local filesystem contract, not distributed ownership.

- `capture(key, record, private=...)` freezes exact input. The private context has
  sessions, runs, other identifiers, checkout roots and a nullable checkout commit.
  The record ID may be omitted; capture allocates it inside the lock. Replaying an
  identical private delivery key returns the same ID without writing. Changed input
  is rejected; callers must reuse the frozen times/scope instead of recapturing them.
- `prepare(record_id, key=..., at=...)` freezes the safe envelope or appends
  `blocked`, preserving the original. Prepare dependencies first. Retrying a blocked
  projection requires `retry=True` and a fresh delivery key, without rerunning work.
  A ready projection cannot be replaced.
- `publication(record_id, key=..., status=..., at=..., evidence=..., error=...)`
  appends pending, blocked or published facts independently of capture. Published
  requires fixed public evidence and is terminal. Callers verify Git/HTTP facts;
  this library does not certify that publication actually happened.
- `public_payloads()` returns only validated safe envelopes in dependency order,
  including sanitization counts. Publishers use these units and revalidate them.
  `public_records()` is the record-only view for graph/grouping consumers. `status()`
  reports capture/publication and missing-state facts without private mappings.

The shared worker-log guard sanitizes prose, then rechecks all public fields and
sanitization metadata. It masks known private IDs and local paths with counted
replacements. Fixed reference URLs, identities and model metadata are checked
without rewriting their meaning. Residual credentials, unknown UUID/session
notations, unsafe URLs and control characters block preparation. This is not a
complete classifier of unknown secrets. Git paths, branches, generated READMEs and
PR text still need the publisher's public checks.

## Display aggregation

`aggregate(records)` groups adjacent machine checks or fetch retries only when the
target, scope, status, action, condition key and model settings match, comparison
links form a chain and every change is explicitly unchanged. Capture must reuse a
random condition key only for identical conditions. Unconfirmed/changed conditions,
AI starts/continuations, decisions, action snapshots and corrections are boundaries.
Fetch attempts within one retrieval action can form a group; separate actions stay
visible. Intermediate observations referenced by later decisions, verification or
corrections remain visible even when that reference follows the entire sequence.
Each group preserves all record IDs, count, first/last recording times and
the last record ID. Consumers look up that record for the final result. Raw captures
and intermediate failures remain available; durations and usage are not invented.

## Validation

```bash
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_logs.py \
  tests/integration/agents_ops/test_decision_log_journal.py \
  tests/unit/agents_ops/test_worker_logs.py -n 1 --durations=10
```

Tests exercise branching/joining, explicit missing states, rejection cases,
sanitization, grouping, two real writer processes, interrupted fsync and corrupt
journals. They use temporary state and require no GPU, agents or network. Live
capture/publication and post-merge operational acceptance remain downstream work.
