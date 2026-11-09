# Explain HTML document contract

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This package provides the shared document/edit contract and durable revision store
for [#1710](https://github.com/Motoki0705/tennis-lab/issues/1710). It contains no
Gemini invocation, HTTP server, browser UI, image generation or quality evaluator.
It uses the existing Pydantic dependency and Python's standard library.

## Contract source and entry points

[schema.py](schema.py) is the sole source of wire models, HTTP paths, response
types, job transitions, public errors and JSON/Python usage examples. Import it
directly; generate OpenAPI 3.1 with the repository Python and `PYTHONPATH=.agents/ops`
using `python -m explain_html.schema`. Do not copy generated schemas into adapters
or web assets. `contract_schema()` returns the same contract in memory;
`examples()` includes an initial candidate and a deliberately incomplete verdict.
Use `parse()` for strict JSON parsing: unknown fields and type coercions fail.

| Module | Responsibility |
|---|---|
| [schema.py](schema.py) | Document, evidence/assets/figures, requests, candidates, verdicts and HTTP contract |
| [document.py](document.py) | Identity, HTML references, selection checks and complete change accounting |
| [store.py](store.py) | Initial import, stage, commit, current/history, restoration and export snapshot |
| [paths.py](paths.py) | POSIX directory-descriptor access without following symlinks |

## Identity, targets and candidates

Producers put `data-content-id` on addressable logical blocks. Preserve an ID when
its logical content survives; allocate a new ID for new content. No headings,
layout, colors, block tag names or order are prescribed. HTML must have balanced,
explicit closing tags; duplicate HTML/content/artifact IDs fail. Native HTML `id`
and `data-content-id` have separate namespaces. Figure and evidence references
use `data-figure-id` and `data-evidence-id`; resources may use `data-asset-id`.

Every request binds its target to `base_revision`. A content target must exist.
A selection additionally supplies exact text, prefix/suffix and start/end offsets
in **UTF-16 code units of the block's DOM `textContent`**. Count after HTML input
CRLF/CR-to-LF preprocessing, entity expansion, and removal of one immediately
following LF for `pre`/`textarea`/`listing`, as specified by the
[HTML parser](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream).
Other whitespace is preserved. Non-BMP characters occupy two code units. Raw HTML
remains unchanged for storage and markup comparisons. Use whole-block offsets, not a text-node offset or
`innerText`. Both position and quote/context must match, including surrogate
boundaries. There is no quote relocation or automatic rebase.

A Candidate is a complete resulting Document plus its originating EditRequest,
content changes, evidence changes with reasons, and a summary. This is the same
contract for initial generation, local edits and whole-document restructuring;
adapters assemble patches into a complete candidate before staging. Initial import
uses a document target and a null base; all later updates require the current ID.

`content_changes()` accounts for added/removed/changed block markup and blocks
whose figure definition, renderer or referenced asset changed. All managed URL
references are resolved against the asset manifest, including ordinary `src`,
`href`, `data` and each `srcset`/`imagesrcset` candidate without ID annotations or
Figure records. Every occurrence must belong to a content block; unowned
references, including global/head resources without an enclosing content ID, fail
with `broken_reference`. Changes notify every containing content ID. Moving unchanged
blocks or changing only their outer layout can have an empty change list.
`evidence_changes()` accounts for evidence additions/removals/updates. Staging
rejects missing, invented or duplicate declarations; the reason for an evidence
change is checked by the independent semantic verifier, not inferred by the store.

## Validation and publication

The candidate digest binds the request, resulting HTML, all metadata and declared
changes, including every asset's content digest. `stage()` records a private
candidate and returns a StagedCandidate. `staged_candidate()` and
`staged_asset_bytes()` expose that immutable input to trusted validators.
Every asset is supplied as bytes keyed by asset ID; unchanged bytes may be reused
only when the current revision has the same asset ID and digest. Missing or
different bytes fail explicitly. Renderer version and renderer bytes are pinned
by each revision's Figure/Asset records.

`commit()` requires a ValidationResult for that exact candidate with structural,
semantic and rendering checks all passed and no outstanding issues. It repeats
structural/change/base and byte-integrity checks under a per-document filesystem
lock. The writer cannot submit its own verdict through Candidate or HTTP.
Only trusted orchestration calls commit; verdicts are not cryptographic proof
of validator execution or of semantic truth.

The controller must provide HTML/SVG sanitization, renderer policy, independent
semantic verification and rendering checks. The structural checker resolves
local HTML references and fragments, permits external citation links, and refuses
remote resource URLs and URL-bearing inline CSS. Source sets accept canonical
managed URLs with optional consistent `w` or `x` descriptors; malformed, duplicate
or mixed descriptors and unresolved candidates fail the entire source set.
It does not interpret
asset internals or execute JavaScript. The visual validator owns CSS/SVG/script
contents and graph-specific definitions. A passed structural check alone is
insufficient for publication.

## Storage, recovery and export

Use a dedicated Git-ignored runtime directory, such as the `outputs/` location
in the source usage example. Keep the whole root private and serve files through
the allowlisted APIs. Storage layout:

```text
<root>/private/documents/<document_id>/
  lock
  current.json
  stages/<stage_id>/{stage.json,public/...}
  revisions/<revision_id>/{record.json,public/...}
<root>/exports/<snapshot_id>/...
```

Each revision contains its complete HTML, evidence, figure definitions and asset
bytes. Files and directories are flushed before replacing `current.json` in one
atomic operation. Failure before that replacement preserves the old current
revision. A process/power loss at the replacement boundary leaves the old or new
complete bundle; after an uncertain acknowledgement, reopen and reconcile by
request ID. Failed pending directories and unreachable installed revisions remain
private, never appear in history and are not readable through `revision()`.
The store does not silently recover a corrupt current revision by choosing an
older one. It has no background garbage collector.

`current()`, `revision()` and `history()` return published revisions, newest first
for history. A repeated successful commit of the same staged candidate is
idempotent; a reused request ID with different content conflicts. `restore()`
copies a published old document into a **new** revision whose parent is current,
records `restored_from`, rechecks every byte, and reuses the old document's passed
semantic/rendering verdicts. Old base IDs never become current again. Identical
restore retries return the already created revision.

`snapshot()` pins an explicit published revision and atomically installs a
standalone bundle. Its closed file list is HTML, declared assets, evidence,
figures and a public manifest. It never copies the store tree, requests, verdicts,
raw prompts, credentials or private logs. Everything in Document, including
evidence and provenance, is intended to be public. Do not put private context in
Document. `asset_bytes()` resolves by manifest ID; `snapshot_file()` resolves by
the snapshot allowlist. Absolute/parent/encoded paths, symlink components and
hard-linked files are rejected. Assets have canonical `assets/<asset_id>.<ext>`
paths. The implementation targets local POSIX filesystems, including WSL;
Windows Python, network filesystem atomicity and real power-loss durability are
not qualified here.

## Downstream responsibilities

HTTP JSON endpoints use the session-token header named in `contract_schema()`;
isolated document/export content has no editing credential. Route and body IDs
must agree. Resolve content files and asset names against the selected revision's
allowlist, never concatenate arbitrary HTTP paths into the private tree. Capture
a revision ID before serving HTML/assets so a current-version switch cannot mix
two revisions. Expose only schema-defined public errors, never raw exceptions.

The job controller owns durable request deduplication before generator execution,
cancel/commit synchronization, job transitions and interrupted-job recovery. It
must check cancellation and revision again at commit, keep the old view while a
job runs, and never regenerate on status reads or reconnect. The store's lock and
base comparison prevent conflicting publications but do not manage subprocesses
or cancel state. The UI/renderer owns sandboxing, selection reporting, change
navigation and the static-export editing-unavailable indicator.

## Verification

Run the repository `.venv/bin/python -m pytest` on
`tests/unit/agents_ops/explain_html` and
`tests/integration/agents_ops/explain_html`, with `--collect-only` to count cases
and `--durations=0 --durations-min=0` to measure them. Tests use
real temporary files, asymmetric blocks, multiple assets,
declarative graph data, exception injection and synchronized competing writers.
They require no external API, GPU, browser or real-time sleep.
