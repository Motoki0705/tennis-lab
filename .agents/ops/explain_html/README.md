# Explain HTML document contract

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This package provides the shared document/edit contract and durable revision store
for [#1710](https://github.com/Motoki0705/tennis-lab/issues/1710), plus visual import,
managed rendering and a qualified image-adapter boundary. It contains no Gemini
invocation, HTTP server, browser UI, image-provider implementation or quality
evaluator. It uses existing Pydantic/Pillow dependencies and the standard library.

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
| [images.py](images.py) | Local raster import, capability reporting and fail-closed image adapter |
| [visual_definitions.py](visual_definitions.py) | Strict observation-error/comparison graph definitions and numerical model |
| [visuals.py](visuals.py) | SVG/HTML/CSS safety, figure markup and offline renderer policy |
| [renderers/visuals-v1.js](renderers/visuals-v1.js) | Immutable managed renderer version 1 |

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

## Visuals and image generation

Use generated raster images for qualitative scenes/metaphors, SVG for exact
structures, and managed graphs for numerical comparisons or parameter changes.
Every Asset/Figure carries its ID, caption, alt and public provenance. Managed
markup displays the figure caption, provenance kind and source together; image
alt and graph accessible labels come from the Figure. Contextual figure captions
may differ from the reusable asset caption; image/SVG provenance must match.

`images.import_image(path, asset_id=..., caption=..., alt=..., provenance=...)`
returns an `ImportedAsset` with `.asset` and unchanged `.content`. It accepts
decoded, single-frame PNG/JPEG/WebP, limited to 16 MiB and 16 megapixels. An image
saved by an interactive Codex imagegen session uses the same import entry point
and `generated_concept` provenance. `visuals.import_svg(content, ...)` accepts
bounded SVG geometry/text with an SVG namespace and finite positive viewBox.
Scripts, foreignObject, animation, styles, entities and resource references are
rejected; nothing is silently stripped. SVG is embedded as an img, never object
or inline active markup.

Use `managed_renderer()` for its Asset/bytes and Renderer version `1`. Graph
Figure.asset_ids must contain this renderer alone. Figure.definition is parsed
by `parse_graph()`; the models and their `model_json_schema()` are the source for
graph-specific settings within the existing wire contract. Unknown fields,
numeric coercion, nonfinite values and out-of-domain data fail. The two supported
types are `observation_error` and `comparison`; there is no formula/JS editor.
The former is explicitly hypothetical, assumes rectified calibrated pinhole
stereo and correct correspondence, and computes `Z = f B / (d + error)` in m
from focal/disparity in px and baseline in m. Its signed error slider, sample
inputs, axis units/domains and displayed data must agree. Comparison preserves
up to eight named series with increasing x values and explicit units/domains.
See the [worked fixtures](../../../tests/support/agents_ops/explain_html_visuals.py).

Put empty `<figure data-figure-id="..."></figure>` slots in any page layout,
inside their declared content block (or on the block itself). A head and body
with explicit closing tags are required. `render_document(document)` fills these
slots and adds a CSP meta and the exact managed script; run it **before** computing
candidate changes. Graphs include initial SVG, source tables, assumptions, units,
input ranges and accessible controls, so the initial explanation survives with
JS disabled. The same script bytes are stored as an Asset and embedded in HTML,
authorized by their CSP hash. The script belongs to the first graph's content
block and mounts all graphs after DOM parsing, preserving reference ownership.
This avoids external-script SRI/CORS failures under
file://. No CDN, fetch, module import or server is needed after snapshot export.
Keep version 1 immutable; introduce a new file/ID/version for a future renderer.
Serve old revision HTML and assets from that revision, never inject today's code.

After stage, call `validate_visuals(document, all_asset_bytes_by_id)` on the
immutable staged document and **all** staged bytes, obtained through the store
APIs. It checks reference closure, digests, MIME, asset internals, managed script
identity and canonical figure/data markup. Inline event handlers, arbitrary
scripts, embedded browsing contexts and URL-bearing CSS are refused. Ordinary
CSS and media/supports/layer/container rules are allowed; imports, escapes and
CSS comments are outside this conservative subset. This deterministic check
does not produce a semantic/rendering verdict. The trusted controller must still
run independent meaning/browser checks before commit and provide iframe sandbox
isolation without editor credentials. The CSP may also be sent as an HTTP header
using `content_security_policy(document)`.

`generate_image(..., adapter=...)` only returns a real validated image. The default
`UnavailableImageGenerator.capability()` gives a reason and reports unavailable;
no credentials or provider are auto-discovered. A trusted, separately qualified
adapter implements `ImageGenerator` (capability and generation returning bytes
with a public source). Unavailable generation raises the shared
`image_generation_unavailable` error; provider/invalid-artifact failures raise
`asset_generation_failed`, without provider details in public errors. The
controller must leave the **whole image request incomplete**, abort publication
and retain current on either outcome. Do not publish a partial text/SVG substitute
as successful image generation. Existing-image import, relocation/caption changes,
SVG and graph updates remain independently supported requests. The interactive
imagegen tool's availability is not proof of a local-service adapter. Qualified
service access, authentication and artifact retrieval require a separate probe;
the capability evidence and unresolved human choices belong in the issue/PR.

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
They require no external API, GPU, browser or real-time sleep. The visual addition
has 9 Python unit and 3 Node cases; run Node with
`node --test tests/unit/agents_ops/explain_html/visuals.test.mjs`.
The two existing restore/snapshot integration cases also use real raster/SVG and
managed graphs. Browser verification uses one shared Chromium setup and two
scenarios (offline slider/comparison and rejected SVG/current preservation with
JS-disabled initial rendering):

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chromium \
  node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

The harness prints setup/scenario times and its temporary screenshot directory.
No real image generation or external API is run in normal tests/CI.
The existing JavaScript pytest entrypoint and CI lane run both visual suites.
