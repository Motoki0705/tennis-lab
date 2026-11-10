# Explain HTML contract, Gemini editor and local service

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This package provides the shared document/edit contract and durable revision store
for [#1710](https://github.com/Motoki0705/tennis-lab/issues/1710), plus visual import,
managed rendering, an explicit image-generation availability boundary, Gemini
candidate generation and independent semantic checks, and a loopback HTTP/job runtime.
The browser UI, image-provider implementation and browser quality evaluator are
injected separately. It uses existing Pydantic/Pillow/jsonschema/FastAPI/uvicorn
dependencies and Python's standard library.

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
| [parsing.py](parsing.py) | Shared rejection of ambiguous HTML syntax and CSS resource notation |
| [images.py](images.py) | Local raster import, capability reporting and fail-closed image adapter |
| [visual_definitions.py](visual_definitions.py) | Strict observation-error/comparison graph definitions and numerical model |
| [visuals.py](visuals.py) | SVG/HTML/CSS safety, figure markup and offline renderer policy |
| [renderers/visuals-v2.js](renderers/visuals-v2.js) | Current renderer with declared figure targets and isolated graph failures |
| [renderers/visuals-v1.js](renderers/visuals-v1.js) | Immutable renderer retained for old revisions |
| [editor.py](editor.py) | Initial writing, editing, fact preservation and independent semantic review |
| [gemini.py](gemini.py) | Pinned, tool-free agy execution, private evidence and cancellation |
| [no_tools.py](no_tools.py) | Trusted PreToolUse hook admitting only schema-checked completion |
| [jobs.py](jobs.py) | Durable deduplication, document queues, publication fencing and recovery |
| [managed.py](managed.py) | Service wiring for real figure compilation, image unavailability and staged visual checks |
| [service.py](service.py) | Contract-driven HTTP routes, token/origin/host checks and public files |
| [__main__.py](__main__.py) | Local start/serve/status/stop and offline export CLI |

## Identity, targets and candidates

Producers put `data-content-id` on addressable logical blocks. Preserve an ID when
its logical content survives; allocate a new ID for new content. No headings,
layout, colors, block tag names or order are prescribed. HTML must have balanced,
explicit closing tags; duplicate HTML/content/artifact IDs fail. Native HTML `id`
and `data-content-id` have separate namespaces. Figure and evidence references
use `data-figure-id` and `data-evidence-id`; resources may use `data-asset-id`.

The shared parser rejects ambiguous/incomplete comments, processing instructions
and every declaration except the simple HTML5 doctype. Accepted comments use
`<!-- ... -->`, without embedded double hyphens or a trailing hyphen. Inputs such
as `<!-->`, `<!--->`, `--!>` closes, open comments or bogus `<!...>` declarations
fail with `invalid_document`; neither the document nor visual validator repairs
them. Both validators also reject CSS resource notation, including `url()`,
`image-set()`/`-webkit-image-set()` string URLs, `image()` and `src()`, in style
attributes and style elements. Unsupported CSS is refused rather than omitted
from reference ownership/change accounting.

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

## Gemini writing and editing

With `PYTHONPATH=.agents/ops`, construct `Gemini(private_root, agy=...)` from
`explain_html.gemini`, then `GeminiEditor(gemini, compile_document=compiler)` from
`explain_html.editor`. The required keyword argument has type
`DocumentCompiler = Callable[[Document], Document]`. Supply a trusted compiler that
expands figure slots, inserts the renderer/CSP and returns the final Document,
or raises ContractError. This package does not choose a visual implementation.
Even without figures, explicitly choose identity with
`GeminiEditor(gemini, compile_document=lambda document: document)`; omission is an
error. The callback receives a deep copy and its return value is revalidated and
detached from mutable callback state. Keep generated asset bytes privately in the
calling controller for staging; the compiler must not publish or commit files.
For managed visuals, pass `compile_document=render_document` from
`explain_html.visuals` directly; its public signature already matches DocumentCompiler.
`write_initial(request, context, *, writer_timeout=900, verifier_timeout=240, cancel=None)` and
`edit(request, current, context, *, writer_timeout=900, verifier_timeout=240, cancel=None)` return an
`EditorialResult`. Use the shared EditRequest/Revision types; initial requests
have a null base and a document target. Each stage has its own budget including
effort/version checks; the writer never consumes the verifier's allowance.
The 2026-11-09 qualification measured initial writers at 270–360 seconds, a
restructure writer at about 633 seconds and ordinary verifiers at 60–110 seconds.
The defaults therefore allow 900 seconds for writing and 240 for verification.
Figure/slot correspondence errors alone receive one independent writer repair
turn, capped at 600 seconds. Its complete context includes the rejected candidate,
figure IDs, expected content IDs and missing-slot/metadata/wrong-owner diagnostics.
Both the repaired result and the compiled result undergo all normal checks;
verification binds to the compiled digest and staged rendering is still mandatory.
An unsuccessful repair stops; semantic, safety and other errors do not trigger
retries. Prompts record the fixed attempt/time limit and `EditorialResult.repairs`
records repair model/effort/time/usage separately from writer/verifier evidence.
The maximum model-stage allowance is 29 minutes with repair (19 without), plus
local compilation/rendering. The UI's cancellable elapsed-time display promises no
completion deadline and must not impose the old shared 600-second limit.
Each turn uses a fresh workspace/conversation. There is no resume or model fallback.

`WritingContext` is trusted controller input: purpose, audience, exact desired
evidence, approved assets, prepared figures, writing preferences and all prior
change summaries. For edits, every difference between current and desired evidence
also requires an `authorized_evidence_changes` declaration. Do not derive this
authorization from the writer or accept it unchecked from browser JSON. The
independent verifier checks its justification against the instruction. The complete
context, current revision, request/target/instruction and history go to both calls;
the verifier additionally receives the candidate and its digest. Prompts and schemas
travel through private files/stdin without an argv size limit or silent truncation.
Writer/verifier schemas are generated directly from their typed models in `editor.py`.
Writer inputs also include `candidate_template`: the complete request and trusted
metadata in the actual output shape, with HTML/summary placeholders. This reduces
native finish errors such as ID-only evidence or invented request fields, without
constraining prose/layout or changing the strict output schema. The independent
verifier receives the compiled candidate, not this writing aid.

The writer returns `WriterOutput`: only `request`, the complete `document`, and
`change_summary`. It reports neither content nor evidence changes. Mechanical
checks bind its request, run structural/reference checks, and require exact purpose, audience,
evidence metadata (including numbers, conditions and provenance) and approved assets.
Surviving rendered evidence references and HTTP(S) source hyperlinks must remain.
Numeric values in unchanged evidence that appeared in that evidence's rendered text
must still appear in its scope; this lexical decimal check does not prove units,
visual visibility, paraphrases or meaning. After these writer checks, the required
compiler may add generated asset metadata and expand the HTML. The editor then
builds the public `Candidate` using `content_changes(current, compiled_document)`;
there is no raw change list to compare or replace after slot expansion.
For edits, `evidence_changes` comes from the checked trusted
`authorized_evidence_changes`, including its human-readable reasons. Initial
evidence receives runtime-generated `added` records with reason
`Supplied initial evidence.`. The context already determines identities, kinds
and reasons, so the writer supplies none of this accounting. The independent
verifier still checks reasons against the instruction. Structure, change integrity
and fact preservation are checked again on the final Candidate; stage/commit
independently recompute differences as before. The originating request and summary are retained.
Only this final candidate is digested and sent to a separate Gemini call, which
reviews those points, logical claims, authorized changes and missing inputs.
Honest explanations of insufficient information are supported. The review must cover every old/new
evidence ID, match the candidate digest and pass without issues. Rejection or invalid
output raises ContractError without returning a candidate. This model review adds
evidence; it does not guarantee truth or complete semantic correctness.

`schema.py` and the public Candidate/Revision, HTTP, UI and stored history contracts
are unchanged. Their change lists describe the final compiled document. Existing
revisions require no migration. Old writer responses containing `changes` or
`evidence_changes` fail the new strict schema; regenerate with the new writer
schema/prompt rather than replaying a legacy response as `WriterOutput`.

Local scope retains its existing boundary: `validate_request()` checks that the
current content/selection target exists and matches, not that every changed block
is inside it. There is no deterministic prohibition on all edits outside a target.
The writer is instructed to edit near it; the independent verifier receives the
request and final computed differences and checks the answer and preservation of
unrelated important content. A semantic rejection preserves current. This does
not provide a strict mechanical guarantee of local scope.

The writer distinguishes supplied evidence, explicit user-supplied facts and
general definitions/derivations. A topic request or a citation URL alone supplies
no product specifications, sport speeds or empirical accuracy. Unsupplied numeric
inputs, product names and performance claims are prohibited; calculations from
supplied inputs retain their assumptions and hypothetical/derived status locally.
Missing inputs are explained in the page rather than guessed.

`SemanticVerdict` in [editor.py](editor.py) requires `unsupported_claims` and
`technical_errors` arrays, including when empty. Each finding contains `quote`,
`reason`, and optional `content_id`/`evidence_id`. The verifier inspects the whole
document, including untagged prose, captions and graph definitions, for unsupported
concrete claims, confused terminology, coordinate frames and unjustified absolute
conclusions. Each finding becomes a `ValidationIssue` with rule `unsupported_claim`
or `technical_error` in `ValidationResult.semantic.issues`; **any** finding vetoes
approval even if the model's overall check says passed. Semantic rejection raises
`SemanticValidationError` (a `ContractError` with the same public `validation_failed`
code); trusted orchestration can retain its digest-bound `.validation` for private
diagnostics. No candidate is returned, and diagnostic text is not an HTTP error.
`schema.py`, stored ValidationResult shape and existing revisions are unchanged.
Old model verdicts missing these arrays are invalid: rerun verification with the
new prompt/schema instead of supplying empty defaults or reusing old approval.
There is no semantic repair/retry loop or relaxed check after repeated rejection;
retain current and require an explicit retry with corrected inputs. The bounded
figure repair described above is unchanged. This review cannot guarantee complete
correctness and performs no external fact-checking.

`EditorialResult` contains candidate, validation, writer and verifier evidence.
The validation has structural/semantic passed and rendering **not_run**; it cannot
authorize commit by itself. The service must stage exactly that compiled candidate
with its generated asset bytes, apply independent visual/renderer/rendering checks
to that same bundle and merge the actual rendering verdict into the digest-bound
ValidationResult. Never simply mark rendering passed. Any transformation or
sanitization that changes HTML,
assets or metadata belongs in compilation, before semantic review. Changing the
returned candidate requires new verification; never replace a digest to reuse an
old verdict. Compiler rejection returns no EditorialResult and leaves publication
to the controller. This editor never opens the store or public files.

The model/effort are explicitly `gemini-3.8-flash-high` / `high`. Constructor
overrides that differ fail. The agy version comes from the existing AGY_VERSION
pin in agents.conf; the issue-supplement runtime and its pins/permissions are
unchanged. Every turn records selected settings, runs the read-only headless
`/effort` check (requiring `command.data.current=high`), then verifies the generation
stream's model and all backend model-override log labels. The
[official effort contract](https://antigravity.google/docs/cli/commands/effort/)
defines effort as a model-variant selection; the matching High backend variant
and explicit effort response are the evidence, not a model's self-report or token
count. A missing/conflicting observation fails with model_mismatch.

Use a dedicated private_root outside served/exported directories. Its directory
must be private (0700); files are 0600. Prompts, stdout, settings, schemas, hooks,
errors and raw CLI logs remain there. Only the returned numeric usage/timing and
validated configuration may be summarized publicly; exception causes and raw logs
are private. Session strict permissions and the hook admit only agy's native
`finish` tool after validating its arguments against the pinned output schema.
This control operation returns structured data; filesystem edits, shell commands,
research and delegation remain denied. The admitted completion must match both the
final structured output and its completed stream step. Any operational tool attempt
disqualifies the result. Hook/settings/schema integrity is checked on success and failure.
This policy is not an OS sandbox, and does not qualify arbitrary global plugins
or hooks. Use an operator-controlled CLI profile. The adapter does not copy
credentials or rewrite user settings; agy retains its own normal profile behavior.

Pass a `threading.Event` as cancel. The service sets the same event on request
cancellation or shutdown; `opsproc` stops/reaps the process group before returning
cancelled. A pre-set event prevents launch. Timeout, cancellation, model mismatch,
invalid model output and semantic failure use the existing public error codes.
The controller must still synchronize cancellation with its later commit.

## Local service and trusted integrations

With `PYTHONPATH=.agents/ops`, construct
`JobController(store, editor_factory, validator, prepare=...)` and
`create_app(controller, port=actual_bound_port, ui_files=...)`. The CLI always binds
**127.0.0.1**, defaults to an OS-assigned port, and has no wildcard/LAN bind option.
One controller holds a nonblocking filesystem lock on the runtime root until all
workers return. Different documents can generate concurrently; one document's
requests run serially. A queued request with a stale base fails before generation.
Status reads never start work.

`BundleValidator` is required. It receives the immutable staged Candidate, **all**
staged asset bytes keyed by asset ID and the cancel Event; it must check HTML/SVG
safety, renderer policy and actual rendering, then return a CheckResult. The
controller binds this result to the editor's exact digest and commits only when
all checks passed. There is no built-in rendering validator or default passed
verdict. A visual safety/figure compiler alone does not prove browser rendering.
Compile figures before independent semantic review; later HTML changes invalidate
that review's digest.

`EditorFactory(prepared)` creates an editor for each request. The CLI wires
[ManagedResources](managed.py) by default: the editor compiles through the real
`visuals.render_document` before independent semantic review. It fills figure
slots, inserts CSP and the pinned managed script; there is no identity fallback.
Optional `--compiler module:factory` is an additional trusted transform **before**
managed compilation, not a replacement for the visual/image policy. It receives
the request's mutable `prepared.assets` map and returns DocumentCompiler. Put new
SVG/renderer bytes in that map; staging and validation use those same bytes.
Each request owns its transform and map, so documents cannot mix resources.

The default `ManagedResources.prepare` preserves current evidence/assets and all
change summaries, and passes the real `UnavailableImageGenerator.capability()`
reason to writing and review. A new AI-image requirement is represented through
the existing Figure contract: kind=image, generated_concept provenance, empty
asset_ids and an empty figure slot. The compiler passes this unmet image intent
to the unavailable image adapter through `generate_image`, producing
`image_generation_unavailable` before review/commit. Existing image moves and
caption changes retain asset IDs. Instruction interpretation remains Gemini's
responsibility; service code does not match language-specific keywords. Custom
`--prepare` integrations must retain this typed unmet-intent convention.

The CLI also runs real `validate_visuals` on every staged byte before invoking
the supplied BundleValidator. The latter must provide the actual rendering check;
visual safety alone is never marked rendering passed.

Run with the repository Python. Replace the example validator with an
operator-controlled module implementing BundleValidator:

```bash
export PYTHONPATH=.agents/ops
.venv/bin/python -m explain_html start --root outputs/explain-html --document triangulation --validator my_rendering:validate_bundle
.venv/bin/python -m explain_html status --root outputs/explain-html
.venv/bin/python -m explain_html stop --root outputs/explain-html
.venv/bin/python -m explain_html export --root outputs/explain-html --document triangulation
```

`serve` runs in the foreground with the same options as `start`. `--agy` selects
the adapter's pinned executable without changing its timeout/model/effort rules.
`--prepare module:callable` and `--ui module:callable` load trusted local code; the
latter returns a closed map of fixed UI filenames to bytes. No browser request
chooses these modules. Without `--ui`, the root shows a service-ready placeholder.
`--document` validates an existing document and includes its ID in the launch fragment.
`start` waits for HTTP startup and prints a localhost URL. `status` checks
PID/boot/start identity. `stop` signals that identity through a Linux pidfd and
waits for shutdown and process recovery. `export` needs no service and prints the
snapshot directory; `--revision` selects an old published revision.

The launch URL gives the **session token in a fragment to the fixed UI only**.
Consume/remove it and keep it in control-UI memory; attach the contract's
`X-Explain-Token` header to JSON APIs. Never put it in generated HTML, iframe URLs,
query strings, cookies or exports. Every process gets a fresh token. Host is
limited to localhost/127.0.0.1 with the actual port. Supplied Origin must match;
opaque `null` cannot call JSON APIs. CLI clients may omit Origin but still need
the token. Public document/export files allow opaque-origin resource requests
without credentials. CSP enforces a sandbox without `allow-same-origin`, blocks
connections and admits only revision-pinned renderer hashes. Fixed UI uses a
separate CSP.

Routes, success statuses and public errors come directly from `contract_schema()`.
Submit returns 202 immediately; poll the returned request ID for JobResponse.
There is no SSE/WebSocket channel. An identical request ID/content returns saved
state/results, including terminal failures; different content conflicts. Explicit
retry needs a new ID and the displayed current base. Job failures live in
JobResponse.error; admission rejection uses ErrorResponse. Undo creates a new
revision and fences active jobs with stale_revision. Cancel cannot undo a
committed success.

Cancel, undo, validation entry and commit share a controller lock. Shutdown sets
the same Event passed to preparation/editor/validator, marks active work
interrupted and joins workers before releasing ownership. Integrations must honor
cancellation and bound their rendering work. Restart marks queued/running/
validating records interrupted without resend. A commit before acknowledgement
is reconciled against the exact saved stage and published revision as succeeded.
Uncertain journal I/O stops editing with an explicit error until restart; it
cannot relabel a committed revision as failed/interrupted.

Undo also reconciles publication before recording a failure: request ID, parent,
restored_from and the complete target document must match. Startup checks restore
journals even when an older runtime recorded an error. An unreadable publication
stops further work without declaring success. Export similarly reports success
after a rename/fsync error only if its manifest and every public byte verify.

## Browser reader and editor

Add `--ui explain_html.web:files` to `start` or `serve` to use the fixed reader/editor.
Open the launch URL including its fragment. The UI immediately removes the fragment,
keeps the token only in memory and sends it only on authenticated JSON APIs. A full
page reload needs the launch URL again; pending request IDs (operation, base and start
time, without the token or instruction) remain in this tab's sessionStorage for
reconciliation. Reconnection polls saved IDs and never resubmits an edit or undo.
If an undo response is lost, a missing history row is inconclusive. Keep its request
ID pending, continue checking published history and offer the existing status recheck
action while the displayed revision remains readable.

Use the target menu for the whole page or a content block. Click a heading/figure,
press Enter or Space on it, or select text inside one block to choose a precise target.
Selection quotes and UTF-16 positions are checked against the displayed revision before
submission. The fixed UI contains the instruction, cancel, elapsed time, public errors,
revision history, old-version preview and restore controls. Answers appear in the document.
Retry is an explicit action with a new request ID and the displayed current base. A stale
revision keeps the displayed page and offers an explicit current-version refresh.

`GET /api/documents/{document_id}/revisions/{revision_id}/content/reader.html` is a
presentation-only view of an already published revision. This small service extension is
necessary because an opaque iframe cannot expose its DOM to the control UI, and srcdoc
inherits the control CSP. The view adds minimal accessibility styles and the fixed `web/reader.js` bridge, and authorizes
its exact bytes with a script hash, alongside revision-pinned renderers. Its HTTP CSP
replaces CSP meta elements in this temporary response; the sandbox, connect-src none,
resource restrictions and renderer hashes are retained. The common content filename enum advertises reader.html; JSON wire models and the export file list stay unchanged. Stored HTML, validators, compiler,
`content/index.html` and export bytes do not change.

The iframe has `sandbox="allow-scripts"`, an opaque origin, no same-origin, forms,
popups or top-navigation permission, and no token or cookies. Generated scripts remain
blocked. The control UI accepts only `ready`, `selection` and reading `state` messages from
the exact frame window, origin `null`, current document/revision and per-view nonce;
selection IDs/quotes/positions and state shape are checked against that revision. None
can send instructions. The reader projection uses the shared StrictHTMLParser.
Target indexing uses an inert template; no parsed nodes enter the control DOM. The
control CSP may report rejected inline style attributes from app.js during this
parse. Keep that policy strict; document styles apply only inside the content iframe.
Parent-to-frame messages use target origin `*` because the receiver
is opaque and contain only connection, restore/highlight or navigation data. The bridge
checks parent window and its HTTP origin; it cannot read a token or call the service.

During a job the old frame remains readable. A successful published revision loads in a
second frame and replaces it only after bridge initialization. Unchanged blocks preserve
details and graph input values when their HTML, figures and referenced asset metadata
match, including a details element that is itself a content block. The reading anchor
is the innermost visible content block containing the viewport top; its relative offset is restored without focusing or scrolling
the outer page. Changes receive a quiet inset mark; navigation happens only through an
explicit button. A removed anchor is reported and the old scroll height is clamped to the
new document. A removed/changed target asks the reader to choose again before sending.

History previews are read-only until returning to current or restoring an old document
as a new revision. ZIP export downloads the service's exact snapshot files. It excludes
the operation UI, bridge, session credentials and request records; extract it and open
index.html to read the static explanation, images and supported graphs offline. AI image
generation is available only during initial creation; reading-time requests can use SVG,
graphs and existing images, with image_generation_unavailable reported as incomplete.

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
<root>/private/requests/<document_id>/<request_id>.json
<root>/private/{service.lock,service.log,session.json,gemini/...}
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

Use `managed_renderer()` for its Asset/bytes and Renderer version `2`. Graph
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
The CSP insertion offset comes from parsing the unique head before body; literal
head text in comments cannot redirect it. The policy precedes the head's first
script, style and resource reference. Noncanonical head/body nesting and
non-ASCII whitespace before head (including encoded NBSP) are rejected.

The compiler owns CSP, renderer script and `data-visual-style` containment CSS.
The writer is explicitly instructed to omit these, even on edits; copying code
or hashes is unnecessary. An equivalent CSP meta is accepted only if parsed
directive names and token sets equal the required policy, then replaced by the
canonical policy. Attribute order/quotes/case of http-equiv, directive order and
whitespace may differ. Duplicate directives/policies, extra directives, changed
sources and weakened policies are rejected. Existing exact managed script bytes
are replaced; changed code, extra attributes and arbitrary scripts are rejected.
Existing containment CSS must be canonical, otherwise it is rejected. These are
declared compiler transformations before semantic review, not silent repairs.
Canonical staged output remains mandatory. Managed CSS confines wide source
tables and plots to horizontally scrollable figures at 390px, even with global
writer rules such as `table{min-width:750px}`.

Formulas use Chromium's native presentation MathML, without CDN or dependencies.
The writer prompt and visual allowlist share the supported MathML elements;
annotation-xml, mglyph, links, foreign content and event attributes are refused.
TeX commands/delimiters are rejected instead of publishing unrendered formulas.
Accessible explanations accompany MathML; static snapshots retain the same bytes.

The TeX text check covers body prose, including figure captions. It rejects
`$...$`, `$$...$$`, `\(...\)`, `\[...\]` and every backslash followed by ASCII
letters (not just known command names). Text is joined across inline elements,
comments and expanded character references within each text run, so splitting a
command or dollar pair does not evade the check. Block elements and literal
regions end the run; separate paragraphs are not joined.
The descendants of `code`, `pre`, `kbd`, `samp` and `math`, plus attribute values
and head metadata, are excluded from this prose check. Use literal elements for
source notation and paths. Native MathML and HTML/CSS/script/attribute safety
checks still apply in those regions; this exception grants no extra markup rights.
A lone single dollar is literal. Adjacent dollar signs that both introduce numeric
prices, such as `$5 and $10.50`, are currency, not a math pair: amounts may have a
sign and decimal/grouping separators and must end at whitespace, punctuation or
the end of the run (a hyphen before another price also separates amounts).
`$5$`, `$x$2` and a numeric formula after a price are still rejected. Ambiguous
literal notation should use a literal element. Invalid prose raises
`invalid_document` in compilation and staged visual validation without changing
text or reusing a previous verdict; there is no TeX-specific automatic retry.

Drawing attributes (`data-managed-*`, `data-definition`, `data-plot`, `data-values`,
`data-slider-value`, `data-render-*`) are reserved for canonical graph figures;
`data-graph-targets` belongs only to the managed script. Version 2 reads that
script's figure/definition manifest, processes only direct children of declared
figure slots and compares their configuration to the manifest. Extra matching
nodes are never discovered as graph inputs. A damaged target produces an explicit
graph error while other declared graphs continue to initialize.

Version 1 bytes remain unchanged. `managed_renderer("1")` supports the old
Asset/Figure records for snapshot and restore; the current default is version 2.
Use one renderer version per document; mixed versions fail explicitly. Future
implementations need a new file/ID/version. Serve old revision HTML and assets
from that revision, never inject today's code. Runtime checks are defense in
depth; they do not replace publication validation or iframe isolation.

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

AI-generated images are created **only while authoring the initial document**,
using interactive Codex imagegen and local import. During reading, the controller
uses `UnavailableImageGenerator`: `capability()` reports unavailable with the
initial-only policy as its reason, and a new AI-image request raises the shared
`image_generation_unavailable` error. The controller must leave the **whole image
request incomplete**, abort publication and retain current. Do not publish a
partial text/SVG substitute as successful image generation. Existing-image import,
relocation/caption changes, SVG and graph updates remain independently supported.
This is the [confirmed human decision](https://github.com/Motoki0705/tennis-lab/issues/1710#issuecomment-6732169335);
there is no pending service-adapter choice in this scope.

`generate_image(..., adapter=...)` retains the tested `ImageGenerator` boundary
(capability and generation returning real bytes with a public source). Injected
adapters exercise success/unavailable/failure contracts; the browsing service does
not select a generation provider. Provider/invalid-artifact failures raise
`asset_generation_failed`, without provider details in public errors, and must
also leave the whole request incomplete with current preserved. No credentials
or provider are auto-discovered. Interactive imagegen availability does not imply
local-service generation capability; the investigation evidence is in the PR.

## Downstream responsibilities

Use the runtime's HTTP/publication boundaries above instead of writing public
files directly. The store's lock/base comparison also protects against trusted
external writers. The UI owns sandboxed presentation, strict selection-message
validation, change navigation and the static-export editing-unavailable indicator.
Bind the displayed page and all asset URLs to one revision; refresh only after a
succeeded job identifies a new revision.

## Verification

Run the repository `.venv/bin/python -m pytest` on
`tests/unit/agents_ops/explain_html` and
`tests/integration/agents_ops/explain_html`, with `--collect-only` to count cases
and `--durations=0 --durations-min=0` to measure them. Contract/store tests use
real temporary files, asymmetric blocks, multiple assets,
declarative graph data, exception injection and synchronized competing writers.
They require no external API, GPU, browser or real-time sleep. The visual addition
has 9 Python unit and 3 Node cases; run Node with
`node --test tests/unit/agents_ops/explain_html/visuals.test.mjs`.
The two existing restore/snapshot integration cases also use real raster/SVG and
managed graphs; an additional integration verifies v1 snapshot/restore after v2
publication and pins the original v1 bytes. Browser verification uses one shared Chromium setup and two
scenarios (offline slider/comparison/CSP placement and rejected syntax/SVG/
reserved-attribute export with current preservation, damaged-target isolation,
JS-disabled initial rendering and old-v1 replay):

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chromium \
  node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

The harness prints setup/scenario times and its temporary screenshot directory.
No real image generation or external API is run in normal tests/CI.
The existing JavaScript pytest entrypoint and CI lane run both visual suites.

Editor coverage adds 11 integration cases
in `test_editor.py` and 3 public-entry cases in
`tests/e2e/agents_ops/test_explain_html_editor.py`: executable fake agy, a real
revision store, complete long inputs, permitted/prohibited fact changes, rejected
or malformed verdicts, actual hook denial, and pipe/Event-synchronized process
cleanup. One case connects the real `render_document` and staged `validate_visuals`
to editor verification and store commit. These editor tests require no browser or
fixed real-time sleep.

Runtime coverage adds 17 real-store integration cases in `test_jobs.py` and 6
HTTP/CLI cases in `tests/e2e/agents_ops/explain_html/test_service.py`. Explicit fake
editor/rendering boundaries, Event gates, OS-assigned ports and private temporary
roots cover duplicate calls/commits, request-scoped compiled bytes, stale bases, cancel/undo/shutdown late
results, active-journal recovery, disk failure after pointer replacement,
cross-session tokens, Host/Origin/path rejection, closed exports and public CLI
startup/shutdown/restart/offline export.
Restore coverage includes faults before/after pointer replacement, unreadable
publication and old error journals. Three store cases also cover export rename
faults and failed readback. These faults intercept actual directory fsync calls.
