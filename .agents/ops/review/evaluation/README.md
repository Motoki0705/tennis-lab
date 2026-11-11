# Private review comparison

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This opt-in harness implements the comparison authorized in
[parent #1797](https://github.com/Motoki0705/tennis-lab/issues/1797#issuecomment-6786941605).
The separately authorized minimum comparison ran on 2026-11-11: fourteen serial
calls, no additional calls, no runtime/format failures and no missing usage.
[Execution evidence and parent adjudication material](../../../../knowledge/reports/review-model-comparison-20261111/README.md)
record the measurements and completed parent adjudication. The result report
contains the parent's policy decision; formal review acceptance remains pending.
Normal pytest runs only offline contract tests; it never runs the comparison.

## Frozen material

[manifest.json](cases/manifest.json) is the snapshot authority: full head, base,
merge base, byte-exact diff fingerprint, cutoff and input fingerprint per case.
All objects must already exist locally. Missing history is an error, with no fetch.
[gold.json](cases/gold.json) contains nine confirmed major issues, historical
parent confirmations and regression evidence, plus independent static checks of
the two provisional negatives. Historical test results are attributed to their
original observers; they were not rerun while preparing this dataset.

| PR | Reviewed boundary | Answer-key evidence |
|---|---|---|
| [#1699](https://github.com/Motoki0705/tennis-lab/pull/1699) | Publication URL credentials and raw-log tracking | Parent resolution, before/after reproduction and ignore-rule check |
| [#1766](https://github.com/Motoki0705/tennis-lab/pull/1766) | Cancellation racing automatic continuation | Parent-confirmed cancellation regression |
| [#1867](https://github.com/Motoki0705/tennis-lab/pull/1867) | Older recovery hiding later findings; event/final mismatch | Parent-confirmed recovery and runtime regressions |
| [#1872](https://github.com/Motoki0705/tennis-lab/pull/1872) | Unconfirmed cleanup and interruption before initial runtime save | Parent-confirmed exclusion and startup-boundary regressions |
| [#1888](https://github.com/Motoki0705/tennis-lab/pull/1888) | Partial coverage settings verification and lost handoff candidates | Parent-confirmed partial-result and failed-final regressions |
| [#1686](https://github.com/Motoki0705/tennis-lab/pull/1686) | Human instruction attribution in memory documentation | Complete frozen diff and referenced instructions inspected independently |
| [#1744](https://github.com/Motoki0705/tennis-lab/pull/1744) | Saved player-view scene bundle | Bundle, assets, CLI, coordinate transform and test assertions inspected independently |

The `inputs/` files are curated projections of the original saved pre-review
contexts. The cutoff is the original runner's start time. Parent requirements,
the selected child's requirements, PR description, changed paths, captured CI
summary and original instructions are retained. Sibling progress and operational
`parent-*` comments are excluded. Later-created or later-edited comments are
excluded, even if their original creation predates the cutoff. Provenance records
the source fingerprint, omitted comment IDs and sanitization count. PR/child
discussions and raw CI logs that were not captured remain unavailable.

Preparation copies only the frozen input and diff. The answer key, fix commits,
current checkout, shared memory and later GitHub comments have no reviewer tools.
Git tools accept pinned revisions and their ancestors. GitHub GET accepts only
exact registered endpoints. Other endpoints fail with `UNREGISTERED_ENDPOINT`;
there is no live client or fallback. The local comment sink cannot publish and is
not exposed as a reviewer tool. Retrieval uses the existing bounded, journaled
tools. Existing formal CLI authorization, publication, exclusion and gate checks
remain unchanged.

The reviewer works in an empty private git repository. An evaluation-only launcher
disables shell, image, browser, apps/plugins, hooks, memory, other agents,
code-mode tools, web search, inherited MCP servers and project instructions. It allows only the registered
`review_evaluation` MCP configuration and shared runner arguments. It fails before
launch if the installed CLI lacks a required restriction. The first CLI version
is pinned for the comparison. These use the official
[Codex configuration controls](https://learn.chatgpt.com/docs/config-file/config-reference).
The pilot and remaining event traces show calls only to the dedicated MCP server.
Effective model and effort were observed; effective service tier remains unavailable.
CPU test execution is deliberately unavailable, so this paired
qualification does not reproduce every tool available to a formal review.

## Preparation and explicit execution

Use a dedicated state outside the checkout and formal parent state. All output,
runtime events, retrieval receipts and local writes stay there with private
permissions. Codex may also retain its usual local CLI transcript.
Do not commit raw runtime/state files.

```bash
export PYTHONPATH=.agents/ops
EVAL_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/review-evaluation/minimum"
.venv/bin/python -m review.evaluation verify --repo .
.venv/bin/python -m review.evaluation init --state-dir "$EVAL_STATE"
.venv/bin/python -m review.evaluation prepare --repo . --state-dir "$EVAL_STATE" --case 1699
```

Before the first call, record the parent's proposed per-case model selection with
`record-selection --state-dir "$EVAL_STATE" --spec selection.json`. The spec has
`schema_version: 1`, a nonempty `source` referring to the parent instruction, and
exactly one `{case_id, model, reason}` choice per approved case. Choices cannot
change after the first reservation. A launch without this record is rejected
before CLI inspection or model startup. The parent must freeze the proposed
selection for condition 3 before spending the pilot calls.

[protocol.json](cases/protocol.json) is the approved numerical protocol. Start
only the initial Astra/Sol pair for its pilot case, one call at a time. The first
measurement can already stop admission on an overrun or failure. After both pilot
calls, `status` recomputes the remaining paired calls and unused reserve from
[historical_usage.json](cases/historical_usage.json), scaled independently by
each model's measured pilot. Unknown usage or an invalid/unverified pilot blocks
further calls. The token envelope is an admission estimate, not an API token cap;
one running call may exceed it. Return to the parent if projected totals exceed it.

Only after the separate execution instruction:

```bash
.venv/bin/python -m review.evaluation run --repo . --state-dir "$EVAL_STATE" \
  --case 1699 --model gpt-6-astra --parent-instruction-file parent-execution.txt --execute
.venv/bin/python -m review.evaluation status --state-dir "$EVAL_STATE"
# Then the Sol pilot, inspect the new forecast, then admit remaining pairs singly.
```

The common runner enforces the approved per-call timeout. Reservations are durable
and serial. A collected paired call is returned idempotently. A reserved call with
files is never restarted; use `collect --run-id ID` after confirming terminal
runtime and cleanup. Missing/unconfirmed runtime keeps the reservation and blocks
new calls. Do not erase it to regain capacity. Extra calls require `--additional`
and `--reason-file`; failed starts and retries consume reserved capacity.
The harness never creates a new formal review, decision capture or GitHub comment.
Code, shared runner dependencies, inputs, answer key and protocol are fingerprinted
at initialization. A changed comparison must return to the parent; creating another
state directory does not reset the approved study's call or token budget.

## Measurements and parent adjudication

`collect` reuses the formal review's saved-final, event, runtime/effective-setting
and schema validators. A runtime failure, a schema failure and unusable output
are separate observations. Output without verified runtime evidence is not scored
as a valid review. Wall time runs from reservation to runtime finish, excluding
manual collection delay; model time uses runner start/finish.

`adjudication-template --state-dir "$EVAL_STATE" --output adjudications.json`
exports findings, path-overlap candidates and pending judgments. The parent checks
the trigger, cause and effect against each gold issue using reproduction, tests or
independent code confirmation. Path overlap alone never establishes a match.
The parent sets `adjudicator: parent`, preserves the exact output and gold
fingerprints, supplies `matches` with `finding_id`, `gold_id`, `reason`, `evidence`,
and classifies other findings as `false`, `unconfirmed` or `valid_new` with reasons
and evidence. An unfilled template is rejected, not interpreted as a verdict.

`report --state-dir "$EVAL_STATE" --parent-adjudications adjudications.json`
produces the following distinct views:

- Paired qualification: all physical Astra/Sol calls, including unsuccessful and
  additional calls. Resource totals count each source counter once.
- Condition 1, `astra_fixed`: measured fresh Astra samples.
- Condition 2, `reuse_fixed`: those same code-review results are the quality
  reference; historical saved-evidence replay measures eligibility separately.
  It does not invent a new model result or measured recovery latency.
- Condition 3, `reuse_select`: the predeclared choices projected from paired
  samples, charging **all** actual additional calls. This counterfactual view is
  not added to physical usage. Combining additional findings requires the parent's
  joint adjudication; per-run true positives must not be summed as a joint score.

Detection/miss counts use distinct gold IDs, with blocker/major breakdowns. Before
parent adjudication, counts are pending rather than zero. A wholly unusable output
misses all gold issues for that case. False findings, finding confirmation failures,
code `not_reviewed`, evidence waits, format failures, runtime failures and their
union are reported separately. The additional-launch rate is episodes with more
than one unique model call divided by observed episodes. An episode is a case/model
for qualification and a case for the selection projection.

Per-turn `agent-runtime` input, cached input and output are separate counters.
Input already includes cached input; reasoning is already included in output.
Cumulative counters and missing values remain excluded/unknown, never zero-filled.
No-AI operations cannot charge the original run again. Identical source counters
are deduplicated, while conflicting counts fail. No price, billing or quota
savings are inferred from token counts. Improvement uses input plus output or
observed model time, after the paired matrix is complete and usage is known.

## Historical reuse evidence and remaining acceptance

[historical_reuse.json](cases/historical_reuse.json) is the read-only audit of
the five repeated calls documented in [#1790](https://github.com/Motoki0705/tennis-lab/issues/1790).
The source and repeat snapshots match. Current saved-evidence validation succeeds,
but every source has `not_reviewed` coverage. Therefore none of these five full
reviews can currently be avoided. Their extra usage/time is recorded as incurred
work, **not** claimed savings. No coverage is promoted and no evidence wait is
invented. The audit excludes present authorization, superseding requests and
publication checks, so historical eligibility never authorizes a formal recovery.
To regenerate from an explicitly supplied original parent state:

```bash
.venv/bin/python -m review.evaluation audit-reuse \
  --parent-state-dir "$PARENT_STATE" --output reuse-audit.json
```

The minimum sample has no gold blockers and only one repetition. Provisional
negatives are not proof of defect absence, model caching/order can affect timings,
and the restricted tool set differs from formal review. The default stays fixed.
Limited selection requires parent-confirmed no extra major misses against Astra,
no worse false/unknown/failure rates, verified effective settings and at least the
approved improvement, including escalation cost. Publication/auth/gate,
cancellation/concurrency and serious data-identity cases stay Astra-first.

After separate authorization, the parent chooses a representative formal PR and
checks prepare/request, pinned retrieval receipts, saved final/events, runtime
settings, result publication, explicit parent receipt and gate dry-run. Confirm
that an unconfirmed result or missing CI blocks gate; complete required evidence
without a new code-review call where possible. Exercise saved-result reuse,
recovery and Astra handoff on applicable authorized runs, preserving source
findings and the explicit causal chain described in
[decision records](../../decision_logs/README.md). Record the real PR/CI/gate
evidence and rollout decision in the parent issue. These are remaining acceptance
tasks, not completed by offline fixtures.

Roll out approved selection only for named low-risk requests with a dedicated
config; retain fixed for other requests. Stop expansion on any extra severe miss,
unexplained setting, lost causal link or exceeded envelope. Roll back subsequent
requests to the existing fixed Astra/xhigh configuration. Frozen or consumed
requests are not edited: reconcile the active run, keep its findings, then create
an explicitly instructed replacement or handoff. No automatic retry or policy
switch is introduced by this harness.

## Offline validation

Run the changed unit file and the two decision-link E2E nodes sequentially with
`pytest -n 0`. Tests use small local git histories and fabricated runtime evidence;
no historical checkout, real model, GPU or live GitHub is needed in CI. The tested
contracts include future-material exclusion, unknown-endpoint refusal, runner
arguments, pilot stop rules, capacity, immutable selection, cache/source accounting
and publication/navigation through the existing decision-log code.
