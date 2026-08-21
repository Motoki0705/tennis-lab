# Codex rollout-trace analyzer

This package turns one local Codex `rollout-trace` bundle into an inference
timeline with three deliberately separate measurements:

1. provider-reported exact totals for each `InferenceCall`;
2. explicitly estimated token attribution for request fields and conversation
   items;
3. raw versus model-visible tool output, including terminal truncation evidence
   and `apply_patch` file/line statistics.

It does not upload trace contents or call a model. The default semantic
classifier is a deterministic baseline whose probability and evidence source
are included in every result; it is not presented as ground truth.

## Primary-source contract

The implementation targets reduced schema version `1` from OpenAI Codex tag
`rust-v0.144.1` (commit `44918ea10c0f99151c6710411b4322c2f5c96bea`).
It is also smoke-tested against a real schema-version-`1` bundle emitted by
Codex CLI `0.149.0`; unknown fields are tolerated for this forward-compatible
case.
The relevant upstream sources are:

- [rollout-trace overview and invariants](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/rollout-trace/README.md)
- [`RolloutTrace` root graph](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/rollout-trace/src/model/mod.rs)
- [`ConversationItem`, `InferenceCall`, and per-inference usage](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/rollout-trace/src/model/conversation.rs)
- [tool and terminal runtime objects](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/rollout-trace/src/model/runtime.rs)
- [sequenced raw event envelope](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/rollout-trace/src/raw_event.rs)
- [inference request/response recording](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/rollout-trace/src/inference.rs)
- [serialized Responses request fields](https://github.com/openai/codex/blob/44918ea10c0f99151c6710411b4322c2f5c96bea/codex-rs/codex-api/src/common.rs)

These sources correct a few easy-to-miss simplifications: a conversation
`channel` is optional, `produced_by` is an array, tool turn IDs are optional,
and code-mode/MCP calls carry additional correlation IDs. The loader accepts
unknown fields but rejects missing required structures and unknown schema
versions instead of silently guessing.

## Capture

Tracing is opt-in and must be enabled when the Codex root session starts. Use a
directory that is not shared or committed:

```bash
mkdir -p .codex-rollout-traces
CODEX_ROLLOUT_TRACE_ROOT=/absolute/path/to/tennis-lab/.codex-rollout-traces codex
```

Root and spawned agent threads share one trace bundle. Each bundle contains
`manifest.json`, append-only `trace.jsonl`, and `payloads/`. The analyzer runs
`codex debug trace-reduce` automatically when `state.json` is absent. Pass
`--no-reduce` to require a pre-reduced input.

## Analyze

Run from the repository root:

```bash
.venv/bin/python -m src.automation.codex_trace \
  .codex-rollout-traces/trace-TRACE_ID-ROOT_THREAD_ID \
  --json outputs/codex-trace/report.json \
  --sqlite outputs/codex-trace/report.sqlite \
  --html outputs/codex-trace/report.html \
  --png outputs/codex-trace/report.png
```

Existing outputs are not overwritten unless `--force` is explicit. The
terminal summary reports exact totals and the exact-minus-estimated input
residual. The JSON keeps the complete analysis record. SQLite provides:

- `inference_steps`: one concrete upstream request attempt, exact usage,
  cluster probabilities, attribution, and residual;
- `segments`: request components, model output items, and raw tool results;
- `tool_calls`: invocation/result estimates, pre-truncation terminal count when
  reported, model-visible output estimate, and patch statistics;
- `metadata`: source identity, exact totals, and warnings.

The HTML report is self-contained and uses inline SVG with hover tooltips. It
contains exact-token, estimated input-cluster, reasoning-cluster, tool-output,
and execution-timeline graphs plus summary tables. The PNG contains a compact
four-panel version for PR descriptions. Neither visualization includes raw
prompts, raw tool bodies, or the local trace-bundle path.

The repository includes a non-sensitive real-session example with three
inference calls and two sequential read-only tool calls:

- [self-contained HTML report](../../../assets/codex_trace/session-01a0248f-report.html)
- [static PNG dashboard](../../../assets/codex_trace/session-01a0248f-report.png)

For direct tools, model-visible output is measured from
`model_visible_output_item_ids`. For a code-mode cell with exactly one nested
tool, the analyzer can attribute the cell output to that tool and records
`single_tool_code_cell_output` as the evidence. Cells with multiple nested
tools remain unassigned and emit a warning rather than duplicating tokens.

`tokens_estimated` uses `compact_json_utf8_bytes_div_4`. It is intentionally
named in every record. It never replaces `input_tokens`, `output_tokens`, or
`reasoning_output_tokens` reported by the provider.

## Semantic clusters

The default classifier uses readable reasoning summary/content plus response
actions. When reasoning is opaque, `reasoning_evidence_mode` becomes
`response_action`; with no usable signal it becomes `unclassified` and assigns
the inference to `other`. Cluster-attributed reasoning is calculated as:

```text
reasoning_tokens_by_cluster[c]
  = usage.reasoning_output_tokens * cluster_probability[c]
```

Supply a task-specific taxonomy with `--cluster-rules clusters.json`:

```json
{
  "clusters": {
    "schema_research": {
      "keywords": ["schema", "serializer", "trace"],
      "tool_kinds": ["web", "exec_command"]
    },
    "implementation": {
      "keywords": ["implement", "patch", "refactor"],
      "tool_kinds": ["apply_patch"]
    }
  }
}
```

Empty clusters and the reserved name `other` are rejected explicitly.

## Measurement boundaries

- Tool-call arguments and patch text are model output and tool-runtime input.
- A raw tool result is external output. It becomes model input only when it is
  present in a later inference request.
- `original_token_count` is retained separately from the formatted/model-visible
  terminal output estimate.
- Provider input usage includes instructions, input items, tool definitions,
  and protocol serialization. Per-field re-tokenization therefore has a
  residual and must not be forced to equal the exact total.
- Cached input usage is exact only as an inference total; no item-level cache
  hit mask exists in the trace.
- Completed response items are recorded, not every stream delta. Token-level
  generation timestamps and within-item cluster-switch positions cannot be
  recovered.
- Readable reasoning content is available only when the provider returned it.
  Encrypted reasoning is preserved as opaque evidence and is not decrypted.

Trace payloads can contain prompts, responses, tool arguments/results,
terminal output, repository paths, and source code. `.codex-rollout-traces/` is
ignored by this repository, but generated JSON and SQLite reports must also be
handled as sensitive local artifacts.
