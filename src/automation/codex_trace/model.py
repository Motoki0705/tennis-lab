"""Typed analysis records emitted by the Codex rollout-trace analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TypeAlias, cast

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class ExactTokenUsage:
    """Provider-reported usage for one inference call."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int


@dataclass(frozen=True)
class Segment:
    """One structural input, model-output, or tool-output attribution unit."""

    segment_id: str
    inference_call_id: str | None
    direction: str
    structural_type: str
    semantic_cluster: str
    cluster_probability: float
    tokens_estimated: int
    token_method: str
    byte_count: int
    item_id: str | None = None
    call_id: str | None = None
    raw_payload_id: str | None = None
    evidence_mode: str = "content"


@dataclass(frozen=True)
class InferenceStep:
    """Analysis of one concrete upstream inference attempt."""

    inference_call_id: str
    thread_id: str
    codex_turn_id: str
    model: str
    provider: str
    started_at_unix_ms: int
    ended_at_unix_ms: int | None
    status: str
    usage_exact: ExactTokenUsage | None
    input_tokens_estimated: int
    input_residual_tokens: int | None
    input_tokens_by_cluster_estimated: dict[str, int]
    reasoning_cluster_probabilities: dict[str, float]
    reasoning_tokens_by_cluster_estimated: dict[str, float]
    reasoning_evidence_mode: str
    request_payload_id: str
    response_payload_id: str | None
    response_item_ids: tuple[str, ...]
    tool_call_ids: tuple[str, ...]


@dataclass(frozen=True)
class ToolCallAnalysis:
    """Runtime and model-visible statistics for one tool execution."""

    tool_call_id: str
    inference_call_id: str | None
    thread_id: str
    codex_turn_id: str | None
    kind: str
    requester: str
    started_at_unix_ms: int
    ended_at_unix_ms: int | None
    status: str
    invocation_tokens_estimated: int | None
    raw_output_tokens_estimated: int | None
    original_output_tokens: int | None
    model_visible_output_tokens_estimated: int
    model_visible_output_evidence: str
    token_method: str
    invocation_payload_id: str | None
    result_payload_id: str | None
    files_touched: tuple[str, ...] = ()
    lines_added: int = 0
    lines_deleted: int = 0


@dataclass(frozen=True)
class ExactTotals:
    """Sum of provider-reported fields; missing-usage attempts are counted."""

    inference_calls: int
    inference_calls_with_usage: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int


@dataclass(frozen=True)
class TraceSource:
    """Identity of the reduced rollout trace used as input."""

    trace_schema_version: int
    trace_id: str
    rollout_id: str
    status: str
    root_thread_id: str
    bundle_path: str


@dataclass(frozen=True)
class AnalysisReport:
    """Complete deterministic analyzer output."""

    schema_version: int
    source: TraceSource
    totals_exact: ExactTotals
    inference_steps: tuple[InferenceStep, ...]
    segments: tuple[Segment, ...]
    tool_calls: tuple[ToolCallAnalysis, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation."""

        return cast(dict[str, JsonValue], asdict(self))
