"""Convert a reduced Codex rollout trace into token-attribution timelines."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from src.automation.codex_trace.bundle import TraceBundle, TraceBundleError
from src.automation.codex_trace.classifier import Classification, SemanticClassifier
from src.automation.codex_trace.estimator import TOKEN_METHOD, estimate_value
from src.automation.codex_trace.model import (
    AnalysisReport,
    ExactTokenUsage,
    ExactTotals,
    InferenceStep,
    Segment,
    ToolCallAnalysis,
    TraceSource,
)

ANALYSIS_SCHEMA_VERSION = 1
_FILE_READ_COMMAND = re.compile(
    r"(?:^|\s)(?:cat|sed|rg|grep|head|tail)\s|(?:^|\s)git\s+show\s",
    re.IGNORECASE,
)
_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
_DIFF_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


class TraceAnalyzer:
    """Analyze exact inference totals and estimated sub-item attribution."""

    def __init__(
        self,
        bundle: TraceBundle,
        *,
        classifier: SemanticClassifier | None = None,
    ) -> None:
        self.bundle = bundle
        self.classifier = SemanticClassifier() if classifier is None else classifier
        self._warnings: list[str] = []
        self._conversation_items = _object_map(
            bundle.state["conversation_items"], "conversation_items"
        )
        self._inference_calls = _object_map(
            bundle.state["inference_calls"], "inference_calls"
        )
        self._code_cells = _object_map(bundle.state["code_cells"], "code_cells")
        self._tool_calls = _object_map(bundle.state["tool_calls"], "tool_calls")
        self._terminal_operations = _object_map(
            bundle.state["terminal_operations"], "terminal_operations"
        )
        self._tool_to_inference = self._build_tool_to_inference()
        self._item_to_tool = self._build_item_to_tool()

    def analyze(self) -> AnalysisReport:
        """Build a deterministic report ordered by trace execution sequence."""

        inference_steps: list[InferenceStep] = []
        segments: list[Segment] = []
        for inference_id, inference in sorted(
            self._inference_calls.items(), key=self._inference_sort_key
        ):
            step, step_segments = self._analyze_inference(inference_id, inference)
            inference_steps.append(step)
            segments.extend(step_segments)

        tool_analyses: list[ToolCallAnalysis] = []
        for tool_id, tool in sorted(self._tool_calls.items(), key=self._tool_sort_key):
            analysis, tool_segment = self._analyze_tool(tool_id, tool)
            tool_analyses.append(analysis)
            if tool_segment is not None:
                segments.append(tool_segment)

        usages = [step.usage_exact for step in inference_steps if step.usage_exact]
        totals = ExactTotals(
            inference_calls=len(inference_steps),
            inference_calls_with_usage=len(usages),
            input_tokens=sum(usage.input_tokens for usage in usages),
            cached_input_tokens=sum(usage.cached_input_tokens for usage in usages),
            output_tokens=sum(usage.output_tokens for usage in usages),
            reasoning_output_tokens=sum(
                usage.reasoning_output_tokens for usage in usages
            ),
        )
        state = self.bundle.state
        source = TraceSource(
            trace_schema_version=_required_int(state, "schema_version", "trace"),
            trace_id=_required_str(state, "trace_id", "trace"),
            rollout_id=_required_str(state, "rollout_id", "trace"),
            status=_required_str(state, "status", "trace"),
            root_thread_id=_required_str(state, "root_thread_id", "trace"),
            bundle_path=str(self.bundle.path),
        )
        return AnalysisReport(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            source=source,
            totals_exact=totals,
            inference_steps=tuple(inference_steps),
            segments=tuple(segments),
            tool_calls=tuple(tool_analyses),
            warnings=tuple(self._warnings),
        )

    def _analyze_inference(
        self, inference_id: str, inference: dict[str, Any]
    ) -> tuple[InferenceStep, list[Segment]]:
        execution = _required_object(
            inference, "execution", f"inference {inference_id}"
        )
        request_payload_id = _required_str(
            inference, "raw_request_payload_id", f"inference {inference_id}"
        )
        raw_request = _request_object(
            self.bundle.payload_json(request_payload_id), inference_id
        )
        request_item_ids = tuple(
            _string_array(inference.get("request_item_ids"), "request_item_ids")
        )
        response_item_ids = tuple(
            _string_array(inference.get("response_item_ids"), "response_item_ids")
        )
        tool_ids = tuple(
            _string_array(
                inference.get("tool_call_ids_started_by_response"),
                "tool_call_ids_started_by_response",
            )
        )

        input_segments = self._input_segments(
            inference_id,
            request_payload_id,
            raw_request,
            request_item_ids,
        )
        output_segments = self._output_segments(inference_id, response_item_ids)
        input_estimated = sum(segment.tokens_estimated for segment in input_segments)
        by_cluster: defaultdict[str, int] = defaultdict(int)
        for segment in input_segments:
            by_cluster[segment.semantic_cluster] += segment.tokens_estimated

        usage = _parse_usage(inference.get("usage"), inference_id)
        if usage is None:
            self._warnings.append(
                f"inference {inference_id} has no provider token usage; exact totals exclude it"
            )
        residual = None if usage is None else usage.input_tokens - input_estimated

        classification = self._classify_inference(response_item_ids, tool_ids)
        reasoning_by_cluster = (
            {}
            if usage is None
            else {
                cluster: usage.reasoning_output_tokens * probability
                for cluster, probability in classification.probabilities.items()
            }
        )
        response_payload = inference.get("raw_response_payload_id")
        if response_payload is not None and not isinstance(response_payload, str):
            raise TraceBundleError(
                f"inference {inference_id} raw_response_payload_id must be a string or null"
            )
        step = InferenceStep(
            inference_call_id=inference_id,
            thread_id=_required_str(
                inference, "thread_id", f"inference {inference_id}"
            ),
            codex_turn_id=_required_str(
                inference, "codex_turn_id", f"inference {inference_id}"
            ),
            model=_required_str(inference, "model", f"inference {inference_id}"),
            provider=_required_str(
                inference, "provider_name", f"inference {inference_id}"
            ),
            started_at_unix_ms=_required_int(
                execution, "started_at_unix_ms", f"inference {inference_id}.execution"
            ),
            ended_at_unix_ms=_optional_int(
                execution.get("ended_at_unix_ms"),
                f"inference {inference_id}.execution.ended_at_unix_ms",
            ),
            status=_required_str(
                execution, "status", f"inference {inference_id}.execution"
            ),
            usage_exact=usage,
            input_tokens_estimated=input_estimated,
            input_residual_tokens=residual,
            input_tokens_by_cluster_estimated=dict(sorted(by_cluster.items())),
            reasoning_cluster_probabilities=classification.probabilities,
            reasoning_tokens_by_cluster_estimated=reasoning_by_cluster,
            reasoning_evidence_mode=classification.evidence_mode,
            request_payload_id=request_payload_id,
            response_payload_id=response_payload,
            response_item_ids=response_item_ids,
            tool_call_ids=tool_ids,
        )
        return step, [*input_segments, *output_segments]

    def _input_segments(
        self,
        inference_id: str,
        request_payload_id: str,
        request: dict[str, Any],
        request_item_ids: tuple[str, ...],
    ) -> list[Segment]:
        segments: list[Segment] = []
        index = 0

        if "instructions" in request:
            value = request["instructions"]
            if value != "":
                segments.append(
                    _segment(
                        f"{inference_id}:input:{index}",
                        inference_id,
                        "input",
                        "instructions",
                        "instructions",
                        value,
                        raw_payload_id=request_payload_id,
                        evidence_mode="request_field",
                    )
                )
                index += 1

        tools = request.get("tools")
        if tools is not None:
            if not isinstance(tools, list):
                raise TraceBundleError(
                    f"inference {inference_id} request.tools must be an array or null"
                )
            for tool in tools:
                segments.append(
                    _segment(
                        f"{inference_id}:input:{index}",
                        inference_id,
                        "input",
                        "tool_definition",
                        "tool_definitions",
                        tool,
                        raw_payload_id=request_payload_id,
                        evidence_mode="request_field",
                    )
                )
                index += 1

        items = request.get("input")
        if not isinstance(items, list):
            raise TraceBundleError(
                f"inference {inference_id} request.input must be an array"
            )
        if len(items) != len(request_item_ids):
            self._warnings.append(
                f"inference {inference_id} request.input has {len(items)} entries but "
                f"request_item_ids has {len(request_item_ids)}; unmatched entries use raw tags"
            )
        for item_index, raw_item in enumerate(items):
            item_id = (
                request_item_ids[item_index]
                if item_index < len(request_item_ids)
                else None
            )
            conversation_item = (
                self._conversation_items.get(item_id) if item_id is not None else None
            )
            structural_type, evidence_mode = self._input_structural_type(
                raw_item, conversation_item, item_id
            )
            call_id = (
                conversation_item.get("call_id")
                if conversation_item is not None
                and isinstance(conversation_item.get("call_id"), str)
                else None
            )
            segments.append(
                _segment(
                    f"{inference_id}:input:{index}",
                    inference_id,
                    "input",
                    structural_type,
                    structural_type,
                    raw_item,
                    item_id=item_id,
                    call_id=call_id,
                    raw_payload_id=request_payload_id,
                    evidence_mode=evidence_mode,
                )
            )
            index += 1

        overhead = {
            key: value
            for key, value in request.items()
            if key not in {"instructions", "input", "tools"}
        }
        if overhead:
            segments.append(
                _segment(
                    f"{inference_id}:input:{index}",
                    inference_id,
                    "input",
                    "other_protocol_overhead",
                    "other_protocol_overhead",
                    overhead,
                    raw_payload_id=request_payload_id,
                    evidence_mode="request_field",
                )
            )
        return segments

    def _input_structural_type(
        self,
        raw_item: Any,
        conversation_item: dict[str, Any] | None,
        item_id: str | None,
    ) -> tuple[str, str]:
        if conversation_item is None:
            return _raw_input_type(raw_item), "raw_item_tag"
        kind = conversation_item.get("kind")
        role = conversation_item.get("role")
        if kind == "reasoning":
            return "prior_reasoning", "trace_structure"
        if kind == "compaction_marker" or conversation_item.get("channel") == "summary":
            return "compaction_summary", "trace_structure"
        if kind in {"function_call_output", "custom_tool_call_output"}:
            tool_id = self._item_to_tool.get(item_id or "")
            if tool_id is None:
                return "tool_result", "trace_structure"
            return self._tool_output_input_type(tool_id)
        if conversation_item.get("agent_message") is not None:
            return "subagent_result", "trace_structure"
        if role == "user":
            return "user_input", "trace_structure"
        if role == "assistant":
            return "prior_assistant_message", "trace_structure"
        return "other_protocol_overhead", "trace_structure"

    def _tool_output_input_type(self, tool_id: str) -> tuple[str, str]:
        tool = self._tool_calls[tool_id]
        kind = _enum_tag(tool.get("kind"), f"tool {tool_id}.kind")
        if kind in {"spawn_agent", "assign_agent_task", "send_message", "wait_agent"}:
            return "subagent_result", "tool_kind"
        if kind not in {"exec_command", "write_stdin"}:
            return "tool_result", "tool_kind"
        operation_id = tool.get("terminal_operation_id")
        operation = (
            self._terminal_operations.get(operation_id)
            if isinstance(operation_id, str)
            else None
        )
        if operation is not None:
            request = operation.get("request")
            command = _terminal_command_text(request)
            if _FILE_READ_COMMAND.search(command):
                return "repository_file_content", "runtime_command_heuristic"
        return "terminal_output", "tool_kind"

    def _output_segments(
        self, inference_id: str, response_item_ids: tuple[str, ...]
    ) -> list[Segment]:
        segments: list[Segment] = []
        for index, item_id in enumerate(response_item_ids):
            item = self._conversation_items.get(item_id)
            if item is None:
                raise TraceBundleError(
                    f"inference {inference_id} references unknown response item {item_id}"
                )
            value, text, content_mode = self._conversation_material(item)
            kind = _required_str(item, "kind", f"conversation item {item_id}")
            structural_type = {
                "reasoning": "reasoning",
                "function_call": "tool_call_arguments",
                "custom_tool_call": "tool_call_arguments",
                "message": "assistant_message",
            }.get(kind, kind)
            tool_id = self._item_to_tool.get(item_id)
            tool_kinds: tuple[str, ...] = ()
            action_text = ""
            if tool_id is not None:
                tool = self._tool_calls[tool_id]
                tool_kinds = (_enum_tag(tool.get("kind"), f"tool {tool_id}.kind"),)
                action_text = self._tool_action_text(tool)
            classification = self.classifier.classify(
                text, tool_kinds=tool_kinds, action_text=action_text
            )
            evidence_mode = (
                content_mode
                if kind == "reasoning"
                and classification.evidence_mode == "reasoning_text"
                else classification.evidence_mode
            )
            estimate = estimate_value(value)
            call_id = (
                item.get("call_id") if isinstance(item.get("call_id"), str) else None
            )
            raw_payload_id = _first_payload_id(item)
            segments.append(
                Segment(
                    segment_id=f"{inference_id}:model_output:{index}",
                    inference_call_id=inference_id,
                    direction="model_output",
                    structural_type=structural_type,
                    semantic_cluster=classification.primary_cluster,
                    cluster_probability=classification.primary_probability,
                    tokens_estimated=estimate.tokens,
                    token_method=estimate.method,
                    byte_count=estimate.byte_count,
                    item_id=item_id,
                    call_id=call_id,
                    raw_payload_id=raw_payload_id,
                    evidence_mode=evidence_mode,
                )
            )
        return segments

    def _classify_inference(
        self, response_item_ids: tuple[str, ...], tool_ids: tuple[str, ...]
    ) -> Classification:
        reasoning_text: list[str] = []
        modes: set[str] = set()
        for item_id in response_item_ids:
            item = self._conversation_items.get(item_id)
            if item is None or item.get("kind") != "reasoning":
                continue
            _, text, mode = self._conversation_material(item)
            if text:
                reasoning_text.append(text)
                modes.add(mode)
        action_texts: list[str] = []
        tool_kinds: list[str] = []
        for tool_id in tool_ids:
            tool = self._tool_calls.get(tool_id)
            if tool is None:
                raise TraceBundleError(
                    f"inference references unknown started tool call {tool_id}"
                )
            tool_kinds.append(_enum_tag(tool.get("kind"), f"tool {tool_id}.kind"))
            action_texts.append(self._tool_action_text(tool))
        classification = self.classifier.classify(
            "\n".join(reasoning_text),
            tool_kinds=tuple(tool_kinds),
            action_text="\n".join(action_texts),
        )
        if reasoning_text and modes == {"reasoning_summary"}:
            return Classification(classification.probabilities, "reasoning_summary")
        return classification

    def _conversation_material(
        self, item: dict[str, Any]
    ) -> tuple[list[Any], str, str]:
        body = _required_object(item, "body", "conversation item")
        parts = body.get("parts")
        if not isinstance(parts, list):
            raise TraceBundleError("conversation item body.parts must be an array")
        values: list[Any] = []
        text_parts: list[str] = []
        has_text = False
        has_summary = False
        for part in parts:
            if not isinstance(part, dict):
                raise TraceBundleError("conversation item part must be an object")
            part_type = part.get("type")
            if part_type in {"text", "summary"}:
                text = part.get("text")
                if not isinstance(text, str):
                    raise TraceBundleError(f"conversation {part_type} part has no text")
                text_parts.append(text)
                has_text |= part_type == "text"
                has_summary |= part_type == "summary"
                values.append(part)
                continue
            payload_id = part.get("raw_payload_id")
            if part_type in {"json", "payload_ref"} and isinstance(payload_id, str):
                payload = self.bundle.payload_json(payload_id)
                values.append(payload)
                text_parts.extend(_iter_strings(payload))
                continue
            values.append(part)
        mode = (
            "reasoning_text"
            if has_text
            else "reasoning_summary"
            if has_summary
            else "encoded_or_structural"
        )
        return values, "\n".join(text_parts), mode

    def _analyze_tool(
        self, tool_id: str, tool: dict[str, Any]
    ) -> tuple[ToolCallAnalysis, Segment | None]:
        execution = _required_object(tool, "execution", f"tool {tool_id}")
        invocation_id = _optional_str(
            tool.get("raw_invocation_payload_id"),
            f"tool {tool_id}.raw_invocation_payload_id",
        )
        result_id = _optional_str(
            tool.get("raw_result_payload_id"), f"tool {tool_id}.raw_result_payload_id"
        )
        invocation = self.bundle.payload_json(invocation_id) if invocation_id else None
        result = self.bundle.payload_json(result_id) if result_id else None
        invocation_estimate = estimate_value(invocation) if invocation_id else None
        result_estimate = estimate_value(result) if result_id else None

        visible_tokens = 0
        output_item_ids, visible_evidence = self._visible_output_items(tool_id, tool)
        for item_id in output_item_ids:
            item = self._conversation_items.get(item_id)
            if item is None:
                raise TraceBundleError(
                    f"tool {tool_id} references unknown item {item_id}"
                )
            value, _, _ = self._conversation_material(item)
            visible_tokens += estimate_value(value).tokens

        operation = self._terminal_operation(tool)
        original_tokens: int | None = None
        if operation is not None:
            terminal_result = operation.get("result")
            if isinstance(terminal_result, dict):
                original_tokens = _optional_int(
                    terminal_result.get("original_token_count"),
                    f"tool {tool_id}.terminal_result.original_token_count",
                )
        files, added, deleted = _patch_stats(invocation)
        kind = _enum_tag(tool.get("kind"), f"tool {tool_id}.kind")
        requester = _enum_tag(tool.get("requester"), f"tool {tool_id}.requester")
        analysis = ToolCallAnalysis(
            tool_call_id=tool_id,
            inference_call_id=self._tool_to_inference.get(tool_id),
            thread_id=_required_str(tool, "thread_id", f"tool {tool_id}"),
            codex_turn_id=_optional_str(
                tool.get("started_by_codex_turn_id"),
                f"tool {tool_id}.started_by_codex_turn_id",
            ),
            kind=kind,
            requester=requester,
            started_at_unix_ms=_required_int(
                execution, "started_at_unix_ms", f"tool {tool_id}.execution"
            ),
            ended_at_unix_ms=_optional_int(
                execution.get("ended_at_unix_ms"),
                f"tool {tool_id}.execution.ended_at_unix_ms",
            ),
            status=_required_str(execution, "status", f"tool {tool_id}.execution"),
            invocation_tokens_estimated=(
                invocation_estimate.tokens if invocation_estimate else None
            ),
            raw_output_tokens_estimated=(
                result_estimate.tokens if result_estimate else None
            ),
            original_output_tokens=original_tokens,
            model_visible_output_tokens_estimated=visible_tokens,
            model_visible_output_evidence=visible_evidence,
            token_method=TOKEN_METHOD,
            invocation_payload_id=invocation_id,
            result_payload_id=result_id,
            files_touched=files,
            lines_added=added,
            lines_deleted=deleted,
        )
        if result_estimate is None:
            return analysis, None
        structural_type, evidence_mode = self._tool_output_input_type(tool_id)
        segment = Segment(
            segment_id=f"{tool_id}:tool_output:0",
            inference_call_id=self._tool_to_inference.get(tool_id),
            direction="tool_output",
            structural_type="raw_tool_result",
            semantic_cluster=structural_type,
            cluster_probability=1.0,
            tokens_estimated=result_estimate.tokens,
            token_method=result_estimate.method,
            byte_count=result_estimate.byte_count,
            call_id=_optional_str(
                tool.get("model_visible_call_id"),
                f"tool {tool_id}.model_visible_call_id",
            ),
            raw_payload_id=result_id,
            evidence_mode=evidence_mode,
        )
        return analysis, segment

    def _terminal_operation(self, tool: dict[str, Any]) -> dict[str, Any] | None:
        operation_id = tool.get("terminal_operation_id")
        if operation_id is None:
            return None
        if not isinstance(operation_id, str):
            raise TraceBundleError(
                "tool terminal_operation_id must be a string or null"
            )
        operation = self._terminal_operations.get(operation_id)
        if operation is None:
            raise TraceBundleError(f"unknown terminal operation: {operation_id}")
        return operation

    def _visible_output_items(
        self, tool_id: str, tool: dict[str, Any]
    ) -> tuple[list[str], str]:
        direct = _string_array(
            tool.get("model_visible_output_item_ids"),
            f"tool {tool_id}.model_visible_output_item_ids",
        )
        if direct:
            return direct, "direct_tool_items"
        requester = tool.get("requester")
        if not isinstance(requester, dict) or requester.get("type") != "code_cell":
            return [], "not_recorded"
        code_cell_id = requester.get("code_cell_id")
        if not isinstance(code_cell_id, str):
            raise TraceBundleError(
                f"tool {tool_id} code_cell requester has no code_cell_id"
            )
        code_cell = self._code_cells.get(code_cell_id)
        if code_cell is None:
            raise TraceBundleError(
                f"tool {tool_id} references unknown code cell {code_cell_id}"
            )
        nested_tool_ids = _string_array(
            code_cell.get("nested_tool_call_ids"),
            f"code cell {code_cell_id}.nested_tool_call_ids",
        )
        if nested_tool_ids != [tool_id]:
            self._warnings.append(
                f"tool {tool_id} belongs to code cell {code_cell_id} with "
                f"{len(nested_tool_ids)} nested tools; model-visible output is ambiguous"
            )
            return [], "ambiguous_code_cell_output"
        return (
            _string_array(
                code_cell.get("output_item_ids"),
                f"code cell {code_cell_id}.output_item_ids",
            ),
            "single_tool_code_cell_output",
        )

    def _tool_action_text(self, tool: dict[str, Any]) -> str:
        values: list[Any] = [tool.get("summary", {})]
        operation = self._terminal_operation(tool)
        if operation is not None:
            values.append(operation.get("request", {}))
        invocation_id = tool.get("raw_invocation_payload_id")
        if isinstance(invocation_id, str):
            values.append(self.bundle.payload_json(invocation_id))
        return "\n".join(_iter_strings(values))

    def _build_tool_to_inference(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for inference_id, inference in self._inference_calls.items():
            tool_ids = _string_array(
                inference.get("tool_call_ids_started_by_response"),
                f"inference {inference_id}.tool_call_ids_started_by_response",
            )
            for tool_id in tool_ids:
                previous = result.get(tool_id)
                if previous is not None and previous != inference_id:
                    raise TraceBundleError(
                        f"tool {tool_id} is attributed to both {previous} and {inference_id}"
                    )
                result[tool_id] = inference_id
        return result

    def _build_item_to_tool(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for tool_id, tool in self._tool_calls.items():
            for field in (
                "model_visible_call_item_ids",
                "model_visible_output_item_ids",
            ):
                for item_id in _string_array(
                    tool.get(field), f"tool {tool_id}.{field}"
                ):
                    previous = result.get(item_id)
                    if previous is not None and previous != tool_id:
                        raise TraceBundleError(
                            f"conversation item {item_id} maps to multiple tools"
                        )
                    result[item_id] = tool_id
        for code_cell_id, code_cell in self._code_cells.items():
            nested_tool_ids = _string_array(
                code_cell.get("nested_tool_call_ids"),
                f"code cell {code_cell_id}.nested_tool_call_ids",
            )
            if len(nested_tool_ids) != 1:
                continue
            tool_id = nested_tool_ids[0]
            if tool_id not in self._tool_calls:
                raise TraceBundleError(
                    f"code cell {code_cell_id} references unknown tool {tool_id}"
                )
            source_item_id = _required_str(
                code_cell, "source_item_id", f"code cell {code_cell_id}"
            )
            code_cell_items = [
                source_item_id,
                *_string_array(
                    code_cell.get("output_item_ids"),
                    f"code cell {code_cell_id}.output_item_ids",
                ),
            ]
            for item_id in code_cell_items:
                previous = result.get(item_id)
                if previous is not None and previous != tool_id:
                    raise TraceBundleError(
                        f"conversation item {item_id} maps to multiple tools"
                    )
                result[item_id] = tool_id
        return result

    def _inference_sort_key(self, entry: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        inference_id, inference = entry
        execution = inference.get("execution")
        seq = execution.get("started_seq") if isinstance(execution, dict) else None
        return (seq if isinstance(seq, int) else 2**63, inference_id)

    def _tool_sort_key(self, entry: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        tool_id, tool = entry
        execution = tool.get("execution")
        seq = execution.get("started_seq") if isinstance(execution, dict) else None
        return (seq if isinstance(seq, int) else 2**63, tool_id)


def _request_object(payload: Any, inference_id: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("input"), list):
        return payload
    if isinstance(payload, dict):
        nested = payload.get("request")
        if isinstance(nested, dict) and isinstance(nested.get("input"), list):
            return nested
    raise TraceBundleError(
        f"inference {inference_id} raw request is not a Responses request object"
    )


def _parse_usage(value: Any, inference_id: str) -> ExactTokenUsage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TraceBundleError(
            f"inference {inference_id} usage must be an object or null"
        )
    return ExactTokenUsage(
        input_tokens=_required_int(
            value, "input_tokens", f"inference {inference_id}.usage"
        ),
        cached_input_tokens=_required_int(
            value, "cached_input_tokens", f"inference {inference_id}.usage"
        ),
        output_tokens=_required_int(
            value, "output_tokens", f"inference {inference_id}.usage"
        ),
        reasoning_output_tokens=_required_int(
            value, "reasoning_output_tokens", f"inference {inference_id}.usage"
        ),
    )


def _segment(
    segment_id: str,
    inference_id: str,
    direction: str,
    structural_type: str,
    semantic_cluster: str,
    value: Any,
    *,
    item_id: str | None = None,
    call_id: str | None = None,
    raw_payload_id: str | None = None,
    evidence_mode: str,
) -> Segment:
    estimate = estimate_value(value)
    return Segment(
        segment_id=segment_id,
        inference_call_id=inference_id,
        direction=direction,
        structural_type=structural_type,
        semantic_cluster=semantic_cluster,
        cluster_probability=1.0,
        tokens_estimated=estimate.tokens,
        token_method=estimate.method,
        byte_count=estimate.byte_count,
        item_id=item_id,
        call_id=call_id,
        raw_payload_id=raw_payload_id,
        evidence_mode=evidence_mode,
    )


def _raw_input_type(value: Any) -> str:
    if not isinstance(value, dict):
        return "other_protocol_overhead"
    item_type = value.get("type")
    role = value.get("role")
    if item_type == "reasoning":
        return "prior_reasoning"
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        return "tool_result"
    if role == "user":
        return "user_input"
    if role == "assistant":
        return "prior_assistant_message"
    return "other_protocol_overhead"


def _terminal_command_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    command = value.get("display_command")
    if isinstance(command, str):
        return command
    command_parts = value.get("command")
    if isinstance(command_parts, list) and all(
        isinstance(part, str) for part in command_parts
    ):
        return " ".join(command_parts)
    return ""


def _patch_stats(value: Any) -> tuple[tuple[str, ...], int, int]:
    if value is None:
        return (), 0, 0
    files: set[str] = set()
    added = 0
    deleted = 0
    for text in _iter_strings(value):
        files.update(match.strip() for match in _PATCH_FILE.findall(text))
        files.update(match.strip() for match in _DIFF_FILE.findall(text))
        for line in text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                deleted += 1
    return tuple(sorted(files)), added, deleted


def _first_payload_id(item: dict[str, Any]) -> str | None:
    body = item.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("parts"), list):
        return None
    for part in body["parts"]:
        if not isinstance(part, dict):
            continue
        raw_payload_id = part.get("raw_payload_id")
        if isinstance(raw_payload_id, str):
            return raw_payload_id
    return None


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_strings(nested)


def _enum_tag(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        tag = value.get("type")
        if isinstance(tag, str):
            return tag
    raise TraceBundleError(f"{label} must be a string or tagged object")


def _object_map(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise TraceBundleError(f"{label} must be an object")
    result: dict[str, dict[str, Any]] = {}
    for key, nested in value.items():
        if not isinstance(key, str) or not isinstance(nested, dict):
            raise TraceBundleError(f"{label} entries must map string ids to objects")
        result[key] = nested
    return result


def _required_object(value: dict[str, Any], field: str, label: str) -> dict[str, Any]:
    nested = value.get(field)
    if not isinstance(nested, dict):
        raise TraceBundleError(f"{label}.{field} must be an object")
    return nested


def _required_str(value: dict[str, Any], field: str, label: str) -> str:
    nested = value.get(field)
    if not isinstance(nested, str):
        raise TraceBundleError(f"{label}.{field} must be a string")
    return nested


def _optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TraceBundleError(f"{label} must be a string or null")
    return value


def _required_int(value: dict[str, Any], field: str, label: str) -> int:
    nested = value.get(field)
    if not isinstance(nested, int):
        raise TraceBundleError(f"{label}.{field} must be an integer")
    return nested


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise TraceBundleError(f"{label} must be an integer or null")
    return value


def _string_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TraceBundleError(f"{label} must be an array of strings")
    return value
