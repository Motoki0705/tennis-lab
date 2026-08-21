from __future__ import annotations

import json
import math
from pathlib import Path

from src.automation.codex_trace.analyzer import TraceAnalyzer
from src.automation.codex_trace.bundle import TraceBundle
from tests.support.codex_trace import write_sample_trace_bundle


def test_analyzer_separates_exact_usage_from_estimated_attribution(
    tmp_path: Path,
) -> None:
    bundle_path = write_sample_trace_bundle(tmp_path)

    report = TraceAnalyzer(TraceBundle.load(bundle_path, auto_reduce=False)).analyze()

    assert report.totals_exact.input_tokens == 240
    assert report.totals_exact.cached_input_tokens == 100
    assert report.totals_exact.output_tokens == 50
    assert report.totals_exact.reasoning_output_tokens == 16
    assert report.totals_exact.inference_calls_with_usage == 2

    first = report.inference_steps[0]
    assert first.reasoning_evidence_mode == "reasoning_summary"
    assert math.isclose(sum(first.reasoning_cluster_probabilities.values()), 1.0)
    assert math.isclose(sum(first.reasoning_tokens_by_cluster_estimated.values()), 12.0)
    assert first.input_residual_tokens == 100 - first.input_tokens_estimated

    second_input_types = {
        segment.structural_type
        for segment in report.segments
        if segment.inference_call_id == "inference-2" and segment.direction == "input"
    }
    assert "tool_result" in second_input_types
    assert "repository_file_content" in second_input_types
    assert "prior_reasoning" in second_input_types


def test_analyzer_distinguishes_raw_and_model_visible_tool_output(
    tmp_path: Path,
) -> None:
    bundle = TraceBundle.load(write_sample_trace_bundle(tmp_path), auto_reduce=False)

    report = TraceAnalyzer(bundle).analyze()

    patch = next(tool for tool in report.tool_calls if tool.kind == "apply_patch")
    terminal = next(tool for tool in report.tool_calls if tool.kind == "exec_command")
    assert patch.files_touched == ("src/example.py",)
    assert patch.lines_added == 1
    assert patch.lines_deleted == 1
    assert patch.raw_output_tokens_estimated is not None
    assert patch.model_visible_output_tokens_estimated > 0
    assert patch.model_visible_output_evidence == "direct_tool_items"
    assert terminal.original_output_tokens == 200
    assert terminal.inference_call_id == "inference-1"

    raw_tool_segments = [
        segment for segment in report.segments if segment.direction == "tool_output"
    ]
    assert len(raw_tool_segments) == 2
    assert {segment.semantic_cluster for segment in raw_tool_segments} == {
        "repository_file_content",
        "tool_result",
    }


def test_single_nested_tool_uses_code_cell_output_as_visible_evidence(
    tmp_path: Path,
) -> None:
    bundle_path = write_sample_trace_bundle(tmp_path)
    state_path = bundle_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tool = state["tool_calls"]["tool-exec"]
    tool["requester"] = {"type": "code_cell", "code_cell_id": "cell-exec"}
    tool["model_visible_call_item_ids"] = []
    tool["model_visible_output_item_ids"] = []
    state["code_cells"] = {
        "cell-exec": {
            "code_cell_id": "cell-exec",
            "source_item_id": "exec-call",
            "output_item_ids": ["exec-output"],
            "nested_tool_call_ids": ["tool-exec"],
        }
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = TraceAnalyzer(TraceBundle.load(bundle_path, auto_reduce=False)).analyze()

    terminal = next(tool for tool in report.tool_calls if tool.kind == "exec_command")
    assert terminal.model_visible_output_tokens_estimated > 0
    assert terminal.model_visible_output_evidence == "single_tool_code_cell_output"
