"""Build a content-free context-pressure snapshot and visualization.

The parser reads a Codex parent-session JSONL transcript but retains only
counts, character lengths, event positions, timestamps, and a source digest.
It never copies message, reasoning, command, or tool-output bodies into the
snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

SCHEMA_VERSION = 1
DEFAULT_BINS = 48

CATEGORY_ORDER = (
    "formal_gate",
    "workflow_design_source_load",
    "workflow_reload_contract",
    "other_tool_output",
)
CATEGORY_META = {
    "formal_gate": {
        "label": "Formal gate / verification",
        "color": "#D55E00",
        "description": "Canonical checks plus direct pytest, Ruff, mypy, and CI verification.",
    },
    "workflow_design_source_load": {
        "label": "Workflow / design / source loading",
        "color": "#0072B2",
        "description": "Read-oriented inspection of workflow, design, source, tests, and repository state.",
    },
    "workflow_reload_contract": {
        "label": "Workflow re-read / contract replay",
        "color": "#CC79A7",
        "description": "Repeated workflow reads and Issue state, artifact, spawn, or verification contracts.",
    },
    "other_tool_output": {
        "label": "Other tool output",
        "color": "#999999",
        "description": "Edits, goal/plan updates, collaboration envelopes, and uncategorized tool results.",
    },
}

TOOL_CALL_PAYLOAD_TYPES = {"custom_tool_call", "function_call"}
TOOL_OUTPUT_PAYLOAD_TYPES = {"custom_tool_call_output", "function_call_output"}

FORMAL_PATTERNS = (
    re.compile(r"manage_issue_task\.py\s+run-check", re.IGNORECASE),
    re.compile(r"python\s+-m\s+pytest", re.IGNORECASE),
    re.compile(r"python\s+-m\s+ruff", re.IGNORECASE),
    re.compile(r"python\s+-m\s+mypy", re.IGNORECASE),
    re.compile(r"python\s+-m\s+spin\s+(?:lint|typecheck|ci)", re.IGNORECASE),
    re.compile(r"training_queue\.sh\s+(?:enqueue|start)", re.IGNORECASE),
)
CONTRACT_PATTERNS = (
    re.compile(r"\.codex/tasks/issue-753", re.IGNORECASE),
    re.compile(r"manage_issue_task\.py", re.IGNORECASE),
    re.compile(r"spawn-contracts\.md", re.IGNORECASE),
    re.compile(r"\.codex/agents/(?:issue-|codebase-)", re.IGNORECASE),
    re.compile(r"issue_task_(?:artifacts|schema|verification|checks)\.py", re.IGNORECASE),
)
WORKFLOW_DOC_PATTERN = re.compile(
    r"(?:issue-subagent-workflow/(?:SKILL\.md|references/[^\s\"'`;]+)"
    r"|test-structure/SKILL\.md"
    r"|training-queue/(?:SKILL\.md|reference/[^\s\"'`;]+))",
    re.IGNORECASE,
)
READ_PATTERN = re.compile(
    r"(?:\bsed\s+-n\b|\brg\b|\b(?:find|head|tail|wc|ls)\b"
    r"|\bgit\s+(?:diff|show|status|log|rev-parse|ls-tree)\b"
    r"|\bgh\s+(?:issue|pr)\s+view\b)",
    re.IGNORECASE,
)
ASYNC_CELL_PATTERN = re.compile(r"Script running with cell ID\s+([^\s\"']+)")
SESSION_OUTPUT_PATTERN = re.compile(r'"session_id"\s*:\s*(\d+)')
SESSION_INPUT_PATTERN = re.compile(r"session_id\s*:\s*(\d+)")

WAIT_TOOL_NAMES = {"wait", "collaboration.wait_agent"}
SUBAGENT_CONTROL_NAMES = {
    "collaboration.followup_task",
    "collaboration.interrupt_agent",
    "collaboration.send_message",
    "collaboration.spawn_agent",
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    raw_input: str
    category: str


def _compact_json_length(value: Any) -> int:
    """Return the character length of a compact, UTF-8-preserving JSON value."""

    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tool_name(payload: dict[str, Any]) -> str:
    namespace = payload.get("namespace")
    name = str(payload.get("name", "unknown"))
    return f"{namespace}.{name}" if namespace else name


def _tool_input(payload: dict[str, Any]) -> str:
    raw = payload.get("input", payload.get("arguments", ""))
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False, sort_keys=True)


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_flatten_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(_flatten_strings(item))
        return strings
    return []


def _classify_call(
    name: str,
    raw_input: str,
    window_index: int,
    seen_workflow_docs: set[str],
) -> str:
    """Classify a call without retaining its input.

    Classification is intentionally deterministic and path/pattern based.
    Continuation waits are reassigned to their originating asynchronous call
    later, when the call/output pair is available.
    """

    if name != "exec":
        return "other_tool_output"

    if any(pattern.search(raw_input) for pattern in FORMAL_PATTERNS):
        category = "formal_gate"
    elif any(pattern.search(raw_input) for pattern in CONTRACT_PATTERNS):
        category = "workflow_reload_contract"
    else:
        workflow_docs = {
            match.group(0).lower() for match in WORKFLOW_DOC_PATTERN.finditer(raw_input)
        }
        repeated = bool(workflow_docs & seen_workflow_docs)
        if workflow_docs and (window_index > 0 or repeated):
            category = "workflow_reload_contract"
        elif "tools.exec_command" in raw_input and (
            READ_PATTERN.search(raw_input)
            or workflow_docs
            or any(token in raw_input for token in ("src/", "tests/", "docs/", "README"))
        ):
            category = "workflow_design_source_load"
        else:
            category = "other_tool_output"
        seen_workflow_docs.update(workflow_docs)
    return category


def _parse_cell_id(raw_input: str) -> str | None:
    try:
        decoded = json.loads(raw_input)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    cell_id = decoded.get("cell_id")
    return str(cell_id) if cell_id is not None else None


def _make_windows(
    record_count: int,
    compactions: list[dict[str, Any]],
    char_counts: dict[int, Counter[str]],
    output_counts: dict[int, int],
    wait_counts: dict[int, int],
    subagent_counts: dict[int, int],
) -> list[dict[str, Any]]:
    boundaries = [0, *[int(item["record_index"]) for item in compactions], record_count]
    windows: list[dict[str, Any]] = []
    for index in range(len(boundaries) - 1):
        start_record = boundaries[index] + (1 if index == 0 else 1)
        end_record = boundaries[index + 1] - (1 if index < len(compactions) else 0)
        counts = {category: int(char_counts[index][category]) for category in CATEGORY_ORDER}
        total = sum(counts.values())
        if index == 0:
            label = "W0\nstart → C1"
        elif index < len(compactions):
            label = f"W{index}\nC{index} → C{index + 1}"
        else:
            label = f"W{index}\nC{index} → end"
        windows.append(
            {
                "window_index": index,
                "label": label,
                "start_record": start_record,
                "end_record": max(start_record, end_record),
                "serialized_tool_output_chars": counts,
                "total_serialized_tool_output_chars": total,
                "tool_output_count": int(output_counts[index]),
                "wait_poll_calls": int(wait_counts[index]),
                "subagent_control_calls": int(subagent_counts[index]),
            }
        )
    return windows


def _make_timeline(
    record_count: int,
    output_events: list[tuple[int, str, int]],
    call_events: list[tuple[int, str]],
    bin_count: int,
) -> list[dict[str, Any]]:
    output_bins: list[Counter[str]] = [Counter() for _ in range(bin_count)]
    call_bins: list[Counter[str]] = [Counter() for _ in range(bin_count)]

    def bin_index(record_index: int) -> int:
        return min(bin_count - 1, (max(record_index, 1) - 1) * bin_count // record_count)

    for record_index, category, chars in output_events:
        output_bins[bin_index(record_index)][category] += chars
    for record_index, call_kind in call_events:
        call_bins[bin_index(record_index)][call_kind] += 1

    timeline: list[dict[str, Any]] = []
    for index in range(bin_count):
        start = index * record_count // bin_count + 1
        end = (index + 1) * record_count // bin_count
        timeline.append(
            {
                "bin_index": index,
                "start_record": start,
                "end_record": max(start, end),
                "wait_poll_calls": int(call_bins[index]["wait_poll"]),
                "subagent_control_calls": int(call_bins[index]["subagent_control"]),
                "workflow_reload_contract_output_chars": int(
                    output_bins[index]["workflow_reload_contract"]
                ),
            }
        )
    return timeline


def build_snapshot(transcript: Path, bin_count: int = DEFAULT_BINS) -> dict[str, Any]:
    """Parse one parent-session transcript into a content-free aggregate."""

    if bin_count < 12:
        raise ValueError("timeline bin count must be at least 12")

    calls: dict[str, ToolCall] = {}
    async_cells: dict[str, str] = {}
    pty_sessions: dict[str, str] = {}
    seen_workflow_docs: set[str] = set()
    compactions: list[dict[str, Any]] = []
    output_events: list[tuple[int, str, int]] = []
    call_events: list[tuple[int, str]] = []
    char_counts: dict[int, Counter[str]] = defaultdict(Counter)
    output_counts: Counter[int] = Counter()
    wait_counts: Counter[int] = Counter()
    subagent_counts: Counter[int] = Counter()
    primitive_call_counts: Counter[str] = Counter()

    record_count = 0
    current_window = 0
    with transcript.open(encoding="utf-8") as stream:
        for record_count, line in enumerate(stream, start=1):
            record = json.loads(line)
            record_type = record.get("type")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                payload = {}

            if record_type == "compacted":
                compactions.append(
                    {
                        "ordinal": len(compactions) + 1,
                        "record_index": record_count,
                        "timestamp": record.get("timestamp"),
                        "window_number": payload.get("window_number"),
                    }
                )
                current_window += 1
                continue

            payload_type = payload.get("type")
            if record_type == "response_item" and payload_type in TOOL_CALL_PAYLOAD_TYPES:
                name = _tool_name(payload)
                raw_input = _tool_input(payload)
                category = _classify_call(
                    name, raw_input, current_window, seen_workflow_docs
                )
                call_id = str(payload.get("call_id", ""))
                calls[call_id] = ToolCall(name=name, raw_input=raw_input, category=category)
                primitive_call_counts[name] += 1

                is_wait = name in WAIT_TOOL_NAMES or "tools.write_stdin(" in raw_input
                if is_wait:
                    wait_counts[current_window] += 1
                    call_events.append((record_count, "wait_poll"))
                if name in SUBAGENT_CONTROL_NAMES:
                    subagent_counts[current_window] += 1
                    call_events.append((record_count, "subagent_control"))
                continue

            if record_type != "response_item" or payload_type not in TOOL_OUTPUT_PAYLOAD_TYPES:
                continue

            call = calls.get(str(payload.get("call_id", "")))
            category = call.category if call else "other_tool_output"
            if call and call.name == "wait":
                cell_id = _parse_cell_id(call.raw_input)
                if cell_id is not None:
                    category = async_cells.get(cell_id, category)
            elif call and "tools.write_stdin(" in call.raw_input:
                session_match = SESSION_INPUT_PATTERN.search(call.raw_input)
                if session_match:
                    category = pty_sessions.get(session_match.group(1), category)

            output = payload.get("output")
            output_chars = _compact_json_length(output)
            char_counts[current_window][category] += output_chars
            output_counts[current_window] += 1
            output_events.append((record_count, category, output_chars))

            flattened = "\n".join(_flatten_strings(output))
            for match in ASYNC_CELL_PATTERN.finditer(flattened):
                async_cells[match.group(1)] = category
            for match in SESSION_OUTPUT_PATTERN.finditer(flattened):
                pty_sessions[match.group(1)] = category

    if record_count == 0:
        raise ValueError(f"empty transcript: {transcript}")

    windows = _make_windows(
        record_count,
        compactions,
        char_counts,
        output_counts,
        wait_counts,
        subagent_counts,
    )
    timeline = _make_timeline(
        record_count, output_events, call_events, bin_count=bin_count
    )
    total_chars = sum(item["total_serialized_tool_output_chars"] for item in windows)
    total_outputs = sum(item["tool_output_count"] for item in windows)
    total_waits = sum(item["wait_poll_calls"] for item in windows)
    total_subagent = sum(item["subagent_control_calls"] for item in windows)

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "Issue #753 parent-session context pressure",
        "source": {
            "filename": transcript.name,
            "sha256": _sha256(transcript),
            "size_bytes": transcript.stat().st_size,
            "jsonl_record_count": record_count,
        },
        "method": {
            "unit": "characters in compact json serialization of payload.output",
            "included_output_types": sorted(TOOL_OUTPUT_PAYLOAD_TYPES),
            "classification": "initiating call patterns; async wait continuations inherit the originating call category",
            "raw_content_retained": False,
            "exclusions": [
                "encrypted reasoning and all reasoning/message bodies",
                "compaction replacement-history bodies",
                "child-agent transcript files (parent-visible collaboration envelopes remain)",
            ],
            "causal_scope": "timeline shows temporal association, not proof that orchestration caused compaction",
        },
        "categories": [
            {"key": key, **CATEGORY_META[key]} for key in CATEGORY_ORDER
        ],
        "summary": {
            "compaction_count": len(compactions),
            "tool_output_count": total_outputs,
            "serialized_tool_output_chars": total_chars,
            "wait_poll_calls": total_waits,
            "subagent_control_calls": total_subagent,
            "primitive_tool_call_counts": dict(sorted(primitive_call_counts.items())),
        },
        "compactions": compactions,
        "windows": windows,
        "timeline_bin_count": bin_count,
        "timeline": timeline,
    }


def write_snapshot(snapshot: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mchars(value: float, _position: float) -> str:
    return f"{value / 1_000_000:.1f}M"


def plot_snapshot(snapshot: dict[str, Any], output: Path) -> None:
    """Render a snapshot without accessing the raw transcript."""

    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported snapshot schema: {snapshot.get('schema_version')!r}"
        )

    category_meta = {item["key"]: item for item in snapshot["categories"]}
    windows = snapshot["windows"]
    timeline = snapshot["timeline"]
    summary = snapshot["summary"]
    source = snapshot["source"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    figure = plt.figure(figsize=(18, 12), dpi=120, facecolor="#FAFAFA")
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.05, 1.0),
        left=0.075,
        right=0.92,
        top=0.84,
        bottom=0.12,
        hspace=0.34,
    )
    top = figure.add_subplot(grid[0])
    bottom = figure.add_subplot(grid[1])
    top.set_facecolor("white")
    bottom.set_facecolor("white")

    figure.suptitle(
        "Issue #753 — parent-session context pressure",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color="#202124",
    )
    figure.text(
        0.075,
        0.928,
        "Tool-output volume by compaction window, with orchestration traffic and workflow-contract replay over time",
        ha="left",
        fontsize=13,
        color="#555555",
    )

    x = np.arange(len(windows))
    stack = np.zeros(len(windows), dtype=float)
    for category in CATEGORY_ORDER:
        values = np.array(
            [window["serialized_tool_output_chars"][category] for window in windows],
            dtype=float,
        )
        meta = category_meta[category]
        top.bar(
            x,
            values,
            bottom=stack,
            width=0.66,
            label=meta["label"],
            color=meta["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        stack += values

    for index, total in enumerate(stack):
        top.text(
            index,
            total + max(stack) * 0.025,
            f"{total / 1_000_000:.2f}M chars",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )
    top.set_xticks(x, [window["label"] for window in windows])
    top.set_ylabel("Serialized tool-output characters")
    top.set_title(
        "Context pressure: tool outputs retained in each compaction window",
        loc="left",
        pad=14,
        fontsize=15,
    )
    top.yaxis.set_major_formatter(FuncFormatter(_mchars))
    top.yaxis.set_major_locator(MaxNLocator(nbins=5))
    top.grid(axis="y", color="#DADCE0", linewidth=0.8, alpha=0.75)
    top.set_axisbelow(True)
    top.spines[["top", "right"]].set_visible(False)
    top_handles, top_labels = top.get_legend_handles_labels()
    figure.legend(
        top_handles,
        top_labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.902),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.6,
    )

    record_count = int(source["jsonl_record_count"])
    centers = np.array(
        [
            100.0 * (item["start_record"] + item["end_record"]) / (2 * record_count)
            for item in timeline
        ]
    )
    widths = np.array(
        [100.0 * (item["end_record"] - item["start_record"] + 1) / record_count for item in timeline]
    )
    wait_calls = np.array([item["wait_poll_calls"] for item in timeline])
    subagent_calls = np.array([item["subagent_control_calls"] for item in timeline])
    replay_kchars = np.array(
        [item["workflow_reload_contract_output_chars"] / 1000 for item in timeline]
    )

    bottom.bar(
        centers,
        wait_calls,
        width=widths * 0.88,
        color="#56B4E9",
        edgecolor="white",
        linewidth=0.4,
        label=f"Wait / process-poll calls ({summary['wait_poll_calls']})",
    )
    bottom.bar(
        centers,
        subagent_calls,
        bottom=wait_calls,
        width=widths * 0.88,
        color="#009E73",
        edgecolor="white",
        linewidth=0.4,
        label=f"Subagent control calls ({summary['subagent_control_calls']})",
    )
    replay_axis = bottom.twinx()
    replay_axis.plot(
        centers,
        replay_kchars,
        color=CATEGORY_META["workflow_reload_contract"]["color"],
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label="Workflow re-read / contract output (kchars/bin)",
        zorder=5,
    )
    replay_axis.fill_between(
        centers,
        replay_kchars,
        color=CATEGORY_META["workflow_reload_contract"]["color"],
        alpha=0.10,
    )

    for compaction in snapshot["compactions"]:
        position = 100.0 * int(compaction["record_index"]) / record_count
        bottom.axvline(position, color="#222222", linestyle=(0, (4, 4)), linewidth=1.3, alpha=0.8)
        bottom.text(
            position + 0.5,
            0.03,
            f"C{compaction['ordinal']}",
            transform=bottom.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )

    bottom.set_xlim(0, 100)
    bottom.set_xlabel(f"Parent transcript progress (% of {record_count:,} JSONL records)")
    bottom.set_ylabel("Calls per timeline bin")
    replay_axis.set_ylabel("Workflow re-read / contract output (kchars per bin)", color="#A64D8D")
    replay_axis.tick_params(axis="y", colors="#A64D8D")
    bottom.set_title(
        "Orchestration frequency and repeated workflow-contract material",
        loc="left",
        pad=14,
        fontsize=15,
    )
    bottom.grid(axis="y", color="#DADCE0", linewidth=0.8, alpha=0.75)
    bottom.set_axisbelow(True)
    bottom.spines[["top", "right"]].set_visible(False)
    replay_axis.spines["top"].set_visible(False)
    bottom.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    replay_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    left_handles, left_labels = bottom.get_legend_handles_labels()
    right_handles, right_labels = replay_axis.get_legend_handles_labels()
    bottom.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.92,
        fontsize=10,
    )

    figure.text(
        0.075,
        0.060,
        (
            f"Snapshot: {source['filename']}  •  {record_count:,} records  •  "
            f"{summary['compaction_count']} compactions  •  "
            "output chars = len(compact JSON(payload.output))"
        ),
        ha="left",
        fontsize=10.5,
        color="#444444",
    )
    figure.text(
        0.075,
        0.032,
        (
            "Scope: parent-session tool outputs/calls only; encrypted reasoning and child transcripts excluded. "
            "Parent-visible collaboration envelopes remain. Timeline is associative, not causal proof."
        ),
        ha="left",
        fontsize=10.5,
        color="#666666",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, facecolor=figure.get_facecolor())
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot Codex parent-session context pressure without retaining transcript content."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcript", type=Path, help="raw parent-session JSONL input")
    source.add_argument("--from-snapshot", type=Path, help="existing aggregate JSON input")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="aggregate JSON output (defaults beside this generator)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="PNG output (defaults beside this generator)",
    )
    parser.add_argument("--timeline-bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    image_path = args.image or base / "issue753_context_pressure.png"
    if args.transcript is not None:
        snapshot_path = args.snapshot or base / "context_snapshot.json"
        snapshot = build_snapshot(args.transcript, bin_count=args.timeline_bins)
        write_snapshot(snapshot, snapshot_path)
    else:
        if args.snapshot is not None:
            raise ValueError("--snapshot is only valid with --transcript")
        snapshot = json.loads(args.from_snapshot.read_text(encoding="utf-8"))
    plot_snapshot(snapshot, image_path)


if __name__ == "__main__":
    main()
