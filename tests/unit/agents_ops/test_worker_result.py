"""Validate the report grammar independently of launching a worker."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
DONE = 'PARENT_RESULT {"schema_version":1,"state":"done","wait":null}'
CI = {"kind": "ci", "pr": 6, "head": "c" * 40, "resume_on_success": False}
QUEUE = {"kind": "queue", "job_ids": ["12_34_sample.job"]}
LOCAL_JOB_A = "20261124T010203Z_0123456789ab_pytest-unit"
LOCAL_JOB_B = "20261124T010204Z_fedcba987654_mypy.check_1"
LOCAL = {"kind": "local", "job_ids": [LOCAL_JOB_A]}


def report(wait: dict[str, Any]) -> str:
    return "PARENT_RESULT " + json.dumps(
        {"schema_version": 1, "state": "waiting", "wait": wait}
    )


@pytest.mark.parametrize(
    "text",
    [
        DONE,
        DONE.replace("done", "needs_parent"),
        report(CI),
        report({**CI, "resume_on_success": True, "next_step": "ドラフトを解除"}),
        report(QUEUE),
        report(
            {
                "kind": "queue",
                "job_ids": [
                    "0-8204867199999999999_2026-11-20T00-00-00-000000000Z_12345_eval.job",
                    "12_34_sample.job",
                ],
            }
        ),
        report(LOCAL),
        report({"kind": "local", "job_ids": [LOCAL_JOB_A, LOCAL_JOB_B]}),
    ],
)
def test_valid_footer(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    parse = importlib.import_module("parent.worker_result").parse_report
    assert parse(f"作業報告\n\n{text}\n") == json.loads(
        text.removeprefix("PARENT_RESULT ")
    )


@pytest.mark.parametrize(
    "text",
    [
        "成果のみ",
        "  " + DONE,
        DONE + "\nその後の文",
        "```\n" + DONE + "\n```",
        DONE + "\n" + DONE,
        "PARENT_RESULT {",
        DONE.replace('"wait":null', '"wait":null,"wait":null'),
        DONE.replace('"wait":null', '"wait":NaN'),
        DONE.replace('"schema_version":1', '"schema_version":true'),
        DONE.replace('"schema_version":1', '"schema_version":2'),
        DONE.replace(',"wait":null', ""),
        DONE.replace('"done"', '"unknown"'),
        DONE.replace('"wait":null', '"wait":{}'),
        report({"kind": "colab", "job_ids": ["12_34_sample.job"]}),
        report({"kind": "ci", "pr": 6}),
        report({**CI, "pr": True}),
        report({**CI, "head": "abcdef"}),
        report({**CI, "resume_on_success": 0}),
        report({**CI, "resume_on_success": True}),
        report({**CI, "resume_on_success": True, "next_step": " "}),
        report({**CI, "next_step": "未宣言の継続"}),
        report({**QUEUE, "job_ids": []}),
        report({**QUEUE, "job_ids": ["../12_34_sample.job"]}),
        report({**QUEUE, "job_ids": ["12_34_sample.job"] * 2}),
        report({**QUEUE, "job_ids": "12_34_sample.job"}),
        report({**QUEUE, "unknown": 1}),
        report({**LOCAL, "job_ids": []}),
        report({**LOCAL, "job_ids": [LOCAL_JOB_A, LOCAL_JOB_A]}),
        report({**LOCAL, "job_ids": [f"../{LOCAL_JOB_A}"]}),
        report({**LOCAL, "job_ids": [f"{LOCAL_JOB_A}/sub"]}),
        report({**LOCAL, "job_ids": ["20261124T010203Z_0123456789ab_a..b"]}),
        report({**LOCAL, "job_ids": ["not-a-valid-job"]}),
        report({**LOCAL, "job_ids": [123]}),
        report({**LOCAL, "job_ids": LOCAL_JOB_A}),
        report({**LOCAL, "unknown": 1}),
    ],
)
def test_invalid_footer(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    parse = importlib.import_module("parent.worker_result").parse_report
    with pytest.raises(ValueError, match="PARENT_RESULT"):
        parse(text)
