"""No missing, unverified or exported obligation may pass as completion."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("parent.completion")


def finish() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "completed",
        "basis_ref": "fixed-agreement",
        "acceptance": [
            {"id": "agreement", "state": "met", "evidence": "合意した実機確認を実施。"},
            {"id": "ac-1", "state": "met", "evidence": "条件Aを入力して結果Bを確認。"},
        ],
        "remaining_required": [],
        "improvements": {"reviewed": True, "candidates": []},
        "summary": "全体受入済み。",
    }


def acceptance_requirements(
    api: Any,
    accepted: str,
    current: str | None = None,
    heading: str = "全体の完了条件",
) -> dict[str, Any]:
    basis = {
        "event": "agreed",
        "revision": 1,
        "agreement": {"acceptance": "合意した条件を検証する。"},
        "acceptance_snapshot": {heading: accepted},
    }
    body = f"## {heading}\n\n{accepted if current is None else current}"
    session = SimpleNamespace(
        parent=1,
        history=lambda: [basis],
        gh=SimpleNamespace(issue=lambda parent: {"body": body}),
    )
    return dict(api.requirements(session))


@pytest.mark.parametrize(
    "heading", ["全体の完了条件", "Acceptance checklist"], ids=["parent", "checklist"]
)
def test_plain_acceptance_preserves_ids_and_continuation_lines(
    api: Any, heading: str
) -> None:
    accepted = "- [ ] 保存できる。\n  文脈と証拠を確認する。\n* [x] 失敗を表示する。"
    required = acceptance_requirements(
        api, accepted, accepted.replace("[ ]", "[X]"), heading
    )
    assert required["criteria"] == [
        {"id": "agreement", "criterion": "合意した条件を検証する。"},
        {"id": "ac-1", "criterion": "保存できる。 文脈と証拠を確認する。"},
        {"id": "ac-2", "criterion": "失敗を表示する。"},
    ]
    with pytest.raises(ValueError, match="acceptance criteria changed"):
        acceptance_requirements(api, accepted, "- [ ] 条件が変わった。", heading)


def test_grouped_acceptance_identifies_candidates_and_validates_finish(
    api: Any,
) -> None:
    accepted = (
        "### I-1878-result-access\n\n- [ ] 保存できる。\n* [ ] 失敗を表示する。\n\n"
        "### I-2007-result-access\n\n- [ ] 保存できる。"
    )
    expected = [
        {"id": "agreement", "criterion": "合意した条件を検証する。"},
        {
            "id": "ac-I-1878-result-access-1",
            "criterion": "[I-1878-result-access] 保存できる。",
        },
        {
            "id": "ac-I-1878-result-access-2",
            "criterion": "[I-1878-result-access] 失敗を表示する。",
        },
        {
            "id": "ac-I-2007-result-access-1",
            "criterion": "[I-2007-result-access] 保存できる。",
        },
    ]
    required = acceptance_requirements(
        api, accepted, accepted.replace("[ ]", "[X]").replace("\n\n", "\n\n\n")
    )
    assert required["criteria"] == expected
    assert required["basis_ref"] == acceptance_requirements(api, accepted)["basis_ref"]
    spec = finish()
    spec.update(
        basis_ref=required["basis_ref"],
        acceptance=[
            {"id": row["id"], "state": "met", "evidence": "各候補の結果を確認。"}
            for row in expected
        ],
    )
    api.validate_finish(spec, required, {}, 1)
    spec["acceptance"].pop()
    with pytest.raises(
        ValueError, match="each agreed acceptance criterion exactly once"
    ):
        api.validate_finish(spec, required, {}, 1)


@pytest.mark.parametrize(
    "section",
    [
        "説明文。\n### I-1-first\n- [ ] 条件。",
        "### Other\n- [ ] 条件。",
        "#### I-1-first\n- [ ] 条件。",
        "### I-0-first\n- [ ] 条件。",
        "### I-1-First\n- [ ] 条件。",
        "### I-1-" + "a" * 65 + "\n- [ ] 条件。",
        "### I-1-first 説明\n- [ ] 条件。",
        "### I-1-first\n### I-1-second\n- [ ] 条件。",
        "### I-1-first\n- [ ] 条件。\n### I-1-second\n### I-1-third\n- [ ] 別条件。",
        "### I-1-first\n- [ ] 条件。\n### I-1-second",
        "### I-1-first",
        "### I-1-first\n- [ ] 条件。\n* [X] 条件。",
        "### I-1-first\n- [ ] 条件。\n### I-1-first\n- [ ] 別条件。",
        "### I-1-first\n- [ ] 条件。\n### Other\n- [ ] 別条件。",
        "### I-1-first\n- [ ] 条件。\n自由文。",
        "### I-1-first\n- [ ] 条件。\n  自由文。",
        "- [ ] 所属不明の条件。\n### I-1-first\n- [ ] 条件。",
    ],
    ids=[
        "leading-prose",
        "unknown-heading",
        "heading-level",
        "zero-parent",
        "uppercase-key",
        "long-key",
        "heading-suffix",
        "empty-first",
        "empty-middle",
        "empty-last",
        "empty-only",
        "duplicate-item",
        "repeated-group",
        "unknown-heading-after-item",
        "prose-after-item",
        "indented-prose",
        "mixed-format",
    ],
)
def test_grouped_acceptance_rejects_ambiguous_sections(api: Any, section: str) -> None:
    with pytest.raises(ValueError, match="acceptance snapshot"):
        api._criteria(section)


@pytest.mark.parametrize(
    "section",
    ["", "説明文。\n- [ ] 条件。", "- [ ] 条件。\n- [x] 条件。"],
    ids=["empty", "prose", "duplicate"],
)
def test_plain_acceptance_still_requires_unique_checklist_items(
    api: Any, section: str
) -> None:
    with pytest.raises(ValueError, match="acceptance snapshot"):
        api._criteria(section)


@pytest.mark.parametrize(
    "current",
    [
        "### I-1878-first\n- [ ] 条件Aを変更。\n- [ ] 条件B。\n### I-1878-second\n- [ ] 条件C。",
        "### I-1878-renamed\n- [ ] 条件A。\n- [ ] 条件B。\n### I-1878-second\n- [ ] 条件C。",
        "### I-1878-first\n- [ ] 条件A。\n### I-1878-second\n- [ ] 条件B。\n- [ ] 条件C。",
        "### I-1878-second\n- [ ] 条件C。\n### I-1878-first\n- [ ] 条件A。\n- [ ] 条件B。",
        "### I-1878-first\n- [ ] 条件A。\n- [ ] 条件B。",
        "### I-1878-first\n- [ ] 条件A。\n- [ ] 条件B。\n### I-1878-second\n- [ ] 条件C。\n### I-1878-third\n- [ ] 条件D。",
    ],
    ids=[
        "text",
        "candidate",
        "moved-item",
        "group-order",
        "removed-group",
        "added-group",
    ],
)
def test_grouped_acceptance_detects_changes_since_agreement(
    api: Any, current: str
) -> None:
    accepted = (
        "### I-1878-first\n- [ ] 条件A。\n- [ ] 条件B。\n"
        "### I-1878-second\n- [ ] 条件C。"
    )
    with pytest.raises(ValueError, match="acceptance criteria changed since agreement"):
        acceptance_requirements(api, accepted, current)


@pytest.mark.parametrize(
    "fault",
    [
        "omitted",
        "duplicate",
        "unmet",
        "unverified",
        "evidence",
        "remaining",
        "wrong_basis",
        "not_reviewed",
        "unknown_candidate",
        "candidate_type",
        "state_type",
        "result_type",
        "exported",
    ],
)
def test_completion_refuses_each_missing_obligation(api: Any, fault: str) -> None:
    spec = finish()
    candidates: dict[str, Any] = {}
    if fault == "omitted":
        spec["acceptance"].pop()
    elif fault == "duplicate":
        spec["acceptance"].append(spec["acceptance"][-1])
    elif fault in {"unmet", "unverified"}:
        spec["acceptance"][-1]["state"] = fault
    elif fault == "evidence":
        spec["acceptance"][-1]["evidence"] = " "
    elif fault == "remaining":
        spec["remaining_required"] = ["必須の実機確認が未実施。"]
    elif fault == "wrong_basis":
        spec["basis_ref"] = "another-agreement"
    elif fault == "not_reviewed":
        spec["improvements"]["reviewed"] = False
    elif fault == "unknown_candidate":
        spec["improvements"]["candidates"] = ["I-1-unknown"]
    elif fault == "candidate_type":
        spec["improvements"]["candidates"] = [1]
    elif fault == "state_type":
        spec["state"] = []
    elif fault == "result_type":
        spec["acceptance"][0]["state"] = {}
    else:
        spec["improvements"]["candidates"] = ["I-1-later"]
        candidates["later"] = {
            "candidate": {"data": {"spec": {"unmet_acceptance": ["ac-1"]}}}
        }
    with pytest.raises(ValueError):
        api.validate_finish(
            spec,
            {
                "basis_ref": "fixed-agreement",
                "criteria": [{"id": "agreement"}, {"id": "ac-1"}],
            },
            candidates,
            1,
        )


def test_no_candidates_is_normal_and_incomplete_is_not_completed(api: Any) -> None:
    spec = finish()
    required = {
        "basis_ref": "fixed-agreement",
        "criteria": [{"id": "agreement"}, {"id": "ac-1"}],
    }
    api.validate_finish(spec, required, {}, 1)
    spec["state"] = "incomplete"
    spec["acceptance"][-1]["state"] = "unmet"
    spec["remaining_required"] = ["必須作業を元親に残す。"]
    api.validate_finish(spec, required, {}, 1)


@pytest.mark.parametrize(
    "fault", ["missing", "job", "generation", "authorization", "response", "finish"]
)
def test_parent_final_requires_the_exact_report_receipt(api: Any, fault: str) -> None:
    job = {"id": "runtime-1", "authorization": {"generation": 1}}
    response = {
        "schema_version": 1,
        "state": "completed",
        "reason": "受入済み",
        "resume_condition": "なし",
    }
    receipt = {
        "job": "runtime-1",
        "generation": 1,
        "authorization": job["authorization"],
        "response_sha256": api.digest(response),
        "finish_sha256": "finish-1",
    }
    row = {
        "generation": 1,
        "completion": {"state": "completed", "finish_sha256": "finish-1"},
        "development_result": {"completion_receipt": receipt},
    }
    api.require_report_receipt(row, job, response)
    if fault == "missing":
        row.pop("development_result")
    else:
        field = {"response": "response_sha256", "finish": "finish_sha256"}.get(
            fault, fault
        )
        receipt[field] = "different"
    with pytest.raises(ValueError, match="matching workflow-report"):
        api.require_report_receipt(row, job, response)
