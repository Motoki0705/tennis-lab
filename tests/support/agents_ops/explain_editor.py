"""Small real-store fixture and executable protocol fake for editor boundaries."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
editor: Any = importlib.import_module("explain_html.editor")
gemini: Any = importlib.import_module("explain_html.gemini")
schema: Any = importlib.import_module("explain_html.schema")
documents: Any = importlib.import_module("explain_html.document")
stores: Any = importlib.import_module("explain_html.store")
opsproc: Any = importlib.import_module("opsproc")

SOURCE = "https://example.invalid/calibration"
HTML = (
    '<html><body><section data-content-id="geometry">'
    '<p data-evidence-id="baseline">間隔は7 m。</p>'
    f'<a href="{SOURCE}">Calibration source</a></section>'
    '<aside data-content-id="limits"><p data-evidence-id="cameras">'
    "2 synchronized cameras; calibration is required.</p></aside></body></html>"
)


def document() -> Any:
    return schema.Document(
        document_id="geometry",
        purpose="Explain triangulation and limits.",
        audience="Readers who know pixel coordinates.",
        html=HTML,
        evidence=[
            schema.Evidence(
                evidence_id="baseline",
                statement="The baseline is 7 m.",
                values={"baseline_m": 7},
                conditions=["Known calibration."],
                provenance=schema.Provenance(kind="measured", source=SOURCE),
            ),
            schema.Evidence(
                evidence_id="cameras",
                statement="2 cameras are synchronized.",
                values={"count": 2},
                conditions=["Synchronized views."],
                provenance=schema.Provenance(
                    kind="reference", source="Supplied notes."
                ),
            ),
        ],
    )


def candidate(after: Any, *, current: Any = None, request: Any = None) -> Any:
    request = request or schema.EditRequest(
        document_id="geometry",
        base_revision=None,
        request_id="initial",
        target=schema.DocumentTarget(),
        instruction="Explain triangulation and missing inputs.",
    )
    return schema.Candidate(
        request=request,
        document=after,
        changes=documents.content_changes(current.document if current else None, after),
        evidence_changes=documents.evidence_changes(
            current.document if current else None,
            after,
            reason="Requested evidence change.",
        ),
        change_summary="Explain the supplied evidence and remaining unknowns.",
    )


def validation(value: Any) -> Any:
    return schema.ValidationResult(
        candidate_digest=schema.candidate_digest(value),
        **{
            name: schema.CheckResult(status="passed")
            for name in ("structural", "semantic", "rendering")
        },
    )


@dataclass
class EditorEnvironment:
    root: Path
    adapter: Any
    store: Any
    current: Any
    request: Any
    context: Any
    proposed: Any

    def outputs(self, value: Any, verdict: dict[str, Any] | None = None) -> None:
        (self.root / "writer.json").write_text(
            json.dumps(value.model_dump(mode="json"))
            if not isinstance(value, dict)
            else json.dumps(value)
        )
        if verdict is None:
            verdict = {
                "candidate_digest": schema.candidate_digest(value),
                "check": {"status": "passed", "issues": []},
                "reviewed_evidence_ids": ["baseline", "cameras"],
            }
        (self.root / "verifier.json").write_text(json.dumps(verdict))

    def calls(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (self.root / "calls.jsonl").read_text().splitlines()
        ]


@pytest.fixture
def editor_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EditorEnvironment:
    executable = tmp_path / "agy"
    executable.write_text(
        f"#!{sys.executable}\n"
        + Path(__file__).with_name("fake_explain_agy.py").read_text().split("\n", 1)[1]
    )
    executable.chmod(0o700)
    monkeypatch.setenv("FAKE_EXPLAIN_ROOT", str(tmp_path))
    monkeypatch.setenv("FAKE_EXPLAIN_CASE", "success")
    store = stores.RevisionStore(tmp_path / "store")
    initial = candidate(document())
    current = store.import_initial(initial, {}, validation(initial))
    request = schema.EditRequest(
        document_id="geometry",
        base_revision=current.revision_id,
        request_id="explain",
        target=schema.ContentTarget(content_id="geometry"),
        instruction="Explain why two rays locate a point; retain facts and citations.",
    )
    data = current.document.model_dump(mode="json")
    data["html"] = HTML.replace("間隔は", "校正したレイを交差させます。間隔は")
    changed = schema.parse(schema.Document, json.dumps(data))
    env = EditorEnvironment(
        tmp_path,
        editor.GeminiEditor(
            gemini.Gemini(tmp_path / "private", agy=executable),
            compile_document=lambda value: value,
        ),
        store,
        current,
        request,
        editor.WritingContext(
            purpose=changed.purpose,
            audience=changed.audience,
            evidence=changed.evidence,
            change_summaries=["Initial evidence."],
        ),
        candidate(changed, current=current, request=request),
    )
    env.outputs(env.proposed)
    return env
