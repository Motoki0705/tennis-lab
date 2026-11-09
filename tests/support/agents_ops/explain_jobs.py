"""Controllable editor/visual boundaries, with real revision and request storage."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.agents_ops.explain_html import (
    PAYLOADS,
    candidate_for,
    document_api,
    make_document,
    schema,
    store_api,
    validation_for,
)

jobs: Any = importlib.import_module("explain_html.jobs")
service: Any = importlib.import_module("explain_html.service")


@dataclass
class Gate:
    entered: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    cancel: Event | None = None

    def hold(self, cancel: Event) -> None:
        self.cancel = cancel
        self.entered.set()
        assert self.release.wait(5), "The test did not release its gate."


class FakeEditor:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.contexts: list[Any] = []
        self.gates: dict[str, Gate] = {}
        self.damage: str | None = None

    def edit(self, request: Any, current: Any, context: Any, *, cancel: Event) -> Any:
        self.calls.append(request)
        self.contexts.append(context)
        if request.request_id in self.gates:
            self.gates[request.request_id].hold(cancel)
        document = current.document.model_copy(
            update={
                "html": current.document.html.replace(
                    "Then measure 7 m.",
                    "Then measure 7 m. The two calibrated rays give an example.",
                )
            }
        )
        candidate = schema.Candidate(
            request=request,
            document=document,
            changes=document_api.content_changes(current.document, document),
            evidence_changes=[],
            change_summary="Add the calibrated-ray example.",
        )
        validation = validation_for(candidate, rendering={"status": "not_run"})
        if self.damage == "digest":
            validation = validation.model_copy(update={"candidate_digest": "0" * 64})
        return SimpleNamespace(candidate=candidate, validation=validation)


class FakeValidator:
    def __init__(self) -> None:
        self.gates: dict[str, Gate] = {}
        self.seen: list[tuple[Any, Mapping[str, bytes]]] = []
        self.reject = False

    def __call__(
        self, candidate: Any, assets: Mapping[str, bytes], *, cancel: Event
    ) -> Any:
        self.seen.append((candidate, dict(assets)))
        if candidate.request.request_id in self.gates:
            self.gates[candidate.request.request_id].hold(cancel)
        return schema.CheckResult(status="failed" if self.reject else "passed")


def fake_rendering(
    candidate: Any, assets: Mapping[str, bytes], *, cancel: Event
) -> Any:
    """CLI test double only; not a production rendering validator."""
    assert "data-content-id" in candidate.document.html
    assert not assets
    assert not cancel.is_set()
    return schema.CheckResult(status="passed")


class JobWorld:
    def __init__(self, root: Path):
        self.root = root
        self.store = store_api.RevisionStore(root)
        candidate = candidate_for(make_document())
        self.first = self.store.import_initial(
            candidate, PAYLOADS, validation_for(candidate)
        )
        self.editor, self.validator = FakeEditor(), FakeValidator()
        self.commits: list[str] = []
        self.store.fault = lambda checkpoint: (
            self.commits.append(checkpoint)
            if checkpoint == "before_current_replace"
            else None
        )
        self.controllers: list[Any] = []

    def controller(self, **kwargs: Any) -> Any:
        controller = jobs.JobController(
            self.store, self.editor, self.validator, **kwargs
        )
        self.controllers.append(controller)
        return controller

    def request(self, name: str = "edit", **overrides: Any) -> Any:
        return schema.EditRequest(
            document_id="ray-geometry",
            base_revision=self.first.revision_id,
            request_id=name,
            target=schema.ContentTarget(content_id="geometry"),
            instruction="Explain how the two calibrated rays give an example.",
        ).model_copy(update=overrides)

    def finish(self) -> None:
        for gate in [*self.editor.gates.values(), *self.validator.gates.values()]:
            gate.release.set()
        for controller in self.controllers:
            controller.close()


@pytest.fixture
def job_world(tmp_path: Path) -> Iterator[JobWorld]:
    world = JobWorld(tmp_path / "runtime")
    try:
        yield world
    finally:
        world.finish()
