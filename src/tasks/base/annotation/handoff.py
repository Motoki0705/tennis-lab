"""Bounded, revision-pinned task summaries and paged edit history for continuation."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from .artifacts import read_document, verify_reference, write_document
from .contracts import ArtifactRef, ClipManifest, Document, FrameRange, Identifier
from .editing.contracts import ReviewAdapter, ReviewContext, Revision
from .editing.store import RevisionConflict, RevisionStore
from .paths import AnnotationLayout
from .plan import references


class SummaryRow(Document):
    frame_index: int = Field(ge=0)
    row_id: str = Field(min_length=1, max_length=128)
    state: Identifier
    detail: str = Field(max_length=500)


class Boundary(Document):
    frame_index: int = Field(ge=0)
    kind: Identifier


class HandoffContent(Document):
    rows: tuple[SummaryRow, ...] = Field(max_length=64)
    total_rows: int = Field(ge=0)
    remaining_rows: int = Field(ge=0)
    next_row_cursor: str | None = Field(max_length=128)
    reviewed_ranges: tuple[FrameRange, ...] = Field(max_length=64)
    unresolved_ranges: tuple[FrameRange, ...] = Field(max_length=64)
    boundaries: tuple[Boundary, ...] = Field(max_length=128)
    next_actions: tuple[str, ...] = Field(min_length=1, max_length=8)


class Handoff(Document):
    context: ArtifactRef
    revision: ArtifactRef
    scope: FrameRange
    after_row: str | None
    content: HandoffContent
    adoption: ArtifactRef | None
    previous_revision: ArtifactRef | None
    commands: tuple[ArtifactRef, ...] = Field(max_length=20)
    history_cursor: ArtifactRef | None


class HandoffAdapter(ReviewAdapter, Protocol):
    def summarize(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: ArtifactRef,
        scope: FrameRange,
        after_row: str | None,
    ) -> HandoffContent: ...


def build_handoff(
    store: RevisionStore,
    context_ref: ArtifactRef,
    adapter: HandoffAdapter,
    scope: FrameRange,
    *,
    previous_revision: ArtifactRef | None = None,
    after_row: str | None = None,
    expected_revision: ArtifactRef | None = None,
) -> ArtifactRef:
    layout = store.layout
    context = read_document(layout, context_ref, ReviewContext)
    _, manifest, _ = store.context(context, adapter)
    if scope.stop > len(manifest.frames) or scope.stop - scope.start > 64:
        raise ValueError(
            "handoff scope must lie in the clip and contain at most 64 frames"
        )
    if after_row is not None and (not after_row or len(after_row) > 128):
        raise ValueError("invalid task row cursor")
    revision_ref, adoption = store.current(context_ref, adapter)
    if after_row is not None and expected_revision is None:
        raise ValueError("continued row pages must pin their starting revision")
    if expected_revision is not None and expected_revision != revision_ref:
        raise RevisionConflict(
            "handoff revision changed; restart from the current revision"
        )
    revision = read_document(layout, revision_ref, Revision)
    verify_reference(layout, revision.payload)
    content = adapter.summarize(
        layout, context, manifest, revision.payload, scope, after_row
    )
    accounted = len(content.rows) + content.remaining_rows
    if accounted > content.total_rows or (
        after_row is None and accounted != content.total_rows
    ):
        raise ValueError("handoff row counts are inconsistent with the cursor")
    if (content.remaining_rows > 0) != (content.next_row_cursor is not None) or (
        content.remaining_rows and not content.rows
    ):
        raise ValueError(
            "truncated task rows require a progressing continuation cursor"
        )
    if any(not scope.start <= row.frame_index < scope.stop for row in content.rows):
        raise ValueError("handoff row lies outside the requested scope")
    if len({(row.frame_index, row.row_id) for row in content.rows}) != len(
        content.rows
    ):
        raise ValueError("handoff rows must have distinct frame/row identities")
    if content.next_row_cursor is not None and content.next_row_cursor == after_row:
        raise ValueError("task row cursor did not advance")
    for ranges in (content.reviewed_ranges, content.unresolved_ranges):
        if any(r.start < scope.start or r.stop > scope.stop for r in ranges) or any(
            a.stop > b.start for a, b in zip(ranges, ranges[1:], strict=False)
        ):
            raise ValueError(
                "handoff coverage must be ordered, disjoint and inside the scope"
            )
    if any(not scope.start <= b.frame_index <= scope.stop for b in content.boundaries):
        raise ValueError("handoff boundary lies outside the requested scope")
    if any(not action.strip() or len(action) > 500 for action in content.next_actions):
        raise ValueError("handoff next actions must be short and nonempty")
    verify_reference(layout, revision.payload)
    commands, cursor = (
        store.changes_since(revision_ref, previous_revision)
        if previous_revision is not None
        else ((), None)
    )
    result = Handoff(
        context=context_ref,
        revision=revision_ref,
        scope=scope,
        after_row=after_row,
        content=content,
        adoption=adoption,
        previous_revision=previous_revision,
        commands=commands,
        history_cursor=cursor,
    )
    return write_document(
        layout,
        layout.path(
            "annotations",
            context.task,
            manifest.clip_id,
            "handoffs",
            result.sha256 + ".json",
        ),
        result,
    )


def read_handoff(layout: AnnotationLayout, reference: ArtifactRef) -> Handoff:
    handoff: Handoff = read_document(layout, reference, Handoff)
    for ref in references(handoff):
        verify_reference(layout, ref)
    revision = read_document(layout, handoff.revision, Revision)
    if revision.context != handoff.context:
        raise ValueError("handoff references a different review context")
    verify_reference(layout, revision.payload)
    return handoff
