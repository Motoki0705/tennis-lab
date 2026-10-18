"""The source checkout must still match the commit and diff frozen in the plan."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import Field, SerializerFunctionWrapHandler

from .artifacts import read_document, verify_reference
from .contracts import ArtifactRef, Digest, Document, manifest_serializer
from .paths import AnnotationLayout

if TYPE_CHECKING:
    from .plan import CampaignPlan

_serialize = cast(
    Callable[[Callable[..., dict[str, object]]], Callable[..., dict[str, object]]],
    manifest_serializer,
)


class CodeBoundDocument(Document):
    code_revision: ArtifactRef | None = None

    @_serialize
    def serialize_code_binding(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        # Keep the exact canonical bytes of already recorded v1 inputs/launches.
        if self.code_revision is None:
            result.pop("code_revision", None)
        return result


class CodeRevision(Document):
    plan: ArtifactRef
    scope: ArtifactRef
    previous_runtime: ArtifactRef
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_diff_digest: Digest
    verification: ArtifactRef
    reason: str = Field(min_length=1)


def checkout_identity(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, check=True, capture_output=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if untracked:
        raise ValueError(
            "source checkout no longer matches the frozen commit/diff, or has untracked files"
        )
    return commit, hashlib.sha256(diff).hexdigest()


def require_frozen_code(
    root: Path,
    plan: CampaignPlan,
    *,
    layout: AnnotationLayout | None = None,
    revision: ArtifactRef | None = None,
    scope: ArtifactRef | None = None,
) -> None:
    expected = (plan.code_commit, plan.code_diff_digest)
    if revision is not None:
        from .plan import CampaignPlan as Plan

        if layout is None or scope is None:
            raise ValueError("a code revision requires its layout and resource scope")
        update = read_document(layout, revision, CodeRevision)
        if read_document(layout, update.plan, Plan) != plan or update.scope != scope:
            raise ValueError("code revision belongs to another plan or resource scope")
        verify_reference(layout, update.previous_runtime)
        verify_reference(layout, update.verification)
        if not update.reason.strip():
            raise ValueError("a code revision requires an explicit reason")
        expected = (update.code_commit, update.code_diff_digest)
    if checkout_identity(root) != expected:
        raise ValueError(
            "source checkout no longer matches the frozen commit/diff, or has untracked files"
        )
