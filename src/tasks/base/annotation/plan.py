"""Frozen campaign inputs, separate from mutable execution state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Self

from pydantic import Field

from .contracts import (
    ArtifactRef,
    Digest,
    Document,
    Identifier,
    ModelRef,
    PositiveInt,
    after_validator,
)
from .paths import AnnotationLayout


class CollectionCounts(Document):
    channels: PositiveInt
    videos_per_channel: PositiveInt
    max_clips_per_video: PositiveInt

    @property
    def videos(self) -> int:
        return self.channels * self.videos_per_channel

    @property
    def max_clips(self) -> int:
        return self.videos * self.max_clips_per_video


class StageCapacity(Document):
    stage: Identifier
    kind: Literal["codex", "cpu", "gpu", "io"]
    parallel: PositiveInt
    threads_per_process: PositiveInt


class ResourceBudget(Document):
    codex_workers: int = Field(ge=1, le=50)
    additional_storage_bytes: PositiveInt
    minimum_free_bytes: PositiveInt
    total_pss_bytes: PositiveInt
    gpu_seconds: PositiveInt
    stages: tuple[StageCapacity, ...] = Field(min_length=1)

    @after_validator
    def bounded_stages(self) -> Self:
        if len({s.stage for s in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        if (
            sum(s.parallel for s in self.stages if s.kind == "codex")
            > self.codex_workers
        ):
            raise ValueError("Codex stage allocations exceed the shared worker budget")
        return self


class ReviewPolicyRef(Document):
    policy: ArtifactRef
    preview_evidence: ArtifactRef
    decision: ArtifactRef


class LearningPlan(Document):
    recipe: ArtifactRef
    validation: ArtifactRef
    selection_policy: ArtifactRef | None


class ProductPlan(Document):
    product: Identifier
    dataset_task: Identifier
    dataset_version: Identifier
    component: Identifier
    base_snapshot: ArtifactRef | None
    generation_policy: ArtifactRef


class TaskPlan(Document):
    task: Identifier
    model: ModelRef
    inference: ArtifactRef
    review: ReviewPolicyRef | None
    learning: LearningPlan | None
    products: tuple[ProductPlan, ...] = Field(min_length=1)

    @after_validator
    def identities(self) -> Self:
        if self.task != self.model.task:
            raise ValueError("model task differs from its campaign task")
        if len({p.product for p in self.products}) != len(self.products):
            raise ValueError("product names must be unique within a task")
        return self


class CampaignPlan(Document):
    schema_version: Literal["annotation.plan.v1"] = "annotation.plan.v1"
    campaign_id: Identifier
    phase: Literal["preview", "production"]
    annotation_root: str
    counts: CollectionCounts
    selection: ArtifactRef
    preparation: ArtifactRef
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_diff_digest: Digest
    agent_model: Literal["gpt-6.1-sol"]
    agent_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"]
    resources: ResourceBudget
    tasks: tuple[TaskPlan, ...] = Field(min_length=1)

    @after_validator
    def consistent(self) -> Self:
        root = Path(self.annotation_root)
        if (
            not root.is_absolute()
            or str(root) != self.annotation_root
            or ".." in root.parts
        ):
            raise ValueError("annotation_root must be a normalized absolute path")
        if len({task.task for task in self.tasks}) != len(self.tasks):
            raise ValueError("campaign task names must be unique")
        if self.phase == "production" and any(
            task.review is None for task in self.tasks
        ):
            raise ValueError("production requires evidence-backed review policies")
        destinations = [
            (p.dataset_task, p.dataset_version, p.component)
            for task in self.tasks
            for p in task.products
        ]
        if len(set(destinations)) != len(destinations):
            raise ValueError("products must have unambiguous publication destinations")
        return self

    @property
    def allows_adoption(self) -> bool:
        return self.phase == "production"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_plan(path: Path, *, expected_digest: str | None = None) -> CampaignPlan:
    payload = path.read_bytes()
    json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    plan: CampaignPlan = CampaignPlan.model_validate_json(payload)
    if expected_digest is not None and plan.sha256 != expected_digest:
        raise ValueError("campaign plan hash mismatch")
    return plan


def save_plan(layout: AnnotationLayout, plan: CampaignPlan) -> Path:
    """Create once, or return exactly the same plan under the campaign lock."""
    plan = CampaignPlan.model_validate_json(plan.canonical_bytes())
    if plan.annotation_root != str(layout.root):
        raise ValueError("plan annotation root differs from the execution root")
    validate_inputs(layout, plan)
    directory = layout.path("campaigns", plan.campaign_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = layout.path("campaigns", plan.campaign_id, "plan.json")
    lock = layout.path("campaigns", plan.campaign_id, "plan.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        layout.path("campaigns", plan.campaign_id, "plan.json")
        if path.exists():
            load_plan(path, expected_digest=plan.sha256)
            return path
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                temporary = handle.name
                handle.write(plan.canonical_bytes() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    return path


def references(document: Document) -> Iterator[ArtifactRef]:
    """Enumerate typed input references without interpreting any task payload."""
    if isinstance(document, ArtifactRef):
        yield document
        return
    for name in type(document).model_fields:
        value: object = getattr(document, name)
        if isinstance(value, Document):
            yield from references(value)
        elif isinstance(value, tuple):
            for child in value:
                if isinstance(child, Document):
                    yield from references(child)


def validate_inputs(layout: AnnotationLayout, plan: CampaignPlan) -> None:
    seen: set[ArtifactRef] = set()
    for reference in references(plan):
        if reference in seen:
            continue
        path = layout.artifact_path(reference)
        if not path.is_file():
            raise ValueError(f"pinned input is not a file: {path}")
        if (
            path.stat().st_size != reference.size_bytes
            or file_digest(path) != reference.digest
        ):
            raise ValueError(f"pinned input hash/size mismatch: {path}")
        seen.add(reference)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
