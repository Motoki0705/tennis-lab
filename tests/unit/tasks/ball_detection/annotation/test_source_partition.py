"""Source replacements retain a reproducible split without label-based choices."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    SourcePartition,
    SourceSplit,
    Split,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.partition import (
    publication_splits,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import PublicationReceipt, content_digest
from src.tasks.base.annotation.plan import CollectionCounts
from tests.support.tasks.ball_detection.annotation_learning import publish


def policy() -> BallPublicationPolicy:
    return BallPublicationPolicy(
        jpeg_quality=90,
        source_splits=(),
        source_partition=SourcePartition(
            seed="fixed-before-collection",
            train_weight=8,
            validation_weight=1,
            test_weight=1,
        ),
    )


def test_complete_source_partition_is_exact_stable_and_preserves_old_holdouts() -> None:
    sources = {content_digest(["youtube", str(i)]) for i in range(100)}
    existing: dict[str, Split] = {
        content_digest("old-train"): "train",
        content_digest("old-val"): "val",
        content_digest("old-test"): "test",
    }
    first = publication_splits(policy(), existing, sources, expected_new_sources=100)
    assert Counter(first[s] for s in sources) == {"train": 80, "val": 10, "test": 10}
    assert {s: first[s] for s in existing} == existing
    assert (
        publication_splits(
            policy(),
            existing,
            frozenset(reversed(sorted(sources))),
            expected_new_sources=100,
        )
        == first
    )
    # A rejected candidate is replaced before publication; no plan/source-list rewrite.
    replaced = (sources - {min(sources)}) | {content_digest("replacement")}
    second = publication_splits(policy(), existing, replaced, expected_new_sources=100)
    assert Counter(second[s] for s in replaced) == {"train": 80, "val": 10, "test": 10}
    assert min(sources) not in second


def test_partition_rejects_partial_sources_and_ambiguous_or_inexact_policy() -> None:
    with pytest.raises(ValueError, match="complete planned"):
        publication_splits(
            policy(),
            {},
            {content_digest(i) for i in range(99)},
            expected_new_sources=100,
        )
    with pytest.raises(ValueError, match="exact source split"):
        publication_splits(
            policy(),
            {},
            {content_digest(i) for i in range(99)},
            expected_new_sources=99,
        )
    with pytest.raises(ValueError, match="Choose explicit"):
        BallPublicationPolicy(
            jpeg_quality=90,
            source_splits=(SourceSplit(source_id="a" * 64, split="train"),),
            source_partition=policy().source_partition,
        )


def test_publisher_applies_recipe_to_new_sources_and_keeps_base(tmp_path: Path) -> None:
    def setup(layout, base):
        recipe = BallPublicationPolicy(
            jpeg_quality=90,
            image_storage="video_q90",
            source_splits=(),
            source_partition=SourcePartition(
                seed="fixture", train_weight=1, validation_weight=1, test_weight=1
            ),
        )
        ref = write_document(
            layout, layout.path("campaigns", "partition-policy.json"), recipe
        )

        def transform(plan):
            task = plan.tasks[0]
            product = task.products[0].model_copy(update={"generation_policy": ref})
            return plan.model_copy(
                update={
                    "counts": CollectionCounts(
                        channels=1, videos_per_channel=3, max_clips_per_video=1
                    ),
                    "tasks": (task.model_copy(update={"products": (product,)}),),
                }
            )

        return {"plan_transform": transform}

    env, publication, _ = publish(
        tmp_path, image_storage="video_q90", new_splits=("train",) * 3, setup=setup
    )
    receipt = read_document(env.layout, publication, PublicationReceipt)
    snapshot, store = load_snapshot(env.layout, receipt.snapshot)
    added = set(receipt.added_clips)
    assert Counter(c.split for c in snapshot.clips if c.clip_id in added) == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert [c.split for c in store.clips[:3]] == ["train", "val", "test"]
    assert len({c.source_id for c in snapshot.clips if c.clip_id in added}) == 3
