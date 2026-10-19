"""Adopted artifacts must retain current dependencies throughout their lineage."""

from typing import Any

from src.tennis_scene.visualization.review.store import StoreReader


def test_stale_upstream_remains_stale_through_descendants() -> None:
    review = StoreReader.__new__(StoreReader)
    first = {"artifact_id": "new", "path": "new.json", "sha256": "newhash"}
    second = {"artifact_id": "second", "path": "second.json", "sha256": "secondhash"}
    review.references = {"producer": first, "consumer": second, "descendant": {"artifact_id": "third"}}
    review._active_by_artifact = {"new": "producer", "second": "consumer", "third": "descendant"}
    review._staleness = {}
    descriptors: dict[str, dict[str, Any]] = {"producer": {"dependencies": {}}, "consumer": {"dependencies": {"input": {"artifact_id": "old"}}},
                   "descendant": {"dependencies": {"input": second}}}
    review.descriptor = lambda node: descriptors[node]  # type: ignore[method-assign]
    assert "superseded" in review.stale_dependencies("consumer")[0]
    assert "consumer is stale" in review.stale_dependencies("descendant")[0]


def test_same_id_with_different_dependency_checksum_is_not_current() -> None:
    review = StoreReader.__new__(StoreReader)
    upstream = {"artifact_id": "new", "path": "new.json", "sha256": "newhash"}
    review.references = {"producer": upstream, "consumer": {"artifact_id": "second"}}
    review._active_by_artifact = {"new": "producer", "second": "consumer"}
    review._staleness = {}
    review.descriptor = lambda node: {"dependencies": {} if node == "producer" else {"input": {**upstream, "sha256": "wrong"}}}  # type: ignore[method-assign]
    assert "disagrees" in review.stale_dependencies("consumer")[0]
