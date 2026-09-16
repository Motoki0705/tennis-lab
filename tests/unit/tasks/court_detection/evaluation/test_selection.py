"""Deterministic, group-balanced sample selection."""

from __future__ import annotations

from src.tasks.court_detection.evaluation.contracts import SampleRef
from src.tasks.court_detection.evaluation.datasets import select_manifest_refs
from src.tasks.court_detection.evaluation.settings import BenchmarkSelectionSettings


def _ref(
    sample_id: str,
    *,
    scene: str = "B00",
    group: str | None = "group-1",
    domain: str = "synthetic_test",
) -> SampleRef:
    return SampleRef(
        domain=domain,  # type: ignore[arg-type]
        sample_id=sample_id,
        scene_id=scene,
        trajectory_group_id=group,
        split="test",
        source_target_sha256="c" * 64,
        width=960,
        height=540,
    )


def _selection(maximum: int | None, seed: int = 7) -> BenchmarkSelectionSettings:
    return BenchmarkSelectionSettings(seed=seed, max_samples_per_domain=maximum)


def test_without_a_cap_every_sample_is_returned_in_stable_order() -> None:
    refs = [_ref("b"), _ref("a"), _ref("c")]

    selected = select_manifest_refs(refs, selection=_selection(None))

    assert [item.sample_id for item in selected] == ["a", "b", "c"]


def test_sampling_is_reproducible_for_a_given_seed() -> None:
    refs = [
        _ref(f"sample-{index:03d}", group=f"group-{index % 5}") for index in range(25)
    ]

    first = select_manifest_refs(refs, selection=_selection(10, seed=3))
    second = select_manifest_refs(refs, selection=_selection(10, seed=3))

    assert [item.sample_id for item in first] == [item.sample_id for item in second]


def test_a_seed_change_moves_the_within_group_offset() -> None:
    refs = [
        _ref(f"sample-{index:03d}", group=f"group-{index % 4}") for index in range(16)
    ]

    first = select_manifest_refs(refs, selection=_selection(8, seed=1))
    second = select_manifest_refs(refs, selection=_selection(8, seed=2))

    assert [item.sample_id for item in first] != [item.sample_id for item in second]


def test_a_cap_spreads_samples_across_groups_before_repeating_one() -> None:
    refs = [_ref(f"a-{index}", group="group-a") for index in range(6)] + [
        _ref(f"b-{index}", group="group-b") for index in range(6)
    ]

    selected = select_manifest_refs(refs, selection=_selection(4))

    groups = [item.trajectory_group_id for item in selected]
    assert groups.count("group-a") == 2
    assert groups.count("group-b") == 2


def test_samples_without_a_trajectory_group_are_their_own_bucket() -> None:
    refs = [
        _ref("real-1", group=None, domain="real_validation"),
        _ref("real-2", group=None, domain="real_validation"),
    ]

    selected = select_manifest_refs(refs, selection=_selection(1))

    assert len(selected) == 1
