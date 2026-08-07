"""Unit tests for the staged variable-T sampling utilities (issue #579)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from src.tasks.ball_detection.data.components.staged_sampler import (
    ConcatVariableTDataset,
    FixedTDataset,
    VariableTBatchSampler,
    accumulation_for,
    linear_decreasing_t_probs,
)
from src.tasks.ball_detection.data.types import BallDetectionSample


class TestLinearDecreasingTProbs:
    def test_t1_gets_half_and_tail_sums_to_half(self) -> None:
        probs = linear_decreasing_t_probs(8, 0.5)
        assert probs[1] == pytest.approx(0.5)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert sum(v for t, v in probs.items() if t > 1) == pytest.approx(0.5)

    def test_tail_is_strictly_decreasing(self) -> None:
        probs = linear_decreasing_t_probs(8, 0.5)
        assert all(probs[t] > probs[t + 1] for t in range(2, 8))

    def test_t_max_one_is_degenerate(self) -> None:
        assert linear_decreasing_t_probs(1) == {1: 1.0}


class TestAccumulationFor:
    def test_rounds_to_nearest_and_floors_at_one(self) -> None:
        assert accumulation_for(21, 21) == 1
        assert accumulation_for(21, 7) == 3
        assert accumulation_for(20, 3) == 7  # round(6.67) -> 7
        assert accumulation_for(8, 100) == 1


class _RangeDataset(Dataset[BallDetectionSample]):
    """Minimal dataset that echoes the (index, T) it was asked for."""

    def __init__(self, n: int, tag: str) -> None:
        self.n = n
        self.tag = tag

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int | tuple[int, int]) -> BallDetectionSample:
        return {
            "images": torch.empty(0),
            "heatmaps": torch.empty(0),
            "coords": torch.empty(0),
            "visibility": torch.empty(0),
            "original_size": torch.empty(0),
            "heatmap_size": torch.empty(0),
            "window_id": f"{self.tag}:{index!r}",
        }


class TestVariableTBatchSampler:
    def _sampler(self) -> VariableTBatchSampler:
        return VariableTBatchSampler(
            num_samples=300,
            t_probs=linear_decreasing_t_probs(8, 0.5),
            batch_size_by_t={1: 8, 2: 6, 3: 4, 4: 3, 5: 3, 6: 2, 7: 2, 8: 2},
            effective_batch=8,
            seed=0,
        )

    def test_each_microbatch_is_uniform_T_and_correct_size(self) -> None:
        sampler = self._sampler()
        for batch in sampler:
            ts = {t for _, t in batch}
            assert len(ts) == 1
            t = next(iter(ts))
            assert len(batch) == sampler.batch_size_by_t[t]

    def test_iterations_vary_across_epochs(self) -> None:
        sampler = self._sampler()
        first = [tuple(b) for b in sampler]
        second = [tuple(b) for b in sampler]
        assert first != second

    @pytest.mark.parametrize(
        ("num_samples", "physical_batch", "effective_batch"),
        [
            (16, 2, 8),
            (17, 2, 8),
            (23, 2, 8),  # scaled-down 10407 / B=2 / EBS=8 trailing-remainder case
            (14, 3, 8),
            (7, 2, 8),
        ],
    )
    def test_fixed_t_len_matches_yielded_batches(
        self,
        num_samples: int,
        physical_batch: int,
        effective_batch: int,
    ) -> None:
        sampler = VariableTBatchSampler(
            num_samples=num_samples,
            t_probs={8: 1.0},
            batch_size_by_t={8: physical_batch},
            effective_batch=effective_batch,
            seed=11,
        )
        accumulate = accumulation_for(effective_batch, physical_batch)
        expected_batches = (
            num_samples // (physical_batch * accumulate)
        ) * accumulate

        assert len(sampler) == expected_batches
        assert sum(1 for _ in sampler) == expected_batches

    @pytest.mark.parametrize("seed", [0, 7, 1234])
    def test_variable_t_len_matches_yielded_batches_across_epochs(
        self, seed: int
    ) -> None:
        sampler = VariableTBatchSampler(
            num_samples=137,
            t_probs=linear_decreasing_t_probs(8, 0.5),
            batch_size_by_t={1: 8, 2: 6, 3: 4, 4: 3, 5: 3, 6: 2, 7: 2, 8: 2},
            effective_batch=8,
            seed=seed,
        )

        for epoch in range(6):
            sampler.set_epoch(epoch)
            expected_batches = len(sampler)

            assert len(sampler) == expected_batches
            assert sum(1 for _ in sampler) == expected_batches

    def test_missing_batch_size_entry_raises(self) -> None:
        with pytest.raises(ValueError):
            VariableTBatchSampler(
                num_samples=10,
                t_probs={1: 0.5, 2: 0.5},
                batch_size_by_t={1: 4},
                effective_batch=4,
            )


class TestConcatVariableTDataset:
    def test_routes_tuple_index_to_local_subdataset(self) -> None:
        concat = ConcatVariableTDataset([_RangeDataset(5, "a"), _RangeDataset(3, "b")])
        assert len(concat) == 8
        # global 4 -> last of dataset "a" at local 4
        assert concat[(4, 3)]["window_id"] == "a:(4, 3)"
        # global 5 -> first of dataset "b" at local 0
        assert concat[(5, 7)]["window_id"] == "b:(0, 7)"

    def test_routes_bare_int_index(self) -> None:
        concat = ConcatVariableTDataset([_RangeDataset(2, "a"), _RangeDataset(2, "b")])
        assert concat[3]["window_id"] == "b:1"


class TestFixedTDataset:
    def test_forwards_fixed_num_frames(self) -> None:
        view = FixedTDataset(_RangeDataset(4, "a"), num_frames=1)
        assert len(view) == 4
        assert view[2]["window_id"] == "a:(2, 1)"
