"""Tests for the extensible typed augmentation registry."""

from typing import cast

import numpy as np
import pytest
import torch

from src.tasks.court_alignment.data.augmentation import (
    AugmentableGroundCourtSample,
    ComposeAugmentation,
    GroundCourtAugmentationConfig,
    RandomGhostLines,
    RandomHeatmapBlur,
    RandomLineDropout,
    RandomLineMorphology,
    RandomProbabilityNoise,
    build_augmentation,
    build_augmentations,
)
from src.tasks.court_alignment.data.datamodule import GroundCourtDataModule
from src.tasks.court_alignment.data.dataset import GroundCourtDataset
from src.tasks.court_alignment.data.splits import GroundCourtSplit


def _sample(*, dtype: torch.dtype = torch.float32) -> AugmentableGroundCourtSample:
    image = torch.zeros((1, 32, 40), dtype=dtype)
    image[0, 8:24, 10] = 1.0
    image[0, 8:24, 29] = 1.0
    image[0, 8, 10:30] = 1.0
    image[0, 23, 10:30] = 1.0
    return AugmentableGroundCourtSample(
        image=image,
        keypoints=torch.arange(28, dtype=torch.float32).reshape(1, 14, 2),
        visibility=torch.ones((1, 14), dtype=torch.bool),
        centers=torch.tensor([[19.5, 15.5]], dtype=torch.float32),
        instance_ids=torch.tensor([7], dtype=torch.long),
    )


def _combined() -> ComposeAugmentation:
    result = build_augmentations(
        [
            {
                "name": "random_line_morphology",
                "params": {
                    "probability": 1.0,
                    "dilate_probability": 1.0,
                    "kernel_size_choices": [3],
                    "iterations": 1,
                },
            },
            {
                "name": "random_heatmap_blur",
                "params": {"probability": 1.0, "sigma_range": [0.5, 1.5]},
            },
            {
                "name": "random_line_dropout",
                "params": {
                    "probability": 1.0,
                    "gap_count_range": [1, 3],
                    "gap_length_px_range": [2.0, 8.0],
                    "gap_width_px_range": [1.0, 3.0],
                },
            },
            {
                "name": "random_ghost_lines",
                "params": {
                    "probability": 1.0,
                    "copy_count_range": [1, 2],
                    "offset_px_range": [3, 6],
                    "amplitude_range": [0.55, 0.9],
                    "long_line_count_range": [1, 2],
                    "long_line_length_px_range": [10.0, 25.0],
                    "long_line_width_px_range": [1.0, 3.0],
                    "long_line_amplitude_range": [0.4, 0.8],
                },
            },
            {
                "name": "random_probability_noise",
                "params": {
                    "probability": 1.0,
                    "foreground_amplitude_range": [0.55, 1.0],
                    "gamma_range": [0.7, 1.5],
                    "speckle_fraction_range": [0.002, 0.008],
                    "speckle_amplitude_range": [0.2, 0.9],
                    "additive_std": 0.0,
                },
            },
        ]
    )
    assert isinstance(result, ComposeAugmentation)
    return result


def test_identity_is_explicit_baseline() -> None:
    augmentation = build_augmentation(GroundCourtAugmentationConfig(name="identity"))
    assert augmentation.__class__.__name__ == "IdentityAugmentation"


def test_unknown_augmentation_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown ground-court augmentation"):
        build_augmentation("does-not-exist")


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown augmentation config fields"):
        GroundCourtAugmentationConfig.from_mapping({"unknown": "identity"})


def test_datamodule_resolves_unknown_transform_during_construction() -> None:
    with pytest.raises(ValueError, match="Unknown ground-court augmentation"):
        GroundCourtDataModule(
            train_samples=1,
            val_samples=0,
            test_samples=0,
            augmentations=[{"type": "does-not-exist"}],
        )


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        (torch.zeros((1, 8, 8), dtype=torch.long), TypeError, "floating-point"),
        (torch.full((1, 8, 8), float("nan")), ValueError, "finite"),
        (torch.full((1, 8, 8), 1.1), ValueError, r"\[0,1\]"),
        (torch.zeros((1, 8, 8)), ValueError, "positive line evidence"),
        (torch.ones((1, 8, 8)), ValueError, "background evidence"),
    ],
)
def test_sample_rejects_invalid_image_contract(
    image: torch.Tensor, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        AugmentableGroundCourtSample(
            image=image,
            keypoints=torch.zeros((1, 14, 2)),
            visibility=torch.ones((1, 14), dtype=torch.bool),
            centers=torch.zeros((1, 2)),
            instance_ids=torch.zeros((1,), dtype=torch.long),
        )


def test_sample_rejects_non_float_or_non_finite_geometry() -> None:
    sample = _sample()
    with pytest.raises(TypeError, match="keypoints must have a floating-point"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=sample.keypoints.long(),
            visibility=sample.visibility,
            centers=sample.centers,
            instance_ids=sample.instance_ids,
        )
    with pytest.raises(ValueError, match="keypoints must be finite"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=sample.keypoints.clone().masked_fill(
                torch.arange(28).reshape(1, 14, 2) == 0, float("nan")
            ),
            visibility=sample.visibility,
            centers=sample.centers,
            instance_ids=sample.instance_ids,
        )
    with pytest.raises(TypeError, match="centers must have a floating-point"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=sample.keypoints,
            visibility=sample.visibility,
            centers=sample.centers.long(),
            instance_ids=sample.instance_ids,
        )
    with pytest.raises(ValueError, match="centers must be finite"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=sample.keypoints,
            visibility=sample.visibility,
            centers=torch.full_like(sample.centers, float("inf")),
            instance_ids=sample.instance_ids,
        )


def test_sample_rejects_invalid_instance_ids_and_geometry_device() -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="visibility must have shape"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=sample.keypoints,
            visibility=sample.visibility.float(),
            centers=sample.centers,
            instance_ids=sample.instance_ids,
        )
    with pytest.raises(ValueError, match="instance_ids must be int64"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=sample.keypoints,
            visibility=sample.visibility,
            centers=sample.centers,
            instance_ids=sample.instance_ids.int(),
        )
    duplicate_geometry = {
        "image": sample.image,
        "keypoints": sample.keypoints.repeat(2, 1, 1),
        "visibility": sample.visibility.repeat(2, 1),
        "centers": sample.centers.repeat(2, 1),
    }
    with pytest.raises(ValueError, match="non-negative"):
        AugmentableGroundCourtSample(
            **duplicate_geometry,
            instance_ids=torch.tensor([0, -1], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="unique"):
        AugmentableGroundCourtSample(
            **duplicate_geometry,
            instance_ids=torch.tensor([4, 4], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="same device"):
        AugmentableGroundCourtSample(
            image=sample.image,
            keypoints=torch.empty((1, 14, 2), device="meta"),
            visibility=sample.visibility,
            centers=sample.centers,
            instance_ids=sample.instance_ids,
        )


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("random_line_morphology", {"kernel_size_choices": [2]}, "positive odd"),
        ("random_heatmap_blur", {"probability": 1.1}, r"\[0,1\]"),
        ("random_line_dropout", {"gap_count_range": [0, 1]}, "minimum 1"),
        ("random_ghost_lines", {"offset_px_range": [5, 2]}, "ordered"),
        ("random_probability_noise", {"gamma_range": [0.0, 1.0]}, "positive"),
        ("random_heatmap_blur", {"unknown": 1}, "Unknown augmentation parameters"),
    ],
)
def test_registered_augmentations_reject_invalid_parameters(
    name: str, params: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_augmentation({"name": name, "params": params})


def test_morphology_dilates_and_preserves_dtype_and_range() -> None:
    sample = _sample(dtype=torch.float64)
    transform = RandomLineMorphology(
        probability=1.0,
        dilate_probability=1.0,
        kernel_size_choices=(3,),
        iterations=1,
    )

    result = transform(sample, np.random.default_rng(1))

    assert torch.count_nonzero(result.image) > torch.count_nonzero(sample.image)
    assert result.image.dtype == torch.float64
    assert 0.0 <= float(result.image.min()) <= float(result.image.max()) <= 1.0


def test_morphology_never_accepts_an_empty_erosion_proposal() -> None:
    sample = _sample()
    transform = RandomLineMorphology(
        probability=1.0,
        dilate_probability=0.0,
        kernel_size_choices=(3,),
        iterations=1,
    )

    result = transform(sample, np.random.default_rng(1))

    assert torch.equal(result.image, sample.image)


def test_blur_produces_continuous_values() -> None:
    sample = _sample()
    transform = RandomHeatmapBlur(probability=1.0, sigma_range=(1.0, 1.0))

    result = transform(sample, np.random.default_rng(2))

    fractional = (result.image > 0.0) & (result.image < 1.0)
    assert bool(fractional.any())
    assert not torch.equal(result.image, sample.image)


def test_dropout_erases_evidence_but_forbids_total_deletion() -> None:
    sample = _sample()
    transform = RandomLineDropout(
        probability=1.0,
        gap_count_range=(8, 8),
        gap_length_px_range=(12.0, 12.0),
        gap_width_px_range=(5.0, 5.0),
    )

    result = transform(sample, np.random.default_rng(3))

    assert torch.count_nonzero(result.image) < torch.count_nonzero(sample.image)
    assert torch.count_nonzero(result.image) > 0


def test_ghost_lines_add_shifted_and_long_evidence() -> None:
    sample = _sample()
    transform = RandomGhostLines(
        probability=1.0,
        copy_count_range=(2, 2),
        offset_px_range=(4, 4),
        amplitude_range=(0.7, 0.7),
        long_line_count_range=(2, 2),
        long_line_length_px_range=(20.0, 20.0),
        long_line_width_px_range=(2.0, 2.0),
        long_line_amplitude_range=(0.6, 0.6),
    )

    result = transform(sample, np.random.default_rng(4))

    assert torch.count_nonzero(result.image) > torch.count_nonzero(sample.image)
    assert bool((result.image == 0.7).any() | (result.image == 0.6).any())


def test_ghost_copy_draws_and_applies_amplitude_exactly_once() -> None:
    sample = _sample(dtype=torch.float64)
    transform = RandomGhostLines(
        probability=1.0,
        copy_count_range=(1, 1),
        offset_px_range=(4, 4),
        amplitude_range=(0.35, 0.70),
        long_line_count_range=(0, 0),
    )
    seed = 1_937
    expected_rng = np.random.default_rng(seed)
    _ = expected_rng.random()  # probability gate
    angle = float(expected_rng.uniform(0.0, 2.0 * np.pi))
    amplitude = float(expected_rng.uniform(0.35, 0.70))
    expected_next_draw = float(expected_rng.random())
    dx = int(round(4 * np.cos(angle)))
    dy = int(round(4 * np.sin(angle)))
    if dx == 0 and dy == 0:
        dx = 4
    shifted = torch.roll(sample.image, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0.0
    elif dy < 0:
        shifted[..., dy:, :] = 0.0
    if dx > 0:
        shifted[..., :, :dx] = 0.0
    elif dx < 0:
        shifted[..., :, dx:] = 0.0
    expected_image = torch.maximum(sample.image, shifted * amplitude)

    actual_rng = np.random.default_rng(seed)
    result = transform(sample, actual_rng)

    torch.testing.assert_close(result.image, expected_image, rtol=0.0, atol=0.0)
    ghost_only = (shifted > 0.0) & (sample.image == 0.0)
    assert bool(ghost_only.any())
    torch.testing.assert_close(
        result.image[ghost_only],
        torch.full_like(result.image[ghost_only], amplitude),
        rtol=0.0,
        atol=0.0,
    )
    assert 0.35 <= amplitude <= 0.70
    assert float(actual_rng.random()) == expected_next_draw


def test_probability_noise_adds_fractional_speckles() -> None:
    sample = _sample()
    transform = RandomProbabilityNoise(
        probability=1.0,
        foreground_amplitude_range=(0.8, 0.8),
        gamma_range=(1.0, 1.0),
        speckle_fraction_range=(0.01, 0.01),
        speckle_amplitude_range=(0.3, 0.3),
        additive_std=0.0,
    )

    result = transform(sample, np.random.default_rng(5))

    assert bool((result.image == 0.8).any())
    assert bool((result.image == 0.3).any())
    assert torch.count_nonzero(result.image) > torch.count_nonzero(sample.image)


def test_combined_pipeline_is_seed_deterministic_and_geometry_invariant() -> None:
    sample = _sample()
    transform = _combined()

    first = transform(sample, np.random.default_rng(123))
    replay = transform(sample, np.random.default_rng(123))
    changed = transform(sample, np.random.default_rng(124))

    assert torch.equal(first.image, replay.image)
    assert not torch.equal(first.image, changed.image)
    assert first.image.dtype == sample.image.dtype
    assert 0.0 <= float(first.image.min()) <= float(first.image.max()) <= 1.0
    assert first.keypoints is sample.keypoints
    assert first.visibility is sample.visibility
    assert first.centers is sample.centers
    assert first.instance_ids is sample.instance_ids


def test_compose_keeps_configuration_order() -> None:
    transform = _combined()

    assert [type(item).__name__ for item in transform.augmentations] == [
        "RandomLineMorphology",
        "RandomHeatmapBlur",
        "RandomLineDropout",
        "RandomGhostLines",
        "RandomProbabilityNoise",
    ]


def test_datamodule_augments_train_only_and_explicit_identity_wins() -> None:
    datamodule = GroundCourtDataModule(
        image_size=64,
        train_samples=1,
        val_samples=1,
        test_samples=1,
        num_workers=0,
        min_courts=1,
        max_courts=1,
        min_scale_px_per_metre=1.0,
        max_scale_px_per_metre=1.0,
        seed=23,
        augmentations=[
            {
                "name": "random_heatmap_blur",
                "params": {"probability": 1.0, "sigma_range": [1.0, 1.0]},
            }
        ],
    )
    datamodule.setup()
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    identity = build_augmentation("identity")
    datasets: dict[GroundCourtSplit, GroundCourtDataset] = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }
    clean_samples: dict[str, dict[str, object]] = {}
    for split, dataset in datasets.items():
        clean_dataset = GroundCourtDataset(
            datamodule.dataset_config,
            split=split,
            augmentation=identity,
        )
        actual = dataset[0]
        clean = clean_dataset[0]
        clean_samples[split] = clean
        for key in ("keypoints", "visibility", "centers", "instance_ids"):
            assert torch.equal(
                cast(torch.Tensor, actual[key]), cast(torch.Tensor, clean[key])
            )
        if split == "train":
            assert not torch.equal(
                cast(torch.Tensor, actual["image"]),
                cast(torch.Tensor, clean["image"]),
            )
        else:
            assert torch.equal(
                cast(torch.Tensor, actual["image"]),
                cast(torch.Tensor, clean["image"]),
            )
        assert actual["sample_id"] == f"{split}-00000000"

    assert not torch.equal(
        cast(torch.Tensor, clean_samples["train"]["image"]),
        cast(torch.Tensor, clean_samples["val"]["image"]),
    )
    assert not torch.equal(
        cast(torch.Tensor, clean_samples["val"]["image"]),
        cast(torch.Tensor, clean_samples["test"]["image"]),
    )
