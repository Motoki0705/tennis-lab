"""Frozen-base metadata-free v1 checkpoint and dataset parity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.base.data import MissingCourtCoordinateMetadataError
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_multiview_trajectories,
)
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    compose_blcs_trajectory_model_io,
)
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io import (
    PLCSDecodedPrediction,
    load_plcs_checkpoint_mapping,
    prepare_plcs_checkpoint_config,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)

pytestmark = pytest.mark.integration

_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "issue_786"
    / "legacy_v1_representative"
)
_BASE_REVISION = "59e3b166c2d010d5e62be52c2be76d98a94af0e0"
_ATOL = 1.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    value: Any = json.loads((_FIXTURE_ROOT / "manifest.json").read_text())
    if not isinstance(value, dict):
        raise TypeError("Representative fixture manifest root must be a mapping.")
    return cast("dict[str, Any]", value)


def _resolver(tmp_path: Path) -> PathResolver:
    return PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT.resolve(),
            data_root=(_FIXTURE_ROOT / "datasets").resolve(),
            checkpoint_root=_FIXTURE_ROOT.resolve(),
            artifact_root=tmp_path.resolve(),
            output_root=tmp_path.resolve(),
            cache_root=(tmp_path / "cache").resolve(),
            external_asset_root=PROJECT_ROOT.resolve(),
        )
    )


def _tensor_arrays(
    batch: Mapping[str, object],
    names: list[str],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        value = batch.get(name)
        if isinstance(value, Tensor):
            array: np.ndarray = value.detach().cpu().numpy()
            arrays[f"input_{name}"] = array
    return arrays


def _scalar_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    array: np.ndarray = np.asarray(value, dtype=dtype)
    return array


def _assert_matches_golden(
    golden_path: Path,
    observed: Mapping[str, np.ndarray],
) -> None:
    with np.load(golden_path) as golden:
        assert set(observed) == set(golden.files)
        for name in golden.files:
            expected = golden[name]
            actual = observed[name]
            assert actual.shape == expected.shape, name
            if expected.dtype == np.bool_:
                np.testing.assert_array_equal(actual, expected, err_msg=name)
            else:
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=_ATOL,
                    rtol=0.0,
                    err_msg=name,
                )


def _run_blcs_parity(tmp_path: Path) -> None:
    checkpoint = _FIXTURE_ROOT / "blcs_representative_legacy_v1.ckpt"
    dataset_root = _FIXTURE_ROOT / "datasets" / "blcs_legacy_v1"
    v1 = resolve_court_coordinate_normalization("v1")
    runtime = load_checkpoint_runtime(
        checkpoint,
        runtime_normalization=v1,
    )
    dataset = BallTrajectoryDataset(
        scene_dir=dataset_root,
        split_file=dataset_root / "test.txt",
        config=runtime.config,
        augment=False,
    )
    dataset.rng = np.random.default_rng(786)
    batch = cast(
        "dict[str, object]",
        dict(collate_multiview_trajectories([dataset[0]])),
    )

    binding = compose_blcs_trajectory_model_io(runtime.config)
    module = BLCSLightningModule.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        model_io=binding,
        config=runtime.config,
        strict=True,
        weights_only=False,
    ).cpu().eval()
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint,
        resolver=_resolver(tmp_path),
        device="cpu",
        court_coordinate_normalization=v1,
    )

    module.test_metrics.reset()
    with torch.no_grad():
        result = module._compute_supervised_result(batch, "test")
        normalized = predictor.predict_batch(batch, denormalize=False)
        physical = predictor.predict_batch(batch, denormalize=True)
    aggregate = module.test_metrics.compute()
    output = cast("BLCSTrajectoryPrediction", result["outputs"])
    torch.testing.assert_close(output.position.cpu(), normalized.position)
    assert normalized.velocity is not None
    assert physical.velocity is not None

    position_target = cast("Tensor", batch["position_3d"])
    velocity_target = cast("Tensor", batch["velocity_3d"])
    losses = cast("Mapping[str, Tensor]", result["losses"])
    batch_metrics = cast("Mapping[str, object]", result["metrics"])
    observed = {
        **_tensor_arrays(
            batch,
            ["ball_uv", "ball_vis", "court_kp", "court_vis", "padding_mask"],
        ),
        "target_position_normalized": position_target.numpy(),
        "target_position_meters": cast("Tensor", v1.denormalize_position(position_target)).numpy(),
        "target_velocity_world_mps": cast(
            "Tensor", v1.denormalize_velocity(velocity_target)
        ).numpy(),
        "prediction_position_normalized": normalized.position.numpy(),
        "prediction_position_meters": physical.position.numpy(),
        "prediction_velocity_normalized": normalized.velocity.numpy(),
        "prediction_velocity_meters_per_second": physical.velocity.numpy(),
        **{
            f"loss_{name}": _scalar_array(value, dtype=np.dtype(np.float32))
            for name, value in losses.items()
        },
        **{
            f"metric_batch_{name}": _scalar_array(value, dtype=np.dtype(np.float64))
            for name, value in batch_metrics.items()
        },
        **{
            f"metric_aggregate_{name}": _scalar_array(
                value,
                dtype=np.dtype(np.float64),
            )
            for name, value in aggregate.items()
        },
    }
    _assert_matches_golden(
        _FIXTURE_ROOT / "blcs_representative_legacy_v1_golden.npz",
        observed,
    )


def _run_plcs_parity(tmp_path: Path) -> None:
    checkpoint = _FIXTURE_ROOT / "plcs_representative_legacy_v1.ckpt"
    dataset_root = _FIXTURE_ROOT / "datasets" / "plcs_legacy_v1"
    v1 = resolve_court_coordinate_normalization("v1")
    checkpoint_mapping = load_plcs_checkpoint_mapping(checkpoint)
    config, contract = prepare_plcs_checkpoint_config(
        checkpoint_mapping,
        v1,
        location=str(checkpoint),
    )
    assert contract == v1
    dataset = SceneDataset(
        scene_dir=dataset_root,
        split_file=dataset_root / "test.txt",
        config=config,
        augment=False,
    )
    dataset.rng = np.random.default_rng(786)
    batch = cast("dict[str, Tensor]", dict(collate_plcs_batch([dataset[0]])))

    module = PLCSLightningModule.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        config=config,
        strict=True,
        weights_only=False,
    ).cpu().eval()
    predictor = PLCSPredictor.load_from_checkpoint(
        checkpoint,
        resolver=_resolver(tmp_path),
        device="cpu",
        court_coordinate_normalization=v1,
    )

    module.test_metrics.reset()
    with torch.no_grad():
        result = module._compute_supervised_result(batch, "test")
        normalized = predictor.predict(
            batch["human_kp"],
            batch["court_kp"],
            batch["human_vis"],
            batch["padding_mask"],
            batch["court_vis"],
            denormalize=False,
        )
        physical = predictor.predict(
            batch["human_kp"],
            batch["court_kp"],
            batch["human_vis"],
            batch["padding_mask"],
            batch["court_vis"],
            denormalize=True,
        )
    aggregate = module.test_metrics.compute()
    output = cast("PLCSDecodedPrediction", result["outputs"])
    torch.testing.assert_close(output.position.cpu(), normalized["position"])

    metrics = cast("Mapping[str, object]", result["metrics"])
    losses = {
        name.removeprefix("loss_"): value
        for name, value in metrics.items()
        if name.startswith("loss_")
    }
    observed = {
        **_tensor_arrays(
            batch,
            ["human_kp", "human_vis", "court_kp", "court_vis", "padding_mask"],
        ),
        "target_position_normalized": batch["position"].numpy(),
        "target_position_meters": cast(
            "Tensor", v1.denormalize_position(batch["position"])
        ).numpy(),
        "target_rotation_cos_sin": batch["rotation"].numpy(),
        "target_human_kp_3d_meters": batch["human_kp_3d"].numpy(),
        "prediction_position_normalized": normalized["position"].numpy(),
        "prediction_position_meters": physical["position_meters"].numpy(),
        "prediction_rotation_cos_sin": normalized["rotation"].numpy(),
        "prediction_yaw_radians": physical["yaw_radians"].numpy(),
        **{
            f"loss_{name}": _scalar_array(value, dtype=np.dtype(np.float32))
            for name, value in losses.items()
        },
        **{
            f"metric_batch_{name}": _scalar_array(value, dtype=np.dtype(np.float64))
            for name, value in metrics.items()
        },
        **{
            f"metric_aggregate_{name}": _scalar_array(
                value,
                dtype=np.dtype(np.float64),
            )
            for name, value in aggregate.items()
        },
    }
    _assert_matches_golden(
        _FIXTURE_ROOT / "plcs_representative_legacy_v1_golden.npz",
        observed,
    )


def test_representative_fixture_has_frozen_base_provenance_and_exact_bytes() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["base"]["sha"] == _BASE_REVISION
    assert manifest["base"]["tracked_clean_before"] is True
    assert manifest["base"]["tracked_clean_after"] is True
    assert manifest["runtime"] == {
        "court_coordinate_scale_xyz_meters": [
            5.485000133514404,
            11.885000228881836,
            1.0700000524520874,
        ],
        "device": "cpu",
        "dtype": "float32",
        "numpy_seed": 786,
        "python_seed": 786,
        "torch_deterministic_algorithms": True,
        "torch_num_threads": 1,
        "torch_seed": 786,
    }

    artifacts = cast("Mapping[str, Mapping[str, Any]]", manifest["artifacts"])
    for key, filename in {
        "generator": "generate_representative.py.txt",
        "blcs_checkpoint": "blcs_representative_legacy_v1.ckpt",
        "plcs_checkpoint": "plcs_representative_legacy_v1.ckpt",
        "blcs_golden": "blcs_representative_legacy_v1_golden.npz",
        "plcs_golden": "plcs_representative_legacy_v1_golden.npz",
    }.items():
        path = _FIXTURE_ROOT / filename
        assert _sha256(path) == artifacts[key]["sha256"]
        if "byte_count" in artifacts[key]:
            assert path.stat().st_size == artifacts[key]["byte_count"]

    for key, relative in {
        "blcs_dataset_fixture": "datasets/blcs_legacy_v1",
        "plcs_dataset_fixture": "datasets/plcs_legacy_v1",
    }.items():
        assert _tree_sha256(_FIXTURE_ROOT / relative) == artifacts[key]["sha256"]

    for task in ("blcs", "plcs"):
        checkpoint = torch.load(
            _FIXTURE_ROOT / f"{task}_representative_legacy_v1.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        assert "court_coordinate_normalization" not in checkpoint
        saved_config = checkpoint["hyper_parameters"]["config"]
        assert "court_coordinate_normalization" not in saved_config
        dataset_root = _FIXTURE_ROOT / "datasets" / f"{task}_legacy_v1"
        assert not (dataset_root / "meta.json").exists()
        scene_meta = next((dataset_root / "scenes").glob("*/meta.json"))
        document = json.loads(scene_meta.read_text())
        assert "court_coordinate_normalization" not in document


def test_frozen_base_v1_checkpoints_replay_inference_loss_and_metrics(
    tmp_path: Path,
) -> None:
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        _run_blcs_parity(tmp_path)
        _run_plcs_parity(tmp_path)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_metadata_free_checkpoint_rejects_v2_before_lightning_state_restore(
    tmp_path: Path,
    task: str,
) -> None:
    checkpoint = _FIXTURE_ROOT / f"{task}_representative_legacy_v1.ckpt"
    v2 = resolve_court_coordinate_normalization("v2")
    if task == "blcs":
        with (
            patch.object(
                BLCSLightningModule,
                "load_from_checkpoint",
            ) as state_load,
            pytest.raises(
                MissingCourtCoordinateMetadataError,
                match="legacy v1 only",
            ),
        ):
            BLCSPredictor.load_from_checkpoint(
                checkpoint,
                resolver=_resolver(tmp_path),
                device="cpu",
                court_coordinate_normalization=v2,
            )
    else:
        with (
            patch.object(
                PLCSLightningModule,
                "load_from_checkpoint",
            ) as state_load,
            pytest.raises(
                MissingCourtCoordinateMetadataError,
                match="legacy v1 only",
            ),
        ):
            PLCSPredictor.load_from_checkpoint(
                checkpoint,
                resolver=_resolver(tmp_path),
                device="cpu",
                court_coordinate_normalization=v2,
            )
    state_load.assert_not_called()


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_metadata_free_dataset_rejects_v2_before_array_payload_load(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    v1 = resolve_court_coordinate_normalization("v1")
    v2_config: DictConfig
    if task == "blcs":
        checkpoint = _FIXTURE_ROOT / "blcs_representative_legacy_v1.ckpt"
        runtime = load_checkpoint_runtime(
            checkpoint,
            runtime_normalization=v1,
        )
        v2_config = cast(
            "DictConfig",
            OmegaConf.merge(
                runtime.config,
                {"court_coordinate_normalization": {"version": "v2"}},
            ),
        )
    else:
        checkpoint = _FIXTURE_ROOT / "plcs_representative_legacy_v1.ckpt"
        mapping = load_plcs_checkpoint_mapping(checkpoint)
        v1_config, _ = prepare_plcs_checkpoint_config(mapping, v1)
        v2_config = cast(
            "DictConfig",
            OmegaConf.merge(
                v1_config,
                {"court_coordinate_normalization": {"version": "v2"}},
            ),
        )

    def _forbid_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Array payload was read before metadata validation.")

    monkeypatch.setattr(np, "load", _forbid_payload)
    dataset_root = _FIXTURE_ROOT / "datasets" / f"{task}_legacy_v1"
    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        if task == "blcs":
            BallTrajectoryDataset(
                scene_dir=dataset_root,
                split_file=dataset_root / "test.txt",
                config=v2_config,
                augment=False,
            )
        else:
            SceneDataset(
                scene_dir=dataset_root,
                split_file=dataset_root / "test.txt",
                config=v2_config,
                augment=False,
            )


def _assert_explicit_v1_contract(
    contract: CourtCoordinateNormalization,
) -> None:
    assert contract.version == "v1"
    assert contract.scale_xyz == (5.485, 11.885, 1.07)


def test_metadata_free_loads_are_explicitly_bound_to_v1() -> None:
    blcs = load_checkpoint_runtime(
        _FIXTURE_ROOT / "blcs_representative_legacy_v1.ckpt",
        runtime_normalization="v1",
    )
    _assert_explicit_v1_contract(blcs.normalization)
    mapping = load_plcs_checkpoint_mapping(
        _FIXTURE_ROOT / "plcs_representative_legacy_v1.ckpt"
    )
    _, plcs_contract = prepare_plcs_checkpoint_config(
        mapping,
        resolve_court_coordinate_normalization("v1"),
    )
    _assert_explicit_v1_contract(plcs_contract)
