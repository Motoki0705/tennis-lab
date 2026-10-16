"""Read-only dataset/model service with reproducible 3D predictions with shared multiview corruption."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.data.preprocessing import corrupt_trajectory
from src.tasks.ball_refiner_3d.data.schema import CorruptedTrajectory, Rally
from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor
from src.tasks.ball_refiner_3d.physics.units import decode_field, physical_field
from src.tasks.ball_refiner_3d.visualization.dataset_review.artifacts import (
    read_receipt,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.checkpoints import (
    Checkpoint,
    CheckpointCatalog,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.contracts import (
    SCHEMA,
    ReviewRequest,
    evaluation_profile,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.payload import (
    ModelOutputs,
    PhysicsOutputs,
    scene_payload,
)
from src.utils.paths import PROJECT_ROOT


@lru_cache(maxsize=2)
def _saved_arrays(path: str, digest: str) -> dict[str, np.ndarray]:
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("保存済み予測の内容hashが一致しません")
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class ReviewService:
    def __init__(
        self, data_root: Path, outputs_root: Path, checkpoints_root: Path
    ) -> None:
        self.data_root, self.outputs_root, self.checkpoints_root = (
            data_root.resolve(),
            outputs_root.resolve(),
            checkpoints_root.resolve(),
        )
        self.dataset = SharedDataset(self.data_root)
        self.rallies = {rally.name: rally for rally in self.dataset.rallies}
        self.checkpoints = CheckpointCatalog(
            self.outputs_root,
            self.checkpoints_root,
            self.dataset.manifest_hash,
            self.dataset.fps,
        )
        self.checkpoints.refresh()
        self.inference_lock = threading.Lock()
        with initialize_config_dir(
            version_base=None,
            config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs"),
        ):
            config = compose(config_name="train")
        self.event_sigma_frames = float(config.loss.event.sigma_frames)
        self.defaults = evaluation_profile(
            cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
        )

    def catalog(self, *, refresh: bool = False) -> dict[str, Any]:
        entries = (
            self.checkpoints.refresh()
            if refresh
            else [entry.info.copy() for entry in self.checkpoints.entries.values()]
        )
        defaults: dict[str, str | None] = {}
        for dimensions in (3,):
            candidates = [
                item
                for item in entries
                if item["compatible"] and item["dimensions"] == dimensions
            ]
            candidates.sort(
                key=lambda item: (
                    item["evaluation_profile"] != self.defaults,
                    not item["recommended"],
                    item["filename"] != "best.ckpt",
                    item["validation_rmse"],
                    item["id"],
                )
            )
            defaults[str(dimensions)] = candidates[0]["id"] if candidates else None
        return {
            "schema": SCHEMA,
            "manifest_sha256": self.dataset.manifest_hash,
            "dataset": str(self.data_root),
            "fps": self.dataset.fps,
            "views": self.dataset.manifest["views"],
            "checkpoints": entries,
            "default_checkpoints": defaults,
            "default_profile": self.defaults,
            "rallies": [
                {
                    "id": rally.name,
                    "split": rally.split,
                    "frames": len(rally.xyz),
                    "events": int(np.count_nonzero(rally.events)),
                }
                for rally in self.dataset.rallies
            ],
        }

    def validate(self, request: ReviewRequest) -> tuple[Rally, dict[int, Checkpoint]]:
        request.corruption()
        if (
            request.manifest_sha256 != self.dataset.manifest_hash
            or hashlib.sha256(
                (self.data_root / "manifest.json").read_bytes()
            ).hexdigest()
            != self.dataset.manifest_hash
        ):
            raise RuntimeError(
                "データセットのmanifestが変更されています。サーバーを再起動してください"
            )
        if request.rally not in self.rallies:
            raise ValueError("カタログにないラリーです")
        rally = self.rallies[request.rally]
        record = self.dataset.manifest["records"][rally.index]
        if (
            hashlib.sha256((self.data_root / record["path"]).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise RuntimeError("ラリーデータがmanifestと一致しません")
        selected = {}
        for dimension, identifier in ((3, request.checkpoint_3d),):
            if identifier:
                selected[dimension] = self.checkpoints.get(
                    identifier, dimension, request.checkpoint_hashes.get(str(dimension))
                )
        return rally, selected

    def _prepare(self, request: ReviewRequest, rally: Rally) -> CorruptedTrajectory:
        seed = int(
            np.random.SeedSequence(
                [request.augmentation_seed, rally.index]
            ).generate_state(1)[0]
        )
        return corrupt_trajectory(
            rally.uv,
            rally.visible,
            rally.projection,
            rally.events,
            config=request.corruption(),
            seed=seed,
            noise_enabled=request.noise_enabled,
        )

    def _saved(
        self,
        checkpoint: Checkpoint,
        request: ReviewRequest,
        rally: Rally,
        corruption: CorruptedTrajectory,
    ) -> ModelOutputs:
        if (
            rally.split != "test"
            or not checkpoint.info["saved_available"]
            or checkpoint.run is None
            or checkpoint.predictions is None
        ):
            raise ValueError(
                "このラリー・checkpointには保存済みtest予測がありません。再推論を実行してください"
            )
        if request.profile() != checkpoint.info["evaluation_profile"]:
            raise ValueError(
                "保存済み予測と拡張条件・seedが異なります。評価条件に戻すか再推論してください"
            )
        contract = json.loads((checkpoint.run / "data_contract.json").read_text())
        if (
            contract["manifest_sha256"] != self.dataset.manifest_hash
            or rally.name not in contract["test_ids"]
        ):
            raise ValueError("保存済み予測のデータ契約が一致しません")
        receipt = read_receipt(
            checkpoint.predictions,
            checkpoint.info["sha256"],
            self.dataset.manifest_hash,
            request.profile(),
        )
        path = checkpoint.predictions / "pred_test.npz"
        arrays = _saved_arrays(str(path), receipt["predictions_sha256"])
        dim = checkpoint.info["dimensions"]
        views = 1
        frames = len(rally.xyz)
        take = arrays["rally_id"] == rally.index
        shape = (views, frames, dim)
        expected_mask = corruption.missing_3d[None]
        expected_input = corruption.xyz_m[None]
        expected_target = rally.xyz[None]
        if (
            int(take.sum()) != views * frames
            or not np.array_equal(
                arrays["view_id"][take], np.repeat(np.arange(views), frames)
            )
            or not np.array_equal(
                arrays["frame_id"][take], np.tile(np.arange(frames), views)
            )
        ):
            raise ValueError("保存予測のラリー・camera・frame対応が不正です")
        if not np.array_equal(
            arrays["missing"][take].reshape(views, frames), expected_mask
        ):
            raise ValueError("保存済み予測の欠損maskが現在の入力と一致しません")
        for key, expected in (("input", expected_input), ("target", expected_target)):
            if not np.allclose(
                arrays[key][take].reshape(shape), expected, rtol=1e-5, atol=2e-4
            ):
                raise ValueError(f"保存済み予測の{key}が現在のデータと一致しません")
        prediction = arrays["prediction"][take].reshape(shape)
        if not np.isfinite(prediction).all():
            raise ValueError("保存済み予測に非有限値があります")
        probability = arrays["event_probability"][take]
        target = gaussian_event_target(
            rally.events, checkpoint.info["event_sigma_frames"]
        )
        if (
            not np.allclose(arrays["event_target"][take], target, atol=1e-6)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
        ):
            raise ValueError("保存済みイベント確率・教師が不正です")
        if ("integrated" in arrays) != checkpoint.info["physics_heads"]:
            raise ValueError(
                "保存済み予測の物理head出力の有無がcheckpointと一致しません"
            )
        physics = None
        if checkpoint.info["physics_heads"]:
            row = np.flatnonzero(arrays["physics_rally_id"] == rally.index)
            if len(row) != 1:
                raise ValueError("保存済みの物理パラメータとラリーの対応が不正です")
            physics = PhysicsOutputs(
                integrated=arrays["integrated"][take],
                segment=arrays["integrated_segment"][take],
                integrated_truth_segments=arrays["integrated_truth_segments"][take],
                field=arrays["physics_field"][row[0]],
                surface_probability=arrays["physics_surface_probability"][row[0]],
            )
        return ModelOutputs(
            cast(np.ndarray, prediction[0]), cast(np.ndarray, probability), physics
        )

    def _result(
        self,
        request: ReviewRequest,
        rally: Rally,
        corruption: CorruptedTrajectory,
        selected: dict[int, Checkpoint],
        predictions: dict[int, ModelOutputs],
        source: str,
        seconds: float,
    ) -> dict[str, Any]:
        self.validate(request)
        path = self.data_root / self.dataset.manifest["records"][rally.index]["path"]
        with np.load(path, allow_pickle=False) as raw:
            cameras = {
                key: raw[key] for key in ("camera_centers", "rotation", "intrinsic")
            }
        scene = scene_payload(
            rally,
            corruption,
            cameras,
            fps=self.dataset.fps,
            model=predictions.get(3),
            event_sigma_frames=selected[3].info["event_sigma_frames"]
            if selected
            else self.event_sigma_frames,
        )
        receipt = request.model_dump()
        receipt["checkpoint_hashes"] = {
            str(dim): entry.info["sha256"] for dim, entry in selected.items()
        }
        return {
            "schema": SCHEMA,
            "scene": scene,
            "request": receipt,
            "models": {str(dim): entry.info for dim, entry in selected.items()},
            "source": source,
            "seconds": seconds,
            "input_sha256": hashlib.sha256(
                corruption.uv_px.tobytes()
                + corruption.missing_2d.tobytes()
                + corruption.xyz_m.tobytes()
                + corruption.missing_3d.tobytes()
            ).hexdigest(),
        }

    def preview(self, request: ReviewRequest) -> dict[str, Any]:
        rally, selected = self.validate(request)
        return self._result(
            request, rally, self._prepare(request, rally), selected, {}, "preview", 0.0
        )

    def saved(self, request: ReviewRequest) -> dict[str, Any]:
        rally, selected = self.validate(request)
        if not selected:
            raise ValueError("比較するモデルを選択してください")
        corruption = self._prepare(request, rally)
        predictions = {
            dim: self._saved(entry, request, rally, corruption)
            for dim, entry in selected.items()
        }
        return self._result(
            request, rally, corruption, selected, predictions, "saved", 0.0
        )

    def infer(self, request: ReviewRequest) -> dict[str, Any]:
        rally, selected = self.validate(request)
        if not selected:
            raise ValueError("比較するモデルを選択してください")
        if request.device == "cuda":
            import os

            if not os.environ.get("TENNIS_RUN_ID"):
                raise RuntimeError("CUDA推論は共有training queueから実行してください")
        start = time.monotonic()
        corruption = self._prepare(request, rally)
        predictions = {}
        with self.inference_lock:
            for dim, checkpoint in selected.items():
                device = torch.device(request.device)
                predictor = RefinerPredictor.from_checkpoint(
                    checkpoint.path, device=device
                )
                coordinates = corruption.xyz_m[None]
                missing = corruption.missing_3d[None]
                output = predictor.predict(
                    torch.from_numpy(coordinates),
                    torch.from_numpy(missing),
                    fps=self.dataset.fps,
                    seed=request.flow_seed + rally.index,
                )
                physics = None
                if output.physics is not None:
                    upper = predictor.predict(
                        torch.from_numpy(coordinates),
                        torch.from_numpy(missing),
                        fps=self.dataset.fps,
                        seed=request.flow_seed + rally.index,
                        segment=torch.from_numpy(rally.physics.frame_segment()[None]),
                    ).physics
                    if upper is None:
                        raise RuntimeError("物理headが区間指定の推論で出力を返しません")
                    physics = PhysicsOutputs(
                        integrated=output.physics.integrated.numpy()[0],
                        segment=output.physics.segment.numpy()[0],
                        integrated_truth_segments=upper.integrated.numpy()[0],
                        field=physical_field(decode_field(upper.field))[0].numpy(),
                        surface_probability=upper.surface_probability.numpy()[0],
                    )
                predictions[dim] = ModelOutputs(
                    output.coordinates.numpy()[0],
                    output.event_probability.numpy()[0],
                    physics,
                )
                del predictor
        return self._result(
            request,
            rally,
            corruption,
            selected,
            predictions,
            "live",
            time.monotonic() - start,
        )
