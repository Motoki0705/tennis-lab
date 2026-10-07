"""CPU store -> model -> masked loss/evaluation, including old checkpoint config."""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.configuration import BallRuntimePaths
from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.evaluation.configuration import build_evaluation_config
from src.tasks.ball_detection.evaluation.contracts import (
    DatasetSpec,
    EvaluationManifest,
    MetricsSpec,
    PerformanceSpec,
)
from src.tasks.ball_detection.evaluation.evaluator import (
    DefaultJobEvaluator,
    evaluate_dataloader,
)
from src.tasks.ball_detection.evaluation.runner import EvaluationPipeline
from src.tasks.ball_detection.model_io.evaluation import CheckpointBallHeatmapPredictor
from src.tasks.ball_detection.scripts.eval import _forward_batch
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from tests.support.tasks.ball_detection.store import (
    ball,
    frame,
    store_config,
    write_store_clip,
)


def test_cpu_gradient_and_manifest_evaluation_use_store_mask(tmp_path: Path) -> None:
    cfg = store_config(tmp_path)
    cfg.paths.project_root = str(Path(__file__).resolve().parents[4])
    cfg.data.sources = ["tracknet"]
    cfg.data.train_sampling = None
    cfg.data.image_size = [64, 64]
    cfg.model.dims = [4, 8, 16, 32]
    cfg.model.depth = 1
    cfg.model.input_mode = "mdd"
    cfg.model.in_channels = 2
    write_store_clip(
        tmp_path / "ball_detection/test-v1",
        "tracknet/val/clip",
        [frame(0, ball()), frame(1, ball("unresolved", None))],
        split="val",
    )
    module = BallStoreDataModule(cfg)
    module.setup("validate")
    batch = next(iter(module.val_dataloader()))
    detector = BallDetectionLightningModule(cfg)
    detector.eval()
    result = detector._compute_supervised_result(batch, "val")
    assert torch.isfinite(result["loss"]) and result["loss"] > 0
    result["loss"].backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in detector.parameters()
    )

    spec = MetricsSpec(0.5, 4.0, 3, 1, False)
    resolver = BallRuntimePaths.from_config(cfg).resolver
    dataset = DatasetSpec(
        "fixture",
        "rgb_sequence",
        ("val",),
        {
            "data_dir": "ball_detection/test-v1",
            "sources": ["tracknet"],
            "train_sampling": None,
            "image_size": [64, 64],
            "heatmap_size": [24, 32],
            "num_workers": 0,
            "augmentation": OmegaConf.to_container(cfg.data.augmentation, resolve=True),
        },
    )
    # Older checkpoints keep their own model/normalization; only their data section is replaced.
    saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved.data.source = "tracknet"
    del saved["paths"]
    del saved["training"]
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "hyper_parameters": {"config": OmegaConf.to_container(saved, resolve=True)},
            "state_dict": {
                f"model.{key}": value
                for key, value in detector.model.state_dict().items()
            },
        },
        checkpoint,
    )
    restored = CheckpointBallHeatmapPredictor.load(
        checkpoint, device=torch.device("cpu"), strict=True, weights_only=False
    )
    loss, maps = _forward_batch(restored, batch, detector.loss_fn)
    torch.testing.assert_close(loss, result["loss"])
    torch.testing.assert_close(maps, result["pred_heatmaps"])

    evaluation_cfg = build_evaluation_config(
        checkpoint_config=saved,
        dataset_spec=dataset,
        metrics_spec=spec,
        resolver=resolver,
    )
    evaluation_data = BallStoreDataModule(evaluation_cfg)
    evaluation_data.setup("validate")
    manifest = EvaluationManifest(
        "ball_detection_evaluation_manifest_v1",
        tmp_path / "output",
        "cpu",
        False,
        True,
        spec,
        PerformanceSpec(0, None),
        {"fixture": dataset},
        (),
        resolver,
    )

    class FixedPredictor:
        device = torch.device("cpu")

        def predict_heatmaps(
            self, images: torch.Tensor, *, target_size_hw: tuple[int, int]
        ) -> torch.Tensor:
            # Confident false positives on both frames: only the known frame may affect metrics.
            maps = torch.zeros(*images.shape[:2], *target_size_hw)
            maps[:, :, 0, 0] = 1
            return maps

    payload = evaluate_dataloader(
        adapter=FixedPredictor(),
        dataloader=evaluation_data.val_dataloader(),
        data_config=evaluation_cfg.data,
        split="val",
        manifest=manifest,
    )
    assert payload["metrics"]["aggregate"]["frames"] == 1
    assert payload["metrics"]["aggregate"]["unsupervised_excluded_frames"] == 1
    assert payload["metrics"]["by_source"]["tracknet"]["frames"] == 1
    assert payload["dataset_provenance"]["schema"] == "ball_detection_frames.v1"
    assert len(payload["dataset_provenance"]["index_sha256"]) == 64
    # Check resumable evaluation can fingerprint the new data config without checkpoint context.
    fingerprint = EvaluationPipeline(
        manifest, evaluator=DefaultJobEvaluator(device=torch.device("cpu"))
    )._dataset_fingerprint(dataset, "val")
    assert len(fingerprint["split_artifacts"]) == 2
