"""Small CPU full-model checkpoints and independent train/validation stores."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.posttraining import SCHEMA
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.manifest import (
    build_coordinate_manifest,
    clip_semantic_digest,
)
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector, MDDPretrainConfig
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def full_model_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True)
    config = MDDPretrainConfig(
        stem_channels=(4, 8, 8, 8), mixed_channels=(8, 16),
        residual_blocks=(0, 1, 1, 1, 1, 1), decoder_channels=8,
        dim=16, heads=2, layers=1, ffn_dim=64, dropout=0., rope_base=10000.,
        activation_checkpointing=False, encoder_variant="convnext_v2", temporal_mixing="factorized",
    )
    torch.manual_seed(81)
    model = DeepMDDQueryDetector(config, mdd_a=.3, mdd_b=.2)
    input_contract = model.mdd.input_contract()
    image_decode = jpeg_decoder_contract("opencv")
    recipe = dict(model=asdict(config), input_contract=input_contract, image_decode=image_decode,
                  parent=dict(input_contract=input_contract, image_decode=image_decode), manifest_sha256="0" * 64)
    # Deliberately unusable optimizer and old counters: none may initialize a new run.
    payload: dict[str, Any] = dict(schema=SCHEMA, stage="query_posttraining", model_config=asdict(config),
        state_dict=model.state_dict(), recipe=recipe, epoch=9, global_step=60000,
        optimizer=dict(not_a_new_optimizer=True), best_epoch=9, best_score=-1.)
    path = directory / "epoch-009.pt"
    torch.save(payload, path)
    return path


def training_manifests(directory: Path, *, overlap_clip_ids: bool = False) -> tuple[Path, Path]:
    result = []
    for split in ("train", "val"):
        clip_id = "train/clip" if overlap_clip_ids else f"{split}/clip"
        root = write_store_clip(directory / split, clip_id,
            [frame(i, ball(xy=(10. + i / 4, 20.))) for i in range(32)], split=split, size=(96, 64))
        if overlap_clip_ids and split == "val":
            # Keep source groups distinct so only clip membership overlaps.
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["clips"][0]["group_id"] = "val"
            metadata_path.write_text(json.dumps(metadata))
        store = BallFrameStore(root)
        reference = dict(clip_ids=[clip_id], clip_semantic_sha256={clip_id: clip_semantic_digest(store, store.clips[0])})
        manifest = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig((1,)),
            input_kind="mdd_only", common_reference=reference if split == "val" else None)
        path = directory / f"{split}.json"
        path.write_text(json.dumps(manifest))
        result.append(path)
    return result[0], result[1]


def full_model_arguments(parent: Path, train: Path, validation: Path, output: Path) -> list[str]:
    from src.utils.checksum import dual_sha256

    augmentation = Path(__file__).resolve().parents[4] / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"
    return ["--manifest", str(train), "--validation-manifest", str(validation),
        "--initial-checkpoint", str(parent), "--initial-checkpoint-sha256", dual_sha256(parent),
        "--augmentation-config", str(augmentation), "--output", str(output),
        "--epochs", "2", "--freeze-epochs", "1", "--windows-per-epoch", "1",
        "--learning-rate", ".001", "--warmup-updates", "0", "--device", "cpu",
        "--precision", "fp32", "--jpeg-decoder", "opencv", "--compile-mode", "off",
        "--input-verification", "lazy", "--seed", "42",
        "--num-workers", "0", "--cpu-threads", "2", "--preview-clips", "0", "--log-every", "1"]
