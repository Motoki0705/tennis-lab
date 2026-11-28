"""Run one matched training condition, then score both frozen real splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from protocol import (
    BATCH_SIZE,
    CONDITIONS,
    LONG_SIDE,
    REAL_REGISTRY_SHA256,
    REAL_SPLIT_COUNTS,
    TRAIN_SEEDS,
    UPDATES,
    score_prediction,
    stratum_counts,
)

from src.synthetic_data_generation.dataset.court.scale.publication import read_json
from src.tasks.court_detection.configuration import TennisCourtEvaluationSourceConfig
from src.tasks.court_detection.data.contracts import CourtSampleRecord
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.court_detection.model_io.contracts import CourtKeypointPrediction
from src.tasks.court_detection.scaling.real_split import (
    TennisCourtEvaluationInput,
    canonical_sha256,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.data.image_record_store import file_sha256
from src.utils.io import write_json_atomic
from src.utils.schema.court import GROUND_COURT_KP_NAMES

DINO_CHECKPOINT = "dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINO_SHA256 = "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d"
DINO_COMMIT = "4add255125e6cbf446c6a43c5f3028aa41b63648"
WARMUP_UPDATES = 900


def run_name(condition: str, seed: int) -> str:
    return f"{condition}-s{seed}"


def training_command(
    campaign: Path,
    checkpoint_root: Path,
    external_asset_root: Path,
    condition: str,
    seed: int,
) -> list[str]:
    overrides = [
        "data/source=synthetic_court_dataset_subset",
        f"data.source.manifest=pool/{condition}.train.subset.json",
        "data.source.validation_manifest=pool/common.val.subset.json",
        "mixed.sources.tennis_court_detector.root=tcd",
        "mixed.sources.tennis_court_detector.registry=registry.json",
        f"paths.data_root={campaign / 'inputs'}",
        f"paths.output_root={campaign}",
        f"paths.artifact_root={campaign}",
        f"paths.checkpoint_root={checkpoint_root}",
        f"paths.external_asset_root={external_asset_root}",
        f"data.batch_size={BATCH_SIZE}",
        f"data.augmentation.train_scales=[{LONG_SIDE}]",
        f"data.augmentation.val_short_side={LONG_SIDE}",
        "model.encoder.backbone_name=dinov3_vits16",
        f"model.encoder.checkpoint_path={DINO_CHECKPOINT}",
        "model.transformer_encoder.dim=384",
        "model.transformer_encoder.depth=8",
        "model.transformer_encoder.num_heads=6",
        "model.transformer_encoder.ffn_dim=1024",
        "model.decoder.size=small",
        "model.decoder.channels=128",
        *(
            f"model.dense_head.{head}.hidden_channels=64"
            for head in ("kp", "seg", "line", "semantic_line")
        ),
        "training.compile.enabled=false",
        "training.checkpoint.save_top_k=0",
        f"training.step_schedule.max_steps={UPDATES}",
        f"training.step_schedule.validation_every_n_steps={UPDATES}",
        f"training.step_schedule.checkpoint_every_n_steps={UPDATES}",
        f"training.warmup_steps={WARMUP_UPDATES}",
        f"run.seed={seed}",
        f"run.output_dir=runs/{run_name(condition, seed)}",
    ]
    return [
        sys.executable,
        "-m",
        "src.tasks.court_detection.scripts.train",
        "--config-name",
        "train_data_scaling",
        *overrides,
    ]


def verify_inputs(
    campaign: Path, checkpoint_root: Path, external_asset_root: Path
) -> dict[str, Any]:
    prep = read_json(campaign / "preparation.json")
    registry = read_json(campaign / "registry.json")
    if (
        prep["real_registry_sha256"] != REAL_REGISTRY_SHA256
        or registry["sha256"] != REAL_REGISTRY_SHA256
    ):
        raise ValueError("The frozen real-video registry changed.")
    for name, expected in prep["manifests"].items():
        if file_sha256(campaign / "inputs/pool" / name) != expected["sha256"]:
            raise ValueError(f"Input subset changed: {name}")
    if file_sha256(campaign / "strata.json") != prep["real_strata_sha256"]:
        raise ValueError("The frozen real strata changed.")
    asset = external_asset_root / "dinov3"
    commit = subprocess.check_output(
        ["git", "-C", str(asset), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(asset), "status", "--porcelain"], text=True
    )
    if (
        commit != DINO_COMMIT
        or dirty
        or file_sha256(checkpoint_root / DINO_CHECKPOINT) != DINO_SHA256
    ):
        raise ValueError(
            "DINOv3 code or pretrained parameters disagree with the frozen experiment."
        )
    return prep


def checkpoint_for(
    run_dir: Path,
    prep: dict[str, Any],
    campaign: Path,
    condition: str,
    seed: int,
) -> Path:
    checkpoints = sorted(
        run_dir.glob(f"logs/version_*/checkpoints/update-{UPDATES:08d}.ckpt")
    )
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one checkpoint at update {UPDATES}: {checkpoints}")
    checkpoint = checkpoints[0]
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    hyper = payload["hyper_parameters"]
    measured = read_json(run_dir / "data-condition.json")
    train = read_json(campaign / "inputs/pool" / f"{condition}.train.subset.json")
    train_ids = [f"{row['scene_id']}:{row['sample_id']}" for row in train["samples"]]
    expected_hashes = {
        "train": prep["manifests"][f"{condition}.train.subset.json"]["sha256"],
        "val": prep["manifests"]["common.val.subset.json"]["sha256"],
    }
    if (
        payload["global_step"] != UPDATES
        or hyper["config"]["run"]["seed"] != seed
        or hyper["data_condition"] != measured
        or measured["training_seed"] != seed
        or measured["evaluation_registry_sha256"] != REAL_REGISTRY_SHA256
        or measured["ordered_training_ids_sha256"] != canonical_sha256(train_ids)
        or measured["dataset_subset"]["manifest_sha256"] != expected_hashes
        or measured["scene_images"] != dict.fromkeys(("B00", "B01", "B02", "B03"), 128)
        or measured["synthetic_validation"]["images"] != 64
    ):
        raise ValueError(
            "Checkpoint update/seed/source does not match its frozen condition."
        )
    return checkpoint


def score_image(prediction: Any, record: CourtSampleRecord) -> dict[str, Any]:
    if isinstance(prediction, ValueError):
        raise ValueError(f"Inference failed for {record.sample_id}") from prediction
    raw = prediction.raw_heads["kp"]
    if not isinstance(raw, CourtKeypointPrediction):
        raise TypeError("Expected Court ordered-keypoint inference.")
    width, height = record.payload["width"], record.payload["height"]
    if prediction.original_size_hw != (height, width):
        raise ValueError(f"Prediction dimensions drifted for {record.sample_id}.")
    return score_prediction(
        raw.keypoints[:, 0].numpy(),
        raw.scores[:, 0].numpy(),
        raw.valid[:, 0].numpy(),
        np.asarray(record.payload["keypoints"]),
        width,
        height,
    )


def evaluate_split(
    model: CourtPredictor,
    source: TennisCourtEvaluationInput,
    split: str,
    views: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    records = source.records(split)
    ids = [record.sample_id for record in records]
    if ids != source.registry["splits"][split] or ids != [
        view["sample_id"] for view in views
    ]:
        raise ValueError(
            f"Registry, annotations and stratum order disagree for {split}."
        )
    images, videos = REAL_SPLIT_COUNTS[split]
    if stratum_counts(views)["all"] != {"images": images, "videos": videos}:
        raise ValueError(f"Wrong evaluation population for {split}.")
    positions_by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        shape = model.batch_image_size(
            record.payload["height"], record.payload["width"]
        )
        positions_by_shape[shape].append(index)

    def prepare(record: CourtSampleRecord):
        return model.prepare_batch_image(source.load(record).image)

    scored: dict[int, dict[str, Any]] = {}
    print(
        f"Evaluating {split}: {images} images across {len(positions_by_shape)} input shapes",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        for positions in positions_by_shape.values():
            indices = iter(positions)
            while batch := list(islice(indices, BATCH_SIZE)):
                samples = [records[i] for i in batch]
                prepared = list(pool.map(prepare, samples))
                predictions = model.predict_prepared(
                    prepared, postprocess="none", heads=("kp",)
                )
                for index, record, prediction in zip(
                    batch, samples, predictions, strict=True
                ):
                    scored[index] = {**views[index], **score_image(prediction, record)}
    rows = [scored[index] for index in range(images)]
    write_json_atomic(output, {"split": split, "rows": rows})
    return {
        "path": output.name,
        "sha256": file_sha256(output),
        "ordered_id_sha256": canonical_sha256(ids),
        "strata": stratum_counts(views),
        "seconds": time.monotonic() - started,
    }


def run(
    campaign: Path,
    checkpoint_root: Path,
    external_asset_root: Path,
    condition: str,
    seed: int,
) -> None:
    started = time.monotonic()
    if os.environ.get("TENNIS_GPU_RESOURCE") != "all" or not os.environ.get(
        "TENNIS_RUN_ID"
    ):
        raise RuntimeError("This experiment requires an exclusive training-queue job.")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The registered GPU must support bf16 training.")
    prep = verify_inputs(campaign, checkpoint_root, external_asset_root)
    run_dir = campaign / "runs" / run_name(condition, seed)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    command = training_command(
        campaign, checkpoint_root, external_asset_root, condition, seed
    )
    training_start = time.monotonic()
    print(f"Training {condition} seed={seed} for {UPDATES} updates", flush=True)
    subprocess.run(command, check=True)
    training_seconds = time.monotonic() - training_start
    checkpoint = checkpoint_for(run_dir, prep, campaign, condition, seed)
    props = torch.cuda.get_device_properties(0)
    execution = {
        "condition": condition,
        "seed": seed,
        "updates": UPDATES,
        "train_command": command,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_queue_run_id": os.environ["TENNIS_RUN_ID"],
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "train_config_sha256": file_sha256(run_dir / "config.yaml"),
        "train_data_condition_sha256": file_sha256(run_dir / "data-condition.json"),
        "data_preparation_sha256": file_sha256(campaign / "preparation.json"),
        "real_registry_sha256": REAL_REGISTRY_SHA256,
        "real_strata_sha256": prep["real_strata_sha256"],
        "dino_commit": DINO_COMMIT,
        "dino_sha256": DINO_SHA256,
        "gpu": props.name,
        "gpu_capacity_bytes": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "opencv": cv2.__version__,
        "training_seconds": training_seconds,
        "eval_precision": "fp32; high matmul precision",
        "eval_batch_limit": BATCH_SIZE,
        "eval_nms_kernel": 1,
        "eval_decoder_threshold": 0.0,
    }
    write_json_atomic(run_dir / "fit.json", execution)

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    source = TennisCourtEvaluationInput(
        TennisCourtEvaluationSourceConfig(
            "tennis_court_evaluation",
            campaign / "inputs/tcd",
            campaign / "registry.json",
        )
    )
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=Path.cwd(),
            data_root=campaign / "inputs",
            checkpoint_root=checkpoint_root,
            external_asset_root=external_asset_root,
            artifact_root=campaign,
            output_root=campaign,
            cache_root=Path.cwd() / ".cache",
        )
    )
    model = CourtPredictor.load_from_checkpoint(
        checkpoint,
        device="cuda",
        resolver=resolver,
        checkpoint_role=PathRole.ARTIFACT,
        peak_threshold=0.0,
        nms_kernel=1,
    )
    kp = model.adapter.spec.target_bundle.targets["kp"]
    if (
        model.checkpoint_identity["checkpoint_sha256"] != execution["checkpoint_sha256"]
        or model.checkpoint_identity["backbone_sha256"] != DINO_SHA256
        or model.short_side != LONG_SIDE
        or not model.adapter.spec.pose_long_side
        or tuple(kp.channel_names) != GROUND_COURT_KP_NAMES
    ):
        raise ValueError(
            "Evaluation model or KP input geometry disagrees with the experiment."
        )
    views = read_json(campaign / "strata.json")
    evaluations = {
        split: evaluate_split(
            model, source, split, views[split], run_dir / f"{split}.json"
        )
        for split in REAL_SPLIT_COUNTS
    }
    receipt = {
        **execution,
        "evaluations": evaluations,
        "seconds": time.monotonic() - started,
    }
    write_json_atomic(run_dir / "complete.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "checkpoint-root", "external-asset-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    args = parser.parse_args()
    run(
        args.campaign.resolve(strict=True),
        args.checkpoint_root.resolve(strict=True),
        args.external_asset_root.resolve(strict=True),
        args.condition,
        args.seed,
    )


if __name__ == "__main__":
    main()
