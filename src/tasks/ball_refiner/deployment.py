"""Portable detector-only pilot bundles, without runtime training-data access."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_detection.model_io.normalization import BallImageNormalization
from src.tasks.ball_refiner.inference import RefinerPair
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

SCHEMA = "ball_refiner_2d_inference_bundle.v1"
CENTRE_SELECTION = "nearest_window_centre_then_earlier_start"


def _sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a SHA-256 content identity")
    return value


def _detector_requirements(raw: dict[str, Any]) -> DetectorRequirements:
    # These upstream dataclasses have defaults; a deployment contract may not
    # silently fill a missing field from today's defaults.
    for key, contract in (("normalization", BallImageNormalization), ("candidates", BallCandidateConfig)):
        if not isinstance(raw[key], dict) or set(raw[key]) != {field.name for field in fields(contract)}:
            raise ValueError(f"Inference detector {key} requires every explicit field")
    values = dict(raw)
    values["image_size_hw"] = tuple(values["image_size_hw"])
    values["normalization"] = BallImageNormalization(**values["normalization"])
    values["candidates"] = BallCandidateConfig(**values["candidates"])
    return DetectorRequirements(**values)


@dataclass(frozen=True)
class DetectorRequirements:
    """Exact checkpoint and input policy used to train the refiner."""

    checkpoint_sha256: str
    image_size_hw: tuple[int, int]
    normalization: BallImageNormalization
    candidates: BallCandidateConfig
    subpixel_refine: bool
    window_length: int
    stride: int

    def __post_init__(self) -> None:
        _sha256(self.checkpoint_sha256)
        if len(self.image_size_hw) != 2 or any(type(x) is not int or x <= 1 for x in self.image_size_hw):
            raise ValueError("Detector image size must contain integer H,W > 1")
        if (type(self.subpixel_refine) is not bool
                or any(type(x) is not int or x < 1 for x in (self.window_length, self.stride))
                or self.stride > self.window_length):
            raise ValueError("Invalid detector temporal/subpixel requirements")


@dataclass(frozen=True)
class InferenceBundle:
    directory: Path
    model_config: Refiner2DConfig
    detector: DetectorRequirements
    window_length: int
    stride: int
    weights_sha256: str
    manifest_sha256: str

    def load_model(self) -> RefinerPair:
        """Restore only exported weights, strictly; do not follow provenance paths."""
        if dual_sha256(self.directory / "manifest.json") != self.manifest_sha256:
            raise ValueError("Inference bundle manifest changed")
        path = self.directory / "weights.pt"
        if dual_sha256(path) != self.weights_sha256:
            raise ValueError("Inference bundle weights checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or any(
            not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()
        ):
            raise ValueError("Inference weights must be finite tensors")
        pair = build_ball_refiner_2d(self.model_config)
        pair.model.load_state_dict(state, strict=True)
        if dual_sha256(path) != self.weights_sha256:
            raise ValueError("Inference bundle weights changed during loading")
        return pair


def load_inference_bundle(directory: Path) -> InferenceBundle:
    """Read the self-contained contract; loading metadata never constructs a model."""
    path = directory / "manifest.json"
    manifest_hash = dual_sha256(path)
    raw = json.loads(path.read_text())
    if set(raw) != {"schema", "model_config", "detector", "window_length", "stride", "weights_sha256",
                    "calibration", "provenance", "window_selection", "coordinate_system"}:
        raise ValueError("Inference bundle fields do not match its schema")
    if (raw["schema"] != SCHEMA or raw["calibration"] != "uncalibrated"
            or raw["window_selection"] != CENTRE_SELECTION
            or raw["coordinate_system"] != "source_xy_div_size_minus_one"):
        raise ValueError("Unsupported inference bundle semantics")
    model = Refiner2DConfig(**raw["model_config"])
    if not model.use_detector or model.use_pose or model.use_court:
        raise ValueError("This bundle schema supports the explicit detector-only pilot")
    requirements = _detector_requirements(raw["detector"])
    length, stride = raw["window_length"], raw["stride"]
    if any(type(x) is not int or x < 1 for x in (length, stride)) or stride > length:
        raise ValueError("Invalid refiner inference windows")
    if model.patch_size != requirements.candidates.patch_size:
        raise ValueError("Refiner and detector patch sizes disagree")
    weights_hash = _sha256(raw["weights_sha256"])
    if dual_sha256(directory / "weights.pt") != weights_hash or dual_sha256(path) != manifest_hash:
        raise ValueError("Inference bundle checksum mismatch")
    return InferenceBundle(directory, model, requirements, length, stride, weights_hash, manifest_hash)


def export_pilot_bundle(training_run: Path, output: Path) -> InferenceBundle:
    """Export a completed, selected pilot. Data paths remain provenance only.

    Failed exports stay in a separate staging directory for inspection. Existing
    bundles are never overwritten. No detector, GPU, annotations or cache is read.
    """
    from src.tasks.ball_refiner.training.configuration import PilotConfig

    if not training_run.is_absolute() or not output.is_absolute():
        raise ValueError("Bundle input and output paths must be absolute")
    if output.exists():
        raise FileExistsError(f"Inference bundle already exists: {output}")
    paths = [training_run / name for name in ("config.yaml", "data_manifest.json", "best.json", "run_state.json")]
    hashes = {path.name: dual_sha256(path) for path in paths}
    state = json.loads((training_run / "run_state.json").read_text())
    best = json.loads((training_run / "best.json").read_text())
    data = json.loads((training_run / "data_manifest.json").read_text())
    config = PilotConfig.from_config(OmegaConf.load(training_run / "config.yaml"))
    if state["status"] != "complete" or data["schema"] != "ball_refiner_pilot_data.v1":
        raise ValueError("Bundle requires a completed pilot")
    if type(best["epoch"]) is not int or best["epoch"] < 0 or best["checkpoint"] != f"epoch-{best['epoch']:03d}.pt":
        raise ValueError("Invalid selected pilot checkpoint")
    checkpoint_path = training_run / best["checkpoint"]
    hashes[checkpoint_path.name] = dual_sha256(checkpoint_path)
    if hashes[checkpoint_path.name] != best["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != "ball_refiner_2d_checkpoint.v1"
            or checkpoint["model_config"] != asdict(config.model)
            or checkpoint["data_manifest_sha256"] != hashes["data_manifest.json"]
            or checkpoint["epoch"] != best["epoch"] or state["best_epoch"] != best["epoch"]
            or checkpoint["selection_nll_uv"] != best["selection_nll_uv"]):
        raise ValueError("Checkpoint config/data/selection identity mismatch")
    detector = data["detector"]
    if (detector["window_selection"] != CENTRE_SELECTION
            or detector["tail_policy"] != "backfill_real_frames_no_padding"
            or detector["rgb"] != "JPEG BGR -> INTER_LINEAR -> RGB float32 [0,1]"):
        raise ValueError("Unsupported pilot detector evidence policy")
    requirements = _detector_requirements({
        "checkpoint_sha256": detector["sha256"], "image_size_hw": detector["image_size_hw"],
        "normalization": detector["image_normalization"], "candidates": detector["candidates"],
        "subpixel_refine": detector["subpixel_refine"],
        "window_length": detector["window_length"], "stride": detector["stride"],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".exporting-", dir=output.parent))
    torch.save(checkpoint["state_dict"], staging / "weights.pt")
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "model_config": asdict(config.model), "detector": asdict(requirements),
        "window_length": config.window_length, "stride": config.stride,
        "weights_sha256": dual_sha256(staging / "weights.pt"), "calibration": "uncalibrated",
        "window_selection": CENTRE_SELECTION, "coordinate_system": "source_xy_div_size_minus_one",
        "provenance": {"training_run": str(training_run), "input_sha256": hashes,
                       "evidence_manifest_sha256": data["evidence_manifest_sha256"],
                       "training_rgb": detector["rgb"],
                       "runtime_rgb": "source video BGR -> INTER_LINEAR -> RGB float32 [0,1]",
                       "media_domain": "JPEG training cache and direct video decoding are not pixel-identical"},
    }
    write_json_atomic(staging / "manifest.json", manifest)
    load_inference_bundle(staging).load_model()
    if any(dual_sha256(training_run / name) != digest for name, digest in hashes.items()):
        raise ValueError("Pilot changed during bundle export")
    if output.exists():
        raise FileExistsError(f"Inference bundle already exists: {output}")
    os.rename(staging, output)
    return load_inference_bundle(output)
