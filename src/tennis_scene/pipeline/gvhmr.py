"""GVHMR module for 3D human mesh estimation."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tennis_scene.pipeline.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

# Default GVHMR virtual environment path
DEFAULT_GVHMR_VENV = Path("third_party/GVHMR/.venv/bin/python")


@dataclass
class GVHMRConfig:
    """Configuration for GVHMR module.

    Attributes:
        model_checkpoint: Path to GVHMR model checkpoint.
        yolo_checkpoint: Path to YOLO checkpoint for tracking.
        vitpose_checkpoint: Path to ViTPose checkpoint.
        hmr2_checkpoint: Path to HMR2 checkpoint for feature extraction.
        device: Inference device.
        subprocess_mode: If True, run GVHMR in a subprocess with separate venv.
        python_executable: Python executable for subprocess mode.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).
        track_ids: Specific track IDs for multi-player mode. If None, auto-select.

    """

    model_checkpoint: str | Path
    yolo_checkpoint: str | Path = "inputs/checkpoints/yolo/yolov8x.pt"
    vitpose_checkpoint: str | Path = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    hmr2_checkpoint: str | Path = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    device: str = "cuda"
    subprocess_mode: bool = False
    python_executable: str | Path | None = None
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None
    track_ids: list[int] | None = None
    multi_player: bool = False  # Enable multi-player tracking


@dataclass
class GVHMRResult:
    """Result of GVHMR inference for a single player.

    Attributes:
        smpl_body_pose: SMPL body pose parameters (T, 63).
        smpl_global_orient: SMPL global orientation (T, 3).
        smpl_betas: SMPL shape parameters (10,).
        smpl_vertices_local: Local SMPL vertices (T, V, 3) or None.
        human_kp_2d: 2D keypoints (T, 17, 2) in pixels.
        human_kp_vis: Keypoint visibility/confidence (T, 17).
        bbx_xys: Bounding boxes (T, 3) - center_x, center_y, size.
        track_id: Track ID for this player (optional).

    """

    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32] | None
    human_kp_2d: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    bbx_xys: NDArray[np.float32]
    track_id: int | None = None

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        result = {
            "smpl_body_pose": self.smpl_body_pose.tolist(),
            "smpl_global_orient": self.smpl_global_orient.tolist(),
            "smpl_betas": self.smpl_betas.tolist(),
            "human_kp_2d": self.human_kp_2d.tolist(),
            "human_kp_vis": self.human_kp_vis.tolist(),
            "bbx_xys": self.bbx_xys.tolist(),
        }
        if self.smpl_vertices_local is not None:
            result["smpl_vertices_local"] = self.smpl_vertices_local.tolist()
        if self.track_id is not None:
            result["track_id"] = self.track_id
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "GVHMRResult":
        """Create result from dict."""
        vertices = None
        if "smpl_vertices_local" in data and data["smpl_vertices_local"] is not None:
            vertices = np.array(data["smpl_vertices_local"], dtype=np.float32)
        return cls(
            smpl_body_pose=np.array(data["smpl_body_pose"], dtype=np.float32),
            smpl_global_orient=np.array(data["smpl_global_orient"], dtype=np.float32),
            smpl_betas=np.array(data["smpl_betas"], dtype=np.float32),
            smpl_vertices_local=vertices,
            human_kp_2d=np.array(data["human_kp_2d"], dtype=np.float32),
            human_kp_vis=np.array(data["human_kp_vis"], dtype=np.float32),
            bbx_xys=np.array(data["bbx_xys"], dtype=np.float32),
            track_id=data.get("track_id"),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved GVHMR result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.smpl_body_pose.ndim != 2 or self.smpl_body_pose.shape[1] != 63:
            errors.append(
                f"smpl_body_pose shape must be (T, 63), got {self.smpl_body_pose.shape}"
            )
        if self.smpl_global_orient.ndim != 2 or self.smpl_global_orient.shape[1] != 3:
            errors.append(
                "smpl_global_orient shape must be (T, 3), "
                f"got {self.smpl_global_orient.shape}"
            )
        if self.smpl_betas.shape != (10,):
            errors.append(f"smpl_betas shape must be (10,), got {self.smpl_betas.shape}")
        if self.human_kp_2d.ndim != 3 or self.human_kp_2d.shape[1:] != (17, 2):
            errors.append(
                f"human_kp_2d shape must be (T, 17, 2), got {self.human_kp_2d.shape}"
            )
        if self.human_kp_vis.ndim != 2 or self.human_kp_vis.shape[1] != 17:
            errors.append(
                f"human_kp_vis shape must be (T, 17), got {self.human_kp_vis.shape}"
            )
        if self.bbx_xys.ndim != 2 or self.bbx_xys.shape[1] != 3:
            errors.append(f"bbx_xys shape must be (T, 3), got {self.bbx_xys.shape}")

        t_pose = self.smpl_body_pose.shape[0]
        if self.smpl_global_orient.shape[0] != t_pose:
            errors.append("smpl_global_orient length does not match smpl_body_pose")
        if self.human_kp_2d.shape[0] != t_pose:
            errors.append("human_kp_2d length does not match smpl_body_pose")
        if self.human_kp_vis.shape[0] != t_pose:
            errors.append("human_kp_vis length does not match smpl_body_pose")
        if self.bbx_xys.shape[0] != t_pose:
            errors.append("bbx_xys length does not match smpl_body_pose")

        if not np.isfinite(self.smpl_body_pose).all():
            errors.append("smpl_body_pose contains non-finite values")
        if not np.isfinite(self.smpl_global_orient).all():
            errors.append("smpl_global_orient contains non-finite values")
        if not np.isfinite(self.smpl_betas).all():
            errors.append("smpl_betas contains non-finite values")
        if not np.isfinite(self.human_kp_2d).all():
            errors.append("human_kp_2d contains non-finite values")
        if not np.isfinite(self.human_kp_vis).all():
            errors.append("human_kp_vis contains non-finite values")
        if not np.isfinite(self.bbx_xys).all():
            errors.append("bbx_xys contains non-finite values")

        if self.smpl_vertices_local is not None:
            if (
                self.smpl_vertices_local.ndim != 3
                or self.smpl_vertices_local.shape[0] != t_pose
                or self.smpl_vertices_local.shape[2] != 3
            ):
                errors.append(
                    "smpl_vertices_local shape must be (T, V, 3), "
                    f"got {self.smpl_vertices_local.shape}"
                )
            if not np.isfinite(self.smpl_vertices_local).all():
                errors.append("smpl_vertices_local contains non-finite values")

        if not np.isin(self.human_kp_vis, [0.0, 1.0]).all():
            errors.append("human_kp_vis must contain only 0 or 1")

        if np.any(self.bbx_xys[:, 2] <= 0):
            errors.append("bbx_xys size must be positive")

        if self.track_id is not None and self.track_id < 0:
            errors.append(f"track_id must be non-negative, got {self.track_id}")

        return len(errors) == 0, errors

    @classmethod
    def load(cls, path: str | Path) -> "GVHMRResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Check if this is a multi-player result
            if "players" in data:
                # Load as multi-player but return first player for compatibility
                multi = GVHMRMultiResult.from_dict(data)
                if multi.players:
                    first_id = next(iter(multi.players))
                    return multi.players[first_id]
            return cls.from_dict(data)


@dataclass
class GVHMRMultiResult:
    """Result of GVHMR inference for multiple players.

    Attributes:
        players: Dict mapping track_id to GVHMRResult.

    """

    players: dict[int, GVHMRResult]

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GVHMRMultiResult":
        """Create result from dict."""
        players = {}
        for k, v in data.get("players", {}).items():
            track_id = int(k)
            result = GVHMRResult.from_dict(v)
            result.track_id = track_id
            players[track_id] = result
        return cls(players=players)

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved GVHMR multi-player result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if not self.players:
            errors.append("players must not be empty")
            return False, errors
        for track_id, result in self.players.items():
            ok, result_errors = result.validate()
            if not ok:
                errors.extend([f"player {track_id}: {msg}" for msg in result_errors])
            if result.track_id is not None and result.track_id != track_id:
                errors.append(
                    f"player {track_id}: track_id mismatch ({result.track_id})"
                )
        return len(errors) == 0, errors

    @classmethod
    def load(cls, path: str | Path) -> "GVHMRMultiResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def get_first(self) -> GVHMRResult | None:
        """Get the first player result (for single-player compatibility)."""
        if self.players:
            first_id = next(iter(self.players))
            return self.players[first_id]
        return None


class GVHMRModule(BasePipelineModule):
    """GVHMR module for 3D human mesh estimation.

    Provides local SMPL parameters and 2D keypoints from video.
    Uses static_cam=True for fixed camera assumption.

    """

    def __init__(self, config: GVHMRConfig) -> None:
        """Initialize the module.

        Args:
            config: GVHMR configuration.

        """
        self.config = config
        self._model = None
        self._tracker = None
        self._vitpose = None
        self._extractor = None
        self._gvhmr_root: Path | None = None

    def load(self) -> None:
        """Load all GVHMR models and preprocessing utilities."""
        if self._model is not None:
            return

        self._gvhmr_root = Path(__file__).parents[3] / "third_party" / "GVHMR"
        sys.path.insert(0, str(self._gvhmr_root))

        self._load_tracker()
        self._load_vitpose()
        self._load_extractor()
        self._load_gvhmr_model()

    def _load_tracker(self) -> None:
        """Load YOLO tracker with custom checkpoint path."""
        LOGGER.info(f"Loading YOLO tracker from {self.config.yolo_checkpoint}")

        yolo_path = self._resolve_path(self.config.yolo_checkpoint)

        from hmr4d.utils.preproc.tracker import Tracker

        self._tracker = Tracker(
            yolo_checkpoint=str(yolo_path),
            device=self.config.device,
        )

    def _load_vitpose(self) -> None:
        """Load ViTPose with custom checkpoint path."""
        LOGGER.info(f"Loading ViTPose from {self.config.vitpose_checkpoint}")

        vitpose_path = self._resolve_path(self.config.vitpose_checkpoint)

        from hmr4d.utils.preproc.vitpose import VitPoseExtractor

        self._vitpose = VitPoseExtractor(
            checkpoint_path=str(vitpose_path),
            device=self.config.device,
            flip_test=True,
            tqdm_leave=True,
        )

    def _load_extractor(self) -> None:
        """Load HMR2 feature extractor with custom checkpoint path."""
        LOGGER.info(f"Loading HMR2 extractor from {self.config.hmr2_checkpoint}")

        hmr2_path = self._resolve_path(self.config.hmr2_checkpoint)

        from hmr4d.utils.preproc.vitfeat_extractor import Extractor

        self._extractor = Extractor(
            checkpoint_path=str(hmr2_path),
            device=self.config.device,
            tqdm_leave=True,
        )

    def _load_gvhmr_model(self) -> None:
        """Load GVHMR model using Hydra configuration."""
        LOGGER.info(f"Loading GVHMR model from {self.config.model_checkpoint}")

        model_path = self._resolve_path(self.config.model_checkpoint)

        import hydra
        from hydra import initialize_config_module, compose
        from hmr4d.configs import register_store_gvhmr

        # Import gvhmr_pl_demo to register it with MainStore
        import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401

        # Register GVHMR config store and compose config
        register_store_gvhmr()
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            cfg = compose(config_name="demo", overrides=[
                f"ckpt_path={model_path}",
                "static_cam=True",
                "video_name=dummy",
            ])

        # Instantiate model using Hydra
        self._model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        self._model.load_pretrained_model(str(model_path))
        self._model.eval()
        if self.config.device == "cuda":
            self._model = self._model.cuda()

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to GVHMR root if not absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        resolved = self._gvhmr_root / path
        if resolved.exists():
            return resolved
        return path

    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._model is not None

    def process(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        """Run GVHMR preprocessing and inference.

        Args:
            video_path: Path to input video.
            max_frames: Maximum frames to process.

        Returns:
            GVHMRResult with SMPL parameters and 2D keypoints.

        """
        # Check if we should load from pre-computed result
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading GVHMR result from {load_path} (skipping inference)")
                return GVHMRResult.load(load_path)
            else:
                LOGGER.warning(f"load_path specified but not found: {load_path}, running inference")

        if self.config.subprocess_mode:
            return self._process_subprocess(video_path, max_frames)

        return self._process_direct(video_path, max_frames)

    def _process_subprocess(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        """Run GVHMR in a subprocess with separate virtual environment."""
        LOGGER.info("Running GVHMR in subprocess mode...")

        output_path = self.config.output_path
        if output_path is None:
            raise ValueError("output_path must be set for subprocess mode")
        output_path = Path(output_path)

        python_exec = self.config.python_executable
        if python_exec is None:
            python_exec = DEFAULT_GVHMR_VENV
        python_exec = Path(python_exec)

        if not python_exec.exists():
            raise FileNotFoundError(
                f"GVHMR Python executable not found: {python_exec}. "
                "Run third_party/GVHMR/setup_gvhmr.sh to set up the environment."
            )

        cmd = [
            str(python_exec),
            "-m",
            "src.tennis_scene.pipeline.gvhmr",
            "--video", str(video_path),
            "--output", str(output_path),
            "--model-checkpoint", str(self.config.model_checkpoint),
            "--yolo-checkpoint", str(self.config.yolo_checkpoint),
            "--vitpose-checkpoint", str(self.config.vitpose_checkpoint),
            "--hmr2-checkpoint", str(self.config.hmr2_checkpoint),
            "--device", self.config.device,
        ]
        if max_frames is not None:
            cmd.extend(["--max-frames", str(max_frames)])

        LOGGER.info(f"Subprocess command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[3],
        )

        if result.returncode != 0:
            LOGGER.error(f"GVHMR subprocess failed:\n{result.stderr}")
            raise RuntimeError(f"GVHMR subprocess failed: {result.stderr}")

        LOGGER.info(f"GVHMR subprocess completed, loading result from {output_path}")
        return GVHMRResult.load(output_path)

    def _process_direct(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        """Run GVHMR directly in current process."""
        if not self.is_loaded:
            self.load()

        video_path = str(video_path)

        preproc = self._run_preprocessing(video_path, max_frames)
        inference = self._run_inference(preproc)

        human_kp_2d = preproc["kp2d"][..., :2].cpu().numpy()
        human_kp_vis = preproc["kp2d"][..., 2].cpu().numpy()

        result = GVHMRResult(
            smpl_body_pose=inference["smpl_body_pose"],
            smpl_global_orient=inference["smpl_global_orient"],
            smpl_betas=inference["smpl_betas"],
            smpl_vertices_local=inference["smpl_vertices_local"],
            human_kp_2d=human_kp_2d.astype(np.float32),
            human_kp_vis=human_kp_vis.astype(np.float32),
            bbx_xys=preproc["bbx_xys"].cpu().numpy().astype(np.float32),
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result

    def _run_preprocessing(
        self, video_path: str, max_frames: int | None
    ) -> dict[str, Any]:
        """Run GVHMR preprocessing pipeline."""
        LOGGER.info("Running GVHMR preprocessing...")

        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
        from hmr4d.utils.video_io_utils import get_video_lwh

        length, width, height = get_video_lwh(video_path)

        # Tracker does not support max_frames, process full video
        bbx_xyxy = self._tracker.get_one_track(video_path)
        if bbx_xyxy is None:
            raise RuntimeError("No person detected in video")

        # Truncate to max_frames if specified
        if max_frames is not None and len(bbx_xyxy) > max_frames:
            bbx_xyxy = bbx_xyxy[:max_frames]

        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy)
        kp2d = self._vitpose.extract(video_path, bbx_xys)
        f_imgseq = self._extractor.extract_video_features(video_path, bbx_xys)
        K_fullimg = estimate_K(width, height)

        return {
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "f_imgseq": f_imgseq,
            "K_fullimg": K_fullimg,
            "width": width,
            "height": height,
        }

    def _run_inference(self, preproc: dict[str, Any]) -> dict[str, NDArray]:
        """Run GVHMR inference with static_cam=True."""
        LOGGER.info("Running GVHMR inference...")

        from hmr4d.utils.geo_transform import compute_cam_angvel

        bbx_xys = preproc["bbx_xys"]
        kp2d = preproc["kp2d"]
        f_imgseq = preproc["f_imgseq"]
        K_fullimg = preproc["K_fullimg"]

        T = len(bbx_xys)
        K_fullimg_batch = K_fullimg.unsqueeze(0).expand(T, -1, -1)

        # For static_cam, use identity rotation matrices
        R_w2c = torch.eye(3).repeat(T, 1, 1)
        cam_angvel = compute_cam_angvel(R_w2c)

        data = {
            "length": torch.tensor(T),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg_batch,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,
        }

        if self.config.device == "cuda":
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()

        with torch.no_grad():
            pred = self._model.predict(data, static_cam=True)

        smpl_params = pred["smpl_params_incam"]

        body_pose = smpl_params["body_pose"].cpu().numpy()
        global_orient = smpl_params["global_orient"].cpu().numpy()
        betas = smpl_params["betas"].cpu().numpy()

        if betas.ndim == 2:
            betas = betas[0]

        vertices = None
        if "verts" in pred:
            vertices = pred["verts"].cpu().numpy()

        return {
            "smpl_body_pose": body_pose.astype(np.float32),
            "smpl_global_orient": global_orient.astype(np.float32),
            "smpl_betas": betas.astype(np.float32),
            "smpl_vertices_local": vertices,
        }


if __name__ == "__main__":
    """CLI entry point for GVHMR subprocess execution.

    This allows running GVHMR in a separate virtual environment.

    Example:
        third_party/GVHMR/.venv/bin/python -m src.tennis_scene.pipeline.gvhmr \
            --video data/samples/clip.mp4 \
            --output outputs/gvhmr_result.json \
            --model-checkpoint third_party/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    """
    import argparse
    # --- Torch 2.6+ の "weights_only=True" 既定への互換処理（Ultralyticsの重み読み込み用） ---
    from ultralytics.nn import tasks as _utasks
    def _torch_safe_load(file):
        # 公式の .pt を使う前提で weights_only=False で読み込む
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, str(file)  # ★ Ultralytics 側が (ckpt, weight) の2値を期待
    _utasks.torch_safe_load = _torch_safe_load
    # --------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="GVHMR CLI for subprocess execution")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to GVHMR model checkpoint",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        type=str,
        default="inputs/checkpoints/yolo/yolov8x.pt",
        help="Path to YOLO checkpoint",
    )
    parser.add_argument(
        "--vitpose-checkpoint",
        type=str,
        default="inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        help="Path to ViTPose checkpoint",
    )
    parser.add_argument(
        "--hmr2-checkpoint",
        type=str,
        default="inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
        help="Path to HMR2 checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = GVHMRConfig(
        model_checkpoint=args.model_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        vitpose_checkpoint=args.vitpose_checkpoint,
        hmr2_checkpoint=args.hmr2_checkpoint,
        device=args.device,
        subprocess_mode=False,
        save_result=True,
        output_path=args.output,
    )

    module = GVHMRModule(config)
    result = module.process(args.video, max_frames=args.max_frames)

    print(f"GVHMR completed. Result saved to {args.output}")
    print(f"  - smpl_body_pose shape: {result.smpl_body_pose.shape}")
    print(f"  - smpl_global_orient shape: {result.smpl_global_orient.shape}")
    print(f"  - smpl_betas shape: {result.smpl_betas.shape}")
    print(f"  - human_kp_2d shape: {result.human_kp_2d.shape}")
