"""GVHMR module for 3D human mesh estimation."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tennis_scene.pipeline.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class GVHMRConfig:
    """Configuration for GVHMR module.

    Attributes:
        model_checkpoint: Path to GVHMR model checkpoint.
        yolo_checkpoint: Path to YOLO checkpoint for tracking.
        vitpose_checkpoint: Path to ViTPose checkpoint.
        hmr2_checkpoint: Path to HMR2 checkpoint for feature extraction.
        venv_path: Path to GVHMR virtual environment directory.
        output_root: Root directory for GVHMR subprocess outputs.
        device: Inference device.

    """

    model_checkpoint: str | Path
    yolo_checkpoint: str | Path = "inputs/checkpoints/yolo/yolov8x.pt"
    vitpose_checkpoint: str | Path = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    hmr2_checkpoint: str | Path = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    venv_path: str | Path = "third_party/GVHMR/.venv"
    output_root: str | Path = "outputs/tennis_scene/gvhmr"
    device: str = "cuda"


@dataclass
class GVHMRResult:
    """Result of GVHMR inference.

    Attributes:
        smpl_body_pose: SMPL body pose parameters (T, 63).
        smpl_global_orient: SMPL global orientation (T, 3).
        smpl_betas: SMPL shape parameters (10,).
        smpl_vertices_local: Local SMPL vertices (T, V, 3) or None.
        human_kp_2d: 2D keypoints (T, 17, 2) in pixels.
        human_kp_vis: Keypoint visibility/confidence (T, 17).
        bbx_xys: Bounding boxes (T, 3) - center_x, center_y, size.

    """

    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32] | None
    human_kp_2d: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    bbx_xys: NDArray[np.float32]


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
        self._gvhmr_root: Path | None = None
        self._repo_root: Path | None = None
        self._venv_python: Path | None = None

    def load(self) -> None:
        """Validate GVHMR subprocess environment."""
        if self._venv_python is not None:
            return

        self._repo_root = Path(__file__).parents[3]
        self._gvhmr_root = self._repo_root / "third_party" / "GVHMR"
        self._venv_python = self._resolve_venv_python()

        if not self._venv_python.exists():
            raise FileNotFoundError(
                "GVHMR venv python not found. Run third_party/GVHMR/setup_gvhmr.sh first."
            )

    def _resolve_repo_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the repo root if not absolute."""
        if self._repo_root is None:
            self._repo_root = Path(__file__).parents[3]
        path = Path(path)
        return path if path.is_absolute() else self._repo_root / path

    def _resolve_venv_python(self) -> Path:
        """Resolve the GVHMR venv python executable path."""
        venv_path = self._resolve_repo_path(self.config.venv_path)
        if venv_path.is_dir():
            return venv_path / "bin" / "python"
        return venv_path

    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._venv_python is not None

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
        if not self.is_loaded:
            self.load()

        video_path = Path(video_path)

        outputs = self._run_subprocess(video_path)
        kp2d = outputs["kp2d"]
        bbx_xys = outputs["bbx_xys"]

        inference = outputs["inference"]
        human_kp_2d = kp2d[..., :2].cpu().numpy()
        human_kp_vis = kp2d[..., 2].cpu().numpy()

        if max_frames is not None:
            max_frames = int(max_frames)
            human_kp_2d = human_kp_2d[:max_frames]
            human_kp_vis = human_kp_vis[:max_frames]
            bbx_xys = bbx_xys[:max_frames]
            inference["smpl_body_pose"] = inference["smpl_body_pose"][:max_frames]
            inference["smpl_global_orient"] = inference["smpl_global_orient"][:max_frames]
            if inference["smpl_vertices_local"] is not None:
                inference["smpl_vertices_local"] = inference["smpl_vertices_local"][:max_frames]

        return GVHMRResult(
            smpl_body_pose=inference["smpl_body_pose"],
            smpl_global_orient=inference["smpl_global_orient"],
            smpl_betas=inference["smpl_betas"],
            smpl_vertices_local=inference["smpl_vertices_local"],
            human_kp_2d=human_kp_2d.astype(np.float32),
            human_kp_vis=human_kp_vis.astype(np.float32),
            bbx_xys=bbx_xys.cpu().numpy().astype(np.float32),
        )

    def _run_subprocess(self, video_path: Path) -> dict[str, Any]:
        """Run GVHMR via subprocess and load outputs."""
        if self._gvhmr_root is None or self._venv_python is None:
            raise RuntimeError("GVHMRModule.load() must be called before subprocess execution.")

        output_root = self._resolve_repo_path(self.config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self._venv_python),
            "tools/demo/demo.py",
            "--video",
            str(video_path),
            "--output_root",
            str(output_root),
            "-s",
        ]

        LOGGER.info("Running GVHMR subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(self._gvhmr_root), check=True)

        output_dir = output_root / video_path.stem
        preprocess_dir = output_dir / "preprocess"
        results_path = output_dir / "hmr4d_results.pt"
        if not results_path.exists():
            raise FileNotFoundError(f"GVHMR results not found at {results_path}")

        bbx_bundle = torch.load(preprocess_dir / "bbx.pt", weights_only=False)
        kp2d = torch.load(preprocess_dir / "vitpose.pt", weights_only=False)
        pred = torch.load(results_path, weights_only=False)

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
            "bbx_xys": bbx_bundle["bbx_xys"],
            "kp2d": kp2d,
            "inference": {
                "smpl_body_pose": body_pose.astype(np.float32),
                "smpl_global_orient": global_orient.astype(np.float32),
                "smpl_betas": betas.astype(np.float32),
                "smpl_vertices_local": vertices,
            },
        }


if __name__ == "__main__":
    module = GVHMRModule(GVHMRConfig(model_checkpoint="inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"))
    python_path = module._resolve_venv_python()
    print("GVHMRModule: subprocess runner")
    print(f"Resolved venv python: {python_path}")
    print(f"Exists: {python_path.exists()}")
