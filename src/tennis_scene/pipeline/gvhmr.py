"""GVHMR module for 3D human mesh estimation."""

from __future__ import annotations

import logging
import sys
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
        device: Inference device.

    """

    model_checkpoint: str | Path
    yolo_checkpoint: str | Path = "inputs/checkpoints/yolo/yolov8x.pt"
    vitpose_checkpoint: str | Path = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    hmr2_checkpoint: str | Path = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
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

        from ultralytics import YOLO

        yolo_path = self._resolve_path(self.config.yolo_checkpoint)
        self._yolo = YOLO(str(yolo_path))

        from hmr4d.utils.preproc.tracker import Tracker

        self._tracker = Tracker.__new__(Tracker)
        self._tracker.yolo = self._yolo

    def _load_vitpose(self) -> None:
        """Load ViTPose with custom checkpoint path."""
        LOGGER.info(f"Loading ViTPose from {self.config.vitpose_checkpoint}")

        vitpose_path = self._resolve_path(self.config.vitpose_checkpoint)

        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.preproc.vitpose_pytorch import build_model

        self._vitpose = VitPoseExtractor.__new__(VitPoseExtractor)
        self._vitpose.pose = build_model("ViTPose_huge_coco_256x192", str(vitpose_path))
        self._vitpose.pose.cuda().eval()
        self._vitpose.flip_test = True
        self._vitpose.tqdm_leave = True

    def _load_extractor(self) -> None:
        """Load HMR2 feature extractor with custom checkpoint path."""
        LOGGER.info(f"Loading HMR2 extractor from {self.config.hmr2_checkpoint}")

        hmr2_path = self._resolve_path(self.config.hmr2_checkpoint)

        from hmr4d.network.hmr2 import load_hmr2
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor

        self._extractor = Extractor.__new__(Extractor)
        self._extractor.extractor = load_hmr2(str(hmr2_path)).cuda().eval()
        self._extractor.tqdm_leave = True

    def _load_gvhmr_model(self) -> None:
        """Load GVHMR model."""
        LOGGER.info(f"Loading GVHMR model from {self.config.model_checkpoint}")

        model_path = self._resolve_path(self.config.model_checkpoint)

        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

        self._model = DemoPL.load_pretrained_model(str(model_path))
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
        if not self.is_loaded:
            self.load()

        video_path = str(video_path)

        preproc = self._run_preprocessing(video_path, max_frames)
        inference = self._run_inference(preproc)

        human_kp_2d = preproc["kp2d"][..., :2].cpu().numpy()
        human_kp_vis = preproc["kp2d"][..., 2].cpu().numpy()

        return GVHMRResult(
            smpl_body_pose=inference["smpl_body_pose"],
            smpl_global_orient=inference["smpl_global_orient"],
            smpl_betas=inference["smpl_betas"],
            smpl_vertices_local=inference["smpl_vertices_local"],
            human_kp_2d=human_kp_2d.astype(np.float32),
            human_kp_vis=human_kp_vis.astype(np.float32),
            bbx_xys=preproc["bbx_xys"].cpu().numpy().astype(np.float32),
        )

    def _run_preprocessing(
        self, video_path: str, max_frames: int | None
    ) -> dict[str, Any]:
        """Run GVHMR preprocessing pipeline."""
        LOGGER.info("Running GVHMR preprocessing...")

        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
        from hmr4d.utils.video_io_utils import get_video_lwh

        length, width, height = get_video_lwh(video_path)

        bbx_xyxy = self._tracker.get_one_track(video_path, max_frames=max_frames)
        if bbx_xyxy is None:
            raise RuntimeError("No person detected in video")

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

        bbx_xys = preproc["bbx_xys"]
        kp2d = preproc["kp2d"]
        f_imgseq = preproc["f_imgseq"]
        K_fullimg = preproc["K_fullimg"]

        T = len(bbx_xys)
        K_fullimg_batch = K_fullimg.unsqueeze(0).expand(T, -1, -1)
        cam_angvel = torch.zeros(T, 3)

        data = {
            "length": T,
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
    print("GVHMRModule: GVHMR 3D human mesh estimation module")
    print("Use GVHMRModule(GVHMRConfig(...)) to create")
