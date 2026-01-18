"""Integrated pipeline for tennis scene 3D reconstruction.

This pipeline combines:
- Court KP Detection: 2D court keypoints (single frame, fixed camera)
- GVHMR: 2D skeleton + local SMPL (static_cam=True)
- WASB: 2D ball detection
- PLCS: 3D player position and yaw
- BLCS: 3D ball trajectory

Example:
    >>> pipeline = TennisScenePipeline.from_checkpoints(...)
    >>> result = pipeline.run("video.mp4")
    >>> result.save("output.npz")

Config entry point: `src/tennis_scene/configs/pipeline.yaml`
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

from src.tennis_scene.io import SceneResult
from src.tennis_scene.transforms import apply_plcs_transform_batch, normalize_keypoints

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class TennisScenePipeline:
    """End-to-end pipeline for tennis scene 3D reconstruction.

    Fixed camera assumptions:
    - Court keypoints estimated from a single frame (frame 0)
    - No camera rotation estimation (GVHMR static_cam=True)
    - GVHMR provides local SMPL only
    - PLCS position and yaw are applied to SMPL mesh

    Attributes:
        device: Inference device.
        court_kp_predictor: Court keypoint detector.
        wasb_pipeline: Ball detection pipeline.
        plcs_predictor: Player localization predictor.
        blcs_predictor: Ball localization predictor.

    """

    def __init__(
        self,
        court_kp_predictor: Any,
        wasb_pipeline: Any,
        plcs_predictor: Any,
        blcs_predictor: Any,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline with all predictors.

        Args:
            court_kp_predictor: CourtKeypointPredictor instance.
            wasb_pipeline: VideoBallLocalizationPipeline instance.
            plcs_predictor: PLCSPredictor instance.
            blcs_predictor: BLCSPredictor instance.
            device: Inference device.

        """
        self.device = device
        self.court_kp_predictor = court_kp_predictor
        self.wasb_pipeline = wasb_pipeline
        self.plcs_predictor = plcs_predictor
        self.blcs_predictor = blcs_predictor

        self._gvhmr_model: Any | None = None
        self._gvhmr_preproc: dict[str, Any] = {}

    @classmethod
    def from_checkpoints(
        cls,
        court_kp_checkpoint: str | Path,
        wasb_checkpoint: str | Path,
        plcs_checkpoint: str | Path,
        blcs_checkpoint: str | Path,
        device: str = "cuda",
        wasb_batch_size: int = 64,
        wasb_completion_enabled: bool = True,
        wasb_completion_checkpoint: str | Path | None = None,
    ) -> TennisScenePipeline:
        """Create pipeline from checkpoint paths.

        Args:
            court_kp_checkpoint: Path to court keypoint model checkpoint.
            wasb_checkpoint: Path to WASB model checkpoint.
            plcs_checkpoint: Path to PLCS model checkpoint.
            blcs_checkpoint: Path to BLCS model checkpoint.
            device: Inference device.
            wasb_batch_size: Batch size for WASB inference.
            wasb_completion_enabled: Enable trajectory completion.
            wasb_completion_checkpoint: Path to completion model checkpoint.

        Returns:
            Initialized TennisScenePipeline.

        """
        from src.court_detection.inference.predictor import CourtKeypointPredictor
        from src.plcs.inference.predictor import PLCSPredictor
        from src.blcs.inference.predictor import BLCSPredictor
        from src.wasb.inference import WASBPredictor, build_completer
        from src.wasb.pipeline import VideoBallLocalizationPipeline

        LOGGER.info("Loading Court KP predictor...")
        court_kp_predictor = CourtKeypointPredictor.from_checkpoint(
            court_kp_checkpoint, device=device
        )

        LOGGER.info("Loading WASB predictor...")
        wasb_predictor = WASBPredictor.load_from_checkpoint(
            wasb_checkpoint, device=device
        )

        completer = None
        if wasb_completion_enabled and wasb_completion_checkpoint is not None:
            LOGGER.info("Loading WASB completion model...")
            completer = build_completer(
                method="bilstm",
                checkpoint_path=str(wasb_completion_checkpoint),
                device=device,
            )

        wasb_pipeline = VideoBallLocalizationPipeline(
            wasb_predictor, completer=completer, batch_size=wasb_batch_size
        )

        LOGGER.info("Loading PLCS predictor...")
        plcs_predictor = PLCSPredictor.load_from_checkpoint(
            plcs_checkpoint, device=device
        )

        LOGGER.info("Loading BLCS predictor...")
        blcs_predictor = BLCSPredictor.load_from_checkpoint(
            blcs_checkpoint, device=device
        )

        return cls(
            court_kp_predictor=court_kp_predictor,
            wasb_pipeline=wasb_pipeline,
            plcs_predictor=plcs_predictor,
            blcs_predictor=blcs_predictor,
            device=device,
        )

    def _load_gvhmr(self, ckpt_path: str | Path) -> None:
        """Load GVHMR model and preprocessing utilities.

        Args:
            ckpt_path: Path to GVHMR checkpoint.

        """
        if self._gvhmr_model is not None:
            return

        LOGGER.info("Loading GVHMR model...")

        sys.path.insert(0, str(Path(__file__).parents[2] / "third_party" / "GVHMR"))

        from hmr4d.utils.preproc import Tracker, VitPoseExtractor, Extractor
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

        self._gvhmr_preproc = {
            "tracker": Tracker(),
            "vitpose": VitPoseExtractor(),
            "extractor": Extractor(),
        }

        self._gvhmr_model = DemoPL.load_pretrained_model(str(ckpt_path))
        self._gvhmr_model.eval()
        if self.device == "cuda":
            self._gvhmr_model = self._gvhmr_model.cuda()

    def _read_video_info(self, video_path: Path) -> dict[str, Any]:
        """Read video metadata.

        Args:
            video_path: Path to video file.

        Returns:
            Dict with fps, width, height, num_frames.

        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }

    def _read_frame(self, video_path: Path, frame_idx: int) -> NDArray[np.uint8]:
        """Read a single frame from video.

        Args:
            video_path: Path to video file.
            frame_idx: Frame index to read.

        Returns:
            RGB frame array (H, W, 3).

        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def _detect_court_keypoints(
        self, video_path: Path, frame_idx: int = 0
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Detect court keypoints from a single frame.

        Args:
            video_path: Path to video file.
            frame_idx: Frame index to use for detection.

        Returns:
            Tuple of (keypoints (20, 2), visibility (20,)).

        """
        LOGGER.info(f"Detecting court keypoints from frame {frame_idx}...")
        frame = self._read_frame(video_path, frame_idx)
        result = self.court_kp_predictor.predict(frame)

        video_info = self._read_video_info(video_path)
        keypoints_norm = normalize_keypoints(
            result["keypoints"].astype(np.float32),
            video_info["width"],
            video_info["height"],
        )

        return keypoints_norm, result["visibility"].astype(np.float32)

    def _run_wasb(
        self, video_path: Path, max_frames: int | None = None
    ) -> dict[str, NDArray]:
        """Run ball detection on video.

        Args:
            video_path: Path to video file.
            max_frames: Maximum frames to process.

        Returns:
            Dict with ball_uv, visibility, score.

        """
        LOGGER.info("Running WASB ball detection...")
        result = self.wasb_pipeline.run(video_path, max_frames=max_frames)

        video_info = self._read_video_info(video_path)
        ball_uv = normalize_keypoints(
            result.ball_xy_px.astype(np.float32),
            video_info["width"],
            video_info["height"],
        )

        return {
            "ball_uv": ball_uv,
            "visibility": result.visibility,
            "score": result.score,
        }

    def _run_gvhmr_preproc(
        self, video_path: Path, max_frames: int | None = None
    ) -> dict[str, Any]:
        """Run GVHMR preprocessing (tracking, VitPose, ViT features).

        Args:
            video_path: Path to video file.
            max_frames: Maximum frames to process.

        Returns:
            Dict with bbx_xys, kp2d, f_imgseq, K_fullimg.

        """
        LOGGER.info("Running GVHMR preprocessing...")

        tracker = self._gvhmr_preproc["tracker"]
        vitpose = self._gvhmr_preproc["vitpose"]
        extractor = self._gvhmr_preproc["extractor"]

        video_info = self._read_video_info(video_path)
        width, height = video_info["width"], video_info["height"]

        bbx_xyxy = tracker.get_one_track(str(video_path), max_frames=max_frames)
        if bbx_xyxy is None:
            raise RuntimeError("No person detected in video")

        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K

        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy)

        kp2d = vitpose.extract(str(video_path), bbx_xys)

        f_imgseq = extractor.extract_video_features(str(video_path), bbx_xys)

        K_fullimg = estimate_K(width, height)

        return {
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "f_imgseq": f_imgseq,
            "K_fullimg": K_fullimg,
            "width": width,
            "height": height,
        }

    def _run_gvhmr_inference(
        self, preproc_data: dict[str, Any]
    ) -> dict[str, NDArray]:
        """Run GVHMR inference (local SMPL only, static_cam=True).

        Args:
            preproc_data: Preprocessing results from _run_gvhmr_preproc.

        Returns:
            Dict with smpl_body_pose, smpl_global_orient, smpl_betas, vertices.

        """
        LOGGER.info("Running GVHMR inference...")

        bbx_xys = preproc_data["bbx_xys"]
        kp2d = preproc_data["kp2d"]
        f_imgseq = preproc_data["f_imgseq"]
        K_fullimg = preproc_data["K_fullimg"]

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

        if self.device == "cuda":
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()

        with torch.no_grad():
            pred = self._gvhmr_model.predict(data, static_cam=True)

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
            "smpl_body_pose": body_pose,
            "smpl_global_orient": global_orient,
            "smpl_betas": betas,
            "smpl_vertices_local": vertices,
        }

    def _run_plcs(
        self,
        human_kp_2d: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        human_vis: NDArray[np.float32] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> dict[str, NDArray]:
        """Run PLCS to get player 3D position and yaw.

        Args:
            human_kp_2d: Human 2D keypoints (T, 17, 2), normalized.
            court_kp: Court keypoints (20, 2), normalized.
            human_vis: Human keypoint visibility (T, 17).
            court_vis: Court keypoint visibility (20,).

        Returns:
            Dict with position (T, 3) and yaw (T,).

        """
        LOGGER.info("Running PLCS player localization...")

        T = len(human_kp_2d)
        positions = []
        yaws = []

        court_kp_t = torch.from_numpy(court_kp).float()
        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()

        for t in range(T):
            human_kp_t = torch.from_numpy(human_kp_2d[t]).float().unsqueeze(0)
            human_vis_t = None
            if human_vis is not None:
                human_vis_t = torch.from_numpy(human_vis[t]).float().unsqueeze(0)

            pred = self.plcs_predictor.predict(
                human_kp=human_kp_t,
                court_kp=court_kp_t.unsqueeze(0),
                human_vis=human_vis_t,
                court_vis=court_vis_t.unsqueeze(0) if court_vis_t is not None else None,
                denormalize=True,
            )

            positions.append(pred["position_meters"].squeeze(0).numpy())
            yaws.append(pred["yaw_radians"].item())

        return {
            "position": np.stack(positions, axis=0).astype(np.float32),
            "yaw": np.array(yaws, dtype=np.float32),
        }

    def _run_blcs(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Run BLCS to get ball 3D trajectory.

        Args:
            ball_uv: Ball 2D positions (T, 2), normalized.
            court_kp: Court keypoints (20, 2), normalized.
            ball_vis: Ball visibility mask (T,).
            court_vis: Court keypoint visibility (20,).

        Returns:
            Ball 3D trajectory (T, 3), meters.

        """
        LOGGER.info("Running BLCS ball localization...")

        ball_uv_t = torch.from_numpy(ball_uv).float()
        court_kp_t = torch.from_numpy(court_kp).float()

        ball_mask_t = None
        if ball_vis is not None:
            ball_mask_t = torch.from_numpy(ball_vis.astype(np.float32))

        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()

        pred = self.blcs_predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_mask=ball_mask_t,
            court_vis=court_vis_t,
            denormalize=True,
        )

        return pred["position"].squeeze(0).numpy().astype(np.float32)

    def run(
        self,
        video_path: str | Path,
        gvhmr_checkpoint: str | Path | None = None,
        max_frames: int | None = None,
        court_kp_frame: int = 0,
        skip_ball: bool = False,
        skip_gvhmr: bool = False,
    ) -> SceneResult:
        """Run the full tennis scene reconstruction pipeline.

        Args:
            video_path: Path to input video.
            gvhmr_checkpoint: Path to GVHMR checkpoint (required if not skip_gvhmr).
            max_frames: Maximum frames to process.
            court_kp_frame: Frame index for court keypoint detection.
            skip_ball: Skip ball detection and BLCS.
            skip_gvhmr: Skip GVHMR (use dummy SMPL data).

        Returns:
            SceneResult with all reconstruction data.

        """
        video_path = Path(video_path)
        video_info = self._read_video_info(video_path)

        court_kp, court_vis = self._detect_court_keypoints(video_path, court_kp_frame)

        if not skip_gvhmr:
            if gvhmr_checkpoint is None:
                raise ValueError("gvhmr_checkpoint required when skip_gvhmr=False")
            self._load_gvhmr(gvhmr_checkpoint)
            gvhmr_preproc = self._run_gvhmr_preproc(video_path, max_frames)
            gvhmr_result = self._run_gvhmr_inference(gvhmr_preproc)

            human_kp_2d = gvhmr_preproc["kp2d"][..., :2].cpu().numpy()
            human_kp_vis = gvhmr_preproc["kp2d"][..., 2].cpu().numpy()
            human_kp_2d_norm = normalize_keypoints(
                human_kp_2d, video_info["width"], video_info["height"]
            )
        else:
            T = max_frames or video_info["num_frames"]
            gvhmr_result = {
                "smpl_body_pose": np.zeros((T, 63), dtype=np.float32),
                "smpl_global_orient": np.zeros((T, 3), dtype=np.float32),
                "smpl_betas": np.zeros(10, dtype=np.float32),
                "smpl_vertices_local": None,
            }
            human_kp_2d_norm = np.zeros((T, 17, 2), dtype=np.float32)
            human_kp_vis = np.ones((T, 17), dtype=np.float32)

        plcs_result = self._run_plcs(
            human_kp_2d=human_kp_2d_norm,
            court_kp=court_kp,
            human_vis=human_kp_vis,
            court_vis=court_vis,
        )

        if gvhmr_result["smpl_vertices_local"] is not None:
            smpl_vertices_global = apply_plcs_transform_batch(
                gvhmr_result["smpl_vertices_local"],
                plcs_result["position"],
                plcs_result["yaw"],
            )
        else:
            smpl_vertices_global = None

        ball_uv = None
        ball_visibility = None
        ball_3d = None
        if not skip_ball:
            wasb_result = self._run_wasb(video_path, max_frames)
            ball_uv = wasb_result["ball_uv"]
            ball_visibility = wasb_result["visibility"]

            ball_3d = self._run_blcs(
                ball_uv=ball_uv,
                court_kp=court_kp,
                ball_vis=ball_visibility,
                court_vis=court_vis,
            )

        T = len(plcs_result["position"])

        return SceneResult(
            num_frames=T,
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"],
            court_kp=court_kp,
            court_vis=court_vis,
            player_position=plcs_result["position"],
            player_yaw=plcs_result["yaw"],
            smpl_body_pose=gvhmr_result["smpl_body_pose"],
            smpl_global_orient=gvhmr_result["smpl_global_orient"],
            smpl_betas=gvhmr_result["smpl_betas"],
            smpl_vertices_local=gvhmr_result["smpl_vertices_local"],
            smpl_vertices_global=smpl_vertices_global,
            ball_uv=ball_uv,
            ball_visibility=ball_visibility,
            ball_3d=ball_3d,
            human_kp_2d=human_kp_2d_norm,
            human_kp_vis=human_kp_vis,
            metadata={
                "video_path": str(video_path),
                "court_kp_frame": court_kp_frame,
            },
        )


if __name__ == "__main__":
    print("TennisScenePipeline module loaded successfully.")
    print("Use TennisScenePipeline.from_checkpoints(...) to create a pipeline.")
