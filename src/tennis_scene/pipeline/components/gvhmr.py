"""GVHMR module for 3D human mesh estimation.

Runs the src/submodules model chain (YOLO tracking -> ViTPose -> HMR2 features
-> GVHMR) in the main ``.venv``; there is no dependency on ``third_party/GVHMR``
code anymore (checkpoints are read via the ``ckpt/`` symlinks).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.io import load_json, save_json

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.submodules.models import (
        GvhmrMeshRecovery,
        Hmr2FeatureExtractor,
        SmplVertexReconstructor,
        ViTPosePose2D,
        YoloPersonTracker,
    )

LOGGER = logging.getLogger(__name__)


@dataclass
class GVHMRConfig:
    """Configuration for GVHMR module."""

    model_checkpoint: str | Path
    yolo_checkpoint: str | Path = "ckpt/yolo/yolov8x.pt"
    vitpose_checkpoint: str | Path = "ckpt/vitpose/vitpose-h-multi-coco.pth"
    hmr2_checkpoint: str | Path = "ckpt/hmr2/epoch=10-step=25000.ckpt"
    device: str = "cuda"
    subprocess_mode: bool = False
    python_executable: str | Path | None = None
    smplx_body_model_path: str | Path | None = None
    track_selection: str = "interactive"  # "interactive" or "auto"
    num_tracks: int = 2  # used when track_selection == "auto"
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class GVHMRResult:
    """Result of GVHMR inference.

    Attributes:
        smpl_body_pose: SMPL body pose parameters, shape (P, T, 63).
        smpl_global_orient: SMPL global orientation, shape (P, T, 3).
        smpl_betas: SMPL shape parameters, shape (P, 10).
        smpl_vertices_local: Local SMPL vertices, shape (P, T, V, 3) or None.
        human_kp_2d: 2D keypoints in pixels, shape (P, T, 17, 2).
        human_kp_vis: Keypoint visibility/confidence, shape (P, T, 17).
        bbx_xys: Bounding boxes, shape (P, T, 3).
        track_ids: Track IDs aligned to player axis, shape (P,).
    """

    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32] | None
    human_kp_2d: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    bbx_xys: NDArray[np.float32]
    track_ids: NDArray[np.int32] | None = None

    def to_dict(self) -> dict:
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
        if self.track_ids is not None:
            result["track_ids"] = self.track_ids.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> GVHMRResult:
        smpl_body_pose = np.array(data["smpl_body_pose"], dtype=np.float32)
        smpl_global_orient = np.array(data["smpl_global_orient"], dtype=np.float32)
        smpl_betas = np.array(data["smpl_betas"], dtype=np.float32)
        human_kp_2d = np.array(data["human_kp_2d"], dtype=np.float32)
        human_kp_vis = np.array(data["human_kp_vis"], dtype=np.float32)
        bbx_xys = np.array(data["bbx_xys"], dtype=np.float32)

        vertices = data.get("smpl_vertices_local")
        smpl_vertices_local = None
        if vertices is not None:
            smpl_vertices_local = np.array(vertices, dtype=np.float32)

        track_ids = data.get("track_ids")
        if track_ids is not None:
            track_ids = np.array(track_ids, dtype=np.int32)

        return cls(
            smpl_body_pose=smpl_body_pose,
            smpl_global_orient=smpl_global_orient,
            smpl_betas=smpl_betas,
            smpl_vertices_local=smpl_vertices_local,
            human_kp_2d=human_kp_2d,
            human_kp_vis=human_kp_vis,
            bbx_xys=bbx_xys,
            track_ids=track_ids,
        )

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)
        LOGGER.info(f"Saved GVHMR result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> GVHMRResult:
        return cls.from_dict(load_json(path))


class GVHMRModule(BasePipelineModule):
    """GVHMR module for 3D human mesh estimation (src/submodules based)."""

    def __init__(self, config: GVHMRConfig) -> None:
        self.config = config
        self._tracker: YoloPersonTracker | None = None
        self._pose_model: ViTPosePose2D | None = None
        self._feature_model: Hmr2FeatureExtractor | None = None
        self._mesh_model: GvhmrMeshRecovery | None = None
        self._vertex_reconstructor: SmplVertexReconstructor | None = None

    def load(self) -> None:
        if self.is_loaded:
            return

        from src.submodules.models import (
            GvhmrMeshRecovery,
            Hmr2FeatureExtractor,
            SmplVertexReconstructor,
            ViTPosePose2D,
            YoloPersonTracker,
        )

        device = self.config.device
        self._tracker = YoloPersonTracker(
            checkpoint=self.config.yolo_checkpoint, device=device
        )
        self._pose_model = ViTPosePose2D(
            checkpoint=self.config.vitpose_checkpoint, device=device
        )
        self._feature_model = Hmr2FeatureExtractor(
            checkpoint=self.config.hmr2_checkpoint, device=device
        )
        self._mesh_model = GvhmrMeshRecovery(
            checkpoint=self.config.model_checkpoint, device=device
        )
        self._vertex_reconstructor = SmplVertexReconstructor(
            device=device,
            body_models_dir=self.config.smplx_body_model_path,
        )
        self._mesh_model.load()

    @property
    def is_loaded(self) -> bool:
        return self._mesh_model is not None

    def process(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        """Run GVHMR preprocessing and inference.

        Returns:
            GVHMRResult with shapes based on (P, T, ...).
        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading GVHMR result from {load_path} (skipping inference)")
                return GVHMRResult.load(load_path)
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running inference"
            )

        if self.config.subprocess_mode:
            return self._process_subprocess(video_path, max_frames)

        return self._process_direct(video_path, max_frames)

    def _process_subprocess(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        LOGGER.info("Running GVHMR in subprocess mode...")

        output_path = self.config.output_path
        if output_path is None:
            raise ValueError("output_path must be set for subprocess mode")
        output_path = Path(output_path)

        python_exec = self.config.python_executable
        if python_exec is None:
            python_exec = sys.executable

        cmd = [
            str(python_exec),
            "-m",
            "src.tennis_scene.pipeline.components.gvhmr",
            "--video",
            str(video_path),
            "--output",
            str(output_path),
            "--model-checkpoint",
            str(self.config.model_checkpoint),
            "--yolo-checkpoint",
            str(self.config.yolo_checkpoint),
            "--vitpose-checkpoint",
            str(self.config.vitpose_checkpoint),
            "--hmr2-checkpoint",
            str(self.config.hmr2_checkpoint),
            "--device",
            self.config.device,
            "--track-selection",
            self.config.track_selection,
            "--num-tracks",
            str(self.config.num_tracks),
        ]
        if self.config.smplx_body_model_path is not None:
            cmd.extend(
                ["--smplx-body-model-path", str(self.config.smplx_body_model_path)]
            )
        if max_frames is not None:
            cmd.extend(["--max-frames", str(max_frames)])

        LOGGER.info(f"Subprocess command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[4],
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
        from src.submodules.models import TrackRequest

        if not self.is_loaded:
            self.load()
        assert self._tracker is not None

        track_result = self._tracker.predict(
            TrackRequest(
                video_path=video_path,
                num_tracks=self.config.num_tracks,
                interactive=self.config.track_selection == "interactive",
            )
        )
        track_ids = track_result.track_ids
        if not track_ids:
            raise RuntimeError("No tracks selected")

        players: list[dict[str, Any]] = []
        for track_id in track_ids:
            bbx_xys = track_result.bbx_xys(track_id)
            if max_frames is not None and len(bbx_xys) > max_frames:
                bbx_xys = bbx_xys[:max_frames]

            LOGGER.info(f"Running GVHMR for track_id={track_id}")
            players.append(self._run_track(video_path, track_id, bbx_xys))

        frame_lengths = {p["human_kp_2d"].shape[0] for p in players}
        if len(frame_lengths) != 1:
            raise RuntimeError(
                f"Selected tracks have inconsistent frame lengths: {sorted(frame_lengths)}"
            )

        result = GVHMRResult(
            smpl_body_pose=np.stack([p["smpl_body_pose"] for p in players], axis=0),
            smpl_global_orient=np.stack(
                [p["smpl_global_orient"] for p in players], axis=0
            ),
            smpl_betas=np.stack([p["smpl_betas"] for p in players], axis=0),
            smpl_vertices_local=np.stack(
                [p["smpl_vertices_local"] for p in players], axis=0
            ),
            human_kp_2d=np.stack([p["human_kp_2d"] for p in players], axis=0),
            human_kp_vis=np.stack([p["human_kp_vis"] for p in players], axis=0),
            bbx_xys=np.stack([p["bbx_xys"] for p in players], axis=0),
            track_ids=np.array(track_ids, dtype=np.int32),
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result

    def _run_track(
        self,
        video_path: str | Path,
        track_id: int,
        bbx_xys: torch.Tensor,
    ) -> dict[str, Any]:
        """Run pose/feature extraction and GVHMR for one person track."""
        from src.submodules.models import (
            GvhmrRequest,
            ImageFeatureRequest,
            Pose2DRequest,
        )
        from src.utils.video.reader import probe_video_info

        assert self._pose_model is not None
        assert self._feature_model is not None
        assert self._mesh_model is not None
        assert self._vertex_reconstructor is not None

        info = probe_video_info(video_path)

        pose = self._pose_model.predict(
            Pose2DRequest(video_path=video_path, bbx_xys=bbx_xys)
        )
        features = self._feature_model.predict(
            ImageFeatureRequest(video_path=video_path, bbx_xys=bbx_xys)
        )
        gvhmr = self._mesh_model.predict(
            GvhmrRequest(
                kp2d=pose.keypoints,
                bbx_xys=bbx_xys,
                f_imgseq=features.features,
                width=info.width,
                height=info.height,
                static_cam=True,
            )
        )

        smpl_params = gvhmr.smpl_params_incam
        vertices = self._vertex_reconstructor.reconstruct(smpl_params)

        betas = smpl_params["betas"].numpy().astype(np.float32)
        if betas.ndim == 2:
            betas = betas[0]

        return {
            "track_id": track_id,
            "smpl_body_pose": smpl_params["body_pose"].numpy().astype(np.float32),
            "smpl_global_orient": smpl_params["global_orient"].numpy().astype(np.float32),
            "smpl_betas": betas,
            "smpl_vertices_local": vertices.numpy().astype(np.float32),
            "human_kp_2d": pose.keypoints[..., :2].numpy().astype(np.float32),
            "human_kp_vis": pose.keypoints[..., 2].numpy().astype(np.float32),
            "bbx_xys": bbx_xys.numpy().astype(np.float32),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GVHMR CLI for subprocess execution")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="ckpt/gvhmr/gvhmr_siga24_release.ckpt",
        help="Path to GVHMR model checkpoint",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        type=str,
        default="ckpt/yolo/yolov8x.pt",
        help="Path to YOLO checkpoint",
    )
    parser.add_argument(
        "--vitpose-checkpoint",
        type=str,
        default="ckpt/vitpose/vitpose-h-multi-coco.pth",
        help="Path to ViTPose checkpoint",
    )
    parser.add_argument(
        "--hmr2-checkpoint",
        type=str,
        default="ckpt/hmr2/epoch=10-step=25000.ckpt",
        help="Path to HMR2 checkpoint",
    )
    parser.add_argument(
        "--smplx-body-model-path",
        type=str,
        default=None,
        help="Optional path to SMPL/SMPL-X body model directory",
    )
    parser.add_argument(
        "--track-selection",
        type=str,
        default="interactive",
        choices=["interactive", "auto"],
        help="Track selection mode",
    )
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=2,
        help="Number of tracks in auto selection mode",
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
        smplx_body_model_path=args.smplx_body_model_path,
        track_selection=args.track_selection,
        num_tracks=args.num_tracks,
        device=args.device,
        subprocess_mode=False,
        save_result=True,
        output_path=args.output,
    )

    module = GVHMRModule(config)
    result = module.process(args.video, max_frames=args.max_frames)

    print(f"GVHMR completed. Result saved to {args.output}")
    print(f"  - players: {result.smpl_body_pose.shape[0]}")
    print(f"  - smpl_body_pose: {result.smpl_body_pose.shape}")
    print(f"  - smpl_global_orient: {result.smpl_global_orient.shape}")
    print(f"  - smpl_betas: {result.smpl_betas.shape}")
    print(f"  - human_kp_2d: {result.human_kp_2d.shape}")
