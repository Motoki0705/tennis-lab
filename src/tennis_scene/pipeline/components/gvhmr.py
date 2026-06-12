"""GVHMR module for 3D human mesh estimation."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tennis_scene.pipeline.components.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

DEFAULT_GVHMR_VENV = Path("third_party/GVHMR/.venv/bin/python")


@dataclass
class GVHMRConfig:
    """Configuration for GVHMR module."""

    model_checkpoint: str | Path
    yolo_checkpoint: str | Path = "inputs/checkpoints/yolo/yolov8x.pt"
    vitpose_checkpoint: str | Path = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    hmr2_checkpoint: str | Path = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    device: str = "cuda"
    subprocess_mode: bool = False
    python_executable: str | Path | None = None
    smplx_model_type: str = "supermotion"
    smplx2smpl_path: str | Path = "hmr4d/utils/body_model/smplx2smpl_sparse.pt"
    smplx_body_model_path: str | Path | None = None
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved GVHMR result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> GVHMRResult:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class GVHMRModule(BasePipelineModule):
    """GVHMR module for 3D human mesh estimation."""

    def __init__(self, config: GVHMRConfig) -> None:
        self.config = config
        self._model = None
        self._tracker = None
        self._vitpose = None
        self._extractor = None
        self._smplx_model = None
        self._smplx2smpl = None
        self._gvhmr_root: Path | None = None

    def load(self) -> None:
        if self._model is not None:
            return

        self._gvhmr_root = Path(__file__).parents[3] / "third_party" / "GVHMR"
        sys.path.insert(0, str(self._gvhmr_root))

        self._load_tracker()
        self._load_vitpose()
        self._load_extractor()
        self._load_gvhmr_model()
        self._load_smpl_vertex_converter()

    def _load_tracker(self) -> None:
        LOGGER.info(f"Loading YOLO tracker from {self.config.yolo_checkpoint}")
        yolo_path = self._resolve_path(self.config.yolo_checkpoint)

        from hmr4d.utils.preproc.tracker import Tracker

        self._tracker = Tracker(yolo_checkpoint=str(yolo_path), device=self.config.device)

    def _load_vitpose(self) -> None:
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
        LOGGER.info(f"Loading HMR2 extractor from {self.config.hmr2_checkpoint}")
        hmr2_path = self._resolve_path(self.config.hmr2_checkpoint)

        from hmr4d.utils.preproc.vitfeat_extractor import Extractor

        self._extractor = Extractor(
            checkpoint_path=str(hmr2_path),
            device=self.config.device,
            tqdm_leave=True,
        )

    def _load_gvhmr_model(self) -> None:
        LOGGER.info(f"Loading GVHMR model from {self.config.model_checkpoint}")
        model_path = self._resolve_path(self.config.model_checkpoint)

        import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401
        import hydra
        from hmr4d.configs import register_store_gvhmr
        from hydra import compose, initialize_config_module

        register_store_gvhmr()
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            cfg = compose(
                config_name="demo",
                overrides=[
                    f"ckpt_path={model_path}",
                    "static_cam=True",
                    "video_name=dummy",
                ],
            )

        self._model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        self._model.load_pretrained_model(str(model_path))
        self._model.eval()
        if self.config.device == "cuda":
            self._model = self._model.cuda()

    def _load_smpl_vertex_converter(self) -> None:
        """Load SMPL-X body model and SMPL-X->SMPL vertex converter."""
        from hmr4d.utils.smplx_utils import make_smplx

        if self._gvhmr_root is None:
            raise RuntimeError("GVHMR root is not initialized")

        LOGGER.info(
            "Loading SMPL-X model for vertex reconstruction "
            f"(type={self.config.smplx_model_type})"
        )
        smplx_kwargs: dict[str, str] = {}
        if self.config.smplx_body_model_path is not None:
            smplx_kwargs["model_path"] = str(
                self._resolve_path(self.config.smplx_body_model_path)
            )
        smplx_model = make_smplx(self.config.smplx_model_type, **smplx_kwargs)

        converter_path = self._resolve_path(self.config.smplx2smpl_path)
        if not converter_path.exists():
            raise FileNotFoundError(f"SMPL-X to SMPL converter not found: {converter_path}")

        device = torch.device(
            "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        smplx_model = smplx_model.to(device)
        smplx_model.eval()
        smplx2smpl = torch.load(converter_path, map_location=device, weights_only=False)
        if not isinstance(smplx2smpl, torch.Tensor):
            smplx2smpl = torch.as_tensor(smplx2smpl)
        smplx2smpl = smplx2smpl.to(device=device, dtype=torch.float32)

        self._smplx_model = smplx_model
        self._smplx2smpl = smplx2smpl

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        resolved = self._gvhmr_root / path
        if resolved.exists():
            return resolved
        return path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

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
        if not self.is_loaded:
            self.load()

        video_path = str(video_path)
        track_boxes = self._tracker.get_multi_track(video_path)
        track_ids = sorted(int(track_id) for track_id in track_boxes)
        if not track_ids:
            raise RuntimeError("No tracks selected in tracker UI")

        players: list[dict[str, Any]] = []
        for track_id in track_ids:
            bbx_xyxy = track_boxes[track_id]
            if max_frames is not None and len(bbx_xyxy) > max_frames:
                bbx_xyxy = bbx_xyxy[:max_frames]

            LOGGER.info(f"Running GVHMR for track_id={track_id}")
            preproc = self._run_preprocessing_for_track(video_path, bbx_xyxy)
            inference = self._run_inference(preproc)

            players.append(
                {
                    "track_id": track_id,
                    "smpl_body_pose": inference["smpl_body_pose"],
                    "smpl_global_orient": inference["smpl_global_orient"],
                    "smpl_betas": inference["smpl_betas"],
                    "smpl_vertices_local": inference["smpl_vertices_local"],
                    "human_kp_2d": preproc["kp2d"][..., :2].cpu().numpy().astype(np.float32),
                    "human_kp_vis": preproc["kp2d"][..., 2].cpu().numpy().astype(np.float32),
                    "bbx_xys": preproc["bbx_xys"].cpu().numpy().astype(np.float32),
                }
            )

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
            smpl_vertices_local=(
                np.stack([p["smpl_vertices_local"] for p in players], axis=0)
                if all(p["smpl_vertices_local"] is not None for p in players)
                else None
            ),
            human_kp_2d=np.stack([p["human_kp_2d"] for p in players], axis=0),
            human_kp_vis=np.stack([p["human_kp_vis"] for p in players], axis=0),
            bbx_xys=np.stack([p["bbx_xys"] for p in players], axis=0),
            track_ids=np.array(track_ids, dtype=np.int32),
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result

    def _run_preprocessing_for_track(
        self,
        video_path: str,
        bbx_xyxy: torch.Tensor,
    ) -> dict[str, Any]:
        LOGGER.info("Running GVHMR preprocessing...")

        from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
        from hmr4d.utils.video_io_utils import get_video_lwh

        _, width, height = get_video_lwh(video_path)

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
        LOGGER.info("Running GVHMR inference...")

        from hmr4d.utils.geo_transform import compute_cam_angvel

        bbx_xys = preproc["bbx_xys"]
        kp2d = preproc["kp2d"]
        f_imgseq = preproc["f_imgseq"]
        K_fullimg = preproc["K_fullimg"]

        T = len(bbx_xys)
        K_fullimg_batch = K_fullimg.unsqueeze(0).expand(T, -1, -1)
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
        body_pose = smpl_params["body_pose"].cpu().numpy().astype(np.float32)
        global_orient = smpl_params["global_orient"].cpu().numpy().astype(np.float32)
        betas = smpl_params["betas"].cpu().numpy().astype(np.float32)

        if betas.ndim == 2:
            betas = betas[0]

        vertices = None
        if self._smplx_model is None or self._smplx2smpl is None:
            raise RuntimeError(
                "SMPL vertex converter is not loaded. Call load() before running inference."
            )

        smpl_param_tensors = {
            key: value.to(self._smplx2smpl.device) if isinstance(value, torch.Tensor) else value
            for key, value in smpl_params.items()
        }
        smplx_out = self._smplx_model(**smpl_param_tensors)
        vertices_smpl = torch.stack(
            [torch.matmul(self._smplx2smpl, verts) for verts in smplx_out.vertices],
            dim=0,
        )
        vertices = vertices_smpl.cpu().numpy().astype(np.float32)

        return {
            "smpl_body_pose": body_pose,
            "smpl_global_orient": global_orient,
            "smpl_betas": betas,
            "smpl_vertices_local": vertices,
        }


if __name__ == "__main__":
    import argparse

    from ultralytics.nn import tasks as _utasks

    def _torch_safe_load(file):
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, str(file)

    _utasks.torch_safe_load = _torch_safe_load

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
    parser.add_argument(
        "--smplx-model-type",
        type=str,
        default="supermotion",
        help="SMPL-X model preset type used to reconstruct vertices",
    )
    parser.add_argument(
        "--smplx2smpl-path",
        type=str,
        default="hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        help="Path to sparse SMPL-X->SMPL conversion matrix",
    )
    parser.add_argument(
        "--smplx-body-model-path",
        type=str,
        default=None,
        help="Optional path to SMPL/SMPL-X body model directory",
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
        smplx_model_type=args.smplx_model_type,
        smplx2smpl_path=args.smplx2smpl_path,
        smplx_body_model_path=args.smplx_body_model_path,
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
