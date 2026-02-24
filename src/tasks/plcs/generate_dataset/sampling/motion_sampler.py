"""Motion sampler for AMASS/ACCAD dataset.

This module provides functionality to sample motion sequences from AMASS
dataset and compute 3D joint positions using SMPL-H model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class MotionSequence:
    """Container for a single motion sequence."""

    # Source information
    source_path: str
    category: str
    gender: str
    fps: float

    # Raw AMASS data
    poses: np.ndarray  # (T, 156)
    trans: np.ndarray  # (T, 3)
    betas: np.ndarray  # (num_betas,)

    # Computed 3D joints (set after SMPL-H forward)
    joints_3d: np.ndarray | None = None  # (T, J, 3)

    # Frame info
    num_frames: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.num_frames = self.poses.shape[0]


@dataclass
class MotionSourceConfig:
    """Configuration for a motion source category."""

    paths: list[str]
    weight: float = 1.0


class MotionSampler:
    """Sample motion sequences from AMASS dataset.

    This class handles:
    - Loading motion sequences from configured paths
    - Category-based weighted sampling
    - Computing 3D joints using SMPL-H model
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        smplh_model_path: Path | str = "data/smplx/smplh",
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the motion sampler.

        Args:
            config: Configuration with motion_sources settings.
            smplh_model_path: Path to SMPL-H model directory.
            device: Device for SMPL-H computation.

        """
        self.config = config or {}
        self.smplh_model_path = Path(smplh_model_path)
        self.device = torch.device(device)

        # Parse motion sources from config
        self._motion_sources: dict[str, MotionSourceConfig] = {}
        self._parse_motion_sources()

        # Index available motion files
        self._motion_files: dict[str, list[Path]] = {}
        self._index_motion_files()

        # SMPL-H models (loaded on demand)
        self._smplh_models: dict[str, object] = {}

    def _parse_motion_sources(self) -> None:
        """Parse motion source configuration."""
        sources_cfg = self.config.get("motion_sources", {})

        if not sources_cfg:
            # Default: use all ACCAD data
            self._motion_sources["default"] = MotionSourceConfig(
                paths=["data/ACCAD"],
                weight=1.0,
            )
            return

        for category, cfg in sources_cfg.items():
            # Handle both dict and OmegaConf DictConfig
            if hasattr(cfg, "get"):
                self._motion_sources[category] = MotionSourceConfig(
                    paths=list(cfg.get("paths", [])),
                    weight=float(cfg.get("weight", 1.0)),
                )

    def _index_motion_files(self) -> None:
        """Index all available motion files by category."""
        for category, source_cfg in self._motion_sources.items():
            files = []
            for path_str in source_cfg.paths:
                path = Path(path_str)
                if path.is_file() and path.suffix == ".npz":
                    files.append(path)
                elif path.is_dir():
                    # Find all *_poses.npz files recursively
                    files.extend(path.rglob("*_poses.npz"))
            self._motion_files[category] = files

        # Log statistics
        total_files = sum(len(f) for f in self._motion_files.values())
        print(f"MotionSampler: indexed {total_files} motion files")
        for cat, files in self._motion_files.items():
            print(f"  - {cat}: {len(files)} files")

    def _get_smplh_model(self, gender: str) -> object:
        """Get or create SMPL-H model for given gender.

        Args:
            gender: Gender string ('male', 'female', 'neutral').

        Returns:
            SMPL-H model instance.

        """
        gender_lower = gender.lower()
        if gender_lower not in self._smplh_models:
            try:
                import smplx

                model = smplx.create(
                    model_path=str(self.smplh_model_path.parent),
                    model_type="smplh",
                    gender=gender_lower,
                    num_betas=10,
                    use_pca=False,
                    ext="pkl",
                )
                model = model.to(self.device)
                model.eval()
                self._smplh_models[gender_lower] = model
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SMPL-H model for gender '{gender}': {e}"
                ) from e

        return self._smplh_models[gender_lower]

    def _infer_category_from_path(self, path: Path) -> str:
        """Infer motion category from file path.

        Args:
            path: Path to motion file.

        Returns:
            Inferred category string.

        """
        # Extract category from folder name (e.g., Female1Running_c3d -> Running)
        parent_name = path.parent.name
        # Common patterns: Female1Running_c3d, Male2Walking_c3d, etc.
        for keyword in ["Running", "Walking", "General", "Jump", "Stand"]:
            if keyword.lower() in parent_name.lower():
                return keyword.lower()
        return "general"

    def sample_motion(
        self,
        category: str | None = None,
        max_frames: int | None = None,
    ) -> MotionSequence:
        """Sample a random motion sequence.

        Args:
            category: Specific category to sample from. If None, uses weighted sampling.
            max_frames: Maximum number of frames to load. If None, loads all.

        Returns:
            MotionSequence with loaded data.

        """
        # Select category
        if category is not None:
            if category not in self._motion_files:
                raise ValueError(f"Unknown category: {category}")
            selected_category = category
        else:
            # Weighted random selection
            categories = list(self._motion_sources.keys())
            weights = [self._motion_sources[c].weight for c in categories]
            # Filter out categories with no files
            valid = [
                (c, w) for c, w in zip(categories, weights, strict=True) if self._motion_files.get(c)
            ]
            if not valid:
                raise RuntimeError("No motion files available")
            categories, weights = zip(*valid, strict=True)
            selected_category = random.choices(categories, weights=weights, k=1)[0]

        # Select random file from category
        files = self._motion_files[selected_category]
        if not files:
            raise RuntimeError(f"No files in category '{selected_category}'")
        selected_file = random.choice(files)

        # Load motion data
        return self.load_motion(selected_file, max_frames=max_frames)

    def load_motion(
        self,
        path: Path | str,
        max_frames: int | None = None,
    ) -> MotionSequence:
        """Load a specific motion file.

        Args:
            path: Path to AMASS npz file.
            max_frames: Maximum frames to load.

        Returns:
            MotionSequence with loaded data.

        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        poses = data["poses"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        betas = data["betas"].astype(np.float32)

        # Handle gender
        gender_raw = data["gender"]
        if isinstance(gender_raw, np.ndarray):
            gender = str(gender_raw.item())
        else:
            gender = str(gender_raw)
        # Clean up gender string (e.g., "b'female'" -> "female")
        gender = gender.strip("b'\"").lower()

        # Get FPS
        fps = float(data.get("mocap_framerate", np.array([60.0])).item())

        # Truncate if needed
        if max_frames is not None and poses.shape[0] > max_frames:
            poses = poses[:max_frames]
            trans = trans[:max_frames]

        category = self._infer_category_from_path(path)

        return MotionSequence(
            source_path=str(path),
            category=category,
            gender=gender,
            fps=fps,
            poses=poses,
            trans=trans,
            betas=betas,
        )

    def compute_joints_3d(
        self,
        motion: MotionSequence,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Compute 3D joints using SMPL-H model.

        Args:
            motion: Motion sequence with poses and betas.
            batch_size: Batch size for SMPL-H forward pass.

        Returns:
            3D joint positions, shape (T, J, 3).

        """
        model = self._get_smplh_model(motion.gender)

        T = motion.num_frames
        poses = motion.poses
        trans = motion.trans
        betas = motion.betas

        # Split poses into components
        # poses: (T, 156) = 52 joints * 3 axis-angle
        aa = poses.reshape(T, 52, 3)
        global_orient = aa[:, 0]  # (T, 3)
        body_pose = aa[:, 1:22].reshape(T, -1)  # (T, 63)
        left_hand_pose = aa[:, 37:52].reshape(T, -1)  # (T, 45)
        right_hand_pose = aa[:, 22:37].reshape(T, -1)  # (T, 45)

        # Prepare betas (truncate to model's num_betas)
        num_betas = min(betas.shape[0], 10)
        betas_truncated = betas[:num_betas]

        # Process in batches
        all_joints = []

        with torch.no_grad():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_t = end - start

                # Convert to tensors
                global_orient_t = torch.from_numpy(global_orient[start:end]).to(
                    self.device
                )
                body_pose_t = torch.from_numpy(body_pose[start:end]).to(self.device)
                left_hand_t = torch.from_numpy(left_hand_pose[start:end]).to(
                    self.device
                )
                right_hand_t = torch.from_numpy(right_hand_pose[start:end]).to(
                    self.device
                )
                transl_t = torch.from_numpy(trans[start:end]).to(self.device)
                betas_t = (
                    torch.from_numpy(betas_truncated[None, :])
                    .to(self.device)
                    .repeat(batch_t, 1)
                )

                output = model(
                    betas=betas_t,
                    global_orient=global_orient_t,
                    body_pose=body_pose_t,
                    left_hand_pose=left_hand_t,
                    right_hand_pose=right_hand_t,
                    transl=transl_t,
                    return_verts=False,
                )

                joints = output.joints.cpu().numpy()  # (batch_t, J, 3)
                all_joints.append(joints)

        joints_3d = np.concatenate(all_joints, axis=0)  # (T, J, 3)
        motion.joints_3d = joints_3d

        return joints_3d

    def get_available_categories(self) -> list[str]:
        """Get list of available motion categories.

        Returns:
            List of category names with available files.

        """
        return [c for c, files in self._motion_files.items() if files]

    def get_category_file_count(self, category: str) -> int:
        """Get number of files in a category.

        Args:
            category: Category name.

        Returns:
            Number of motion files.

        """
        return len(self._motion_files.get(category, []))
