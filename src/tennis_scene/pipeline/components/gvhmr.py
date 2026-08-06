"""GVHMR pipeline stage backed by a pre-resolved typed model chain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import src.tennis_scene.pipeline.model_io.gvhmr as gvhmr_io
from src.submodules.configuration import BundledModelAssetPaths, SubmoduleRuntimeConfig
from src.tennis_scene.pipeline.components.base import BasePipelineModule

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GVHMRConfig:
    """Validated configuration for GVHMR composition and stage I/O."""

    gvhmr_checkpoint: Path
    source: Literal["execute", "load"]
    detector: str
    yolo_checkpoint: Path
    dino_checkpoint: Path
    dino_repository: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    body_models_dir: Path
    bundled_assets: BundledModelAssetPaths
    runtime: SubmoduleRuntimeConfig
    track_selection: str
    num_tracks: int
    save_result: bool
    output_path: Path
    load_path: Path | None

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "GVHMR source='load' requires load_path; execute forbids it"
            )
        if self.detector not in {"yolo", "dino"}:
            raise ValueError(
                f"detector must be 'yolo' or 'dino', got {self.detector!r}"
            )
        if self.track_selection not in {"interactive", "auto"}:
            raise ValueError(
                "track_selection must be 'interactive' or 'auto', got "
                f"{self.track_selection!r}"
            )


class GVHMRModule(BasePipelineModule):
    """Load or execute GVHMR without selecting or decoding a model variant."""

    def __init__(
        self,
        config: GVHMRConfig,
        chain: gvhmr_io.GVHMRChain | None,
    ) -> None:
        if config.source == "execute" and chain is None:
            raise ValueError("GVHMR source='execute' requires a resolved chain.")
        if config.source == "load" and chain is not None:
            raise ValueError("GVHMR source='load' forbids an inference chain.")
        self.config = config
        self._chain = chain

    def load(self) -> None:
        if self._chain is not None:
            self._chain.load()

    @property
    def is_loaded(self) -> bool:
        return self._chain is None or self._chain.is_loaded

    def process(
        self,
        video_path: Path,
        max_frames: int | None = None,
    ) -> gvhmr_io.GVHMRResult:
        """Load an artifact or invoke the already composed typed chain."""
        if self.config.source == "load":
            load_path = self.config.load_path
            if load_path is None:
                raise RuntimeError("Validated load source is missing load_path")
            if not load_path.is_file():
                raise FileNotFoundError(f"GVHMR artifact not found: {load_path}")
            LOGGER.info("Loading GVHMR result from %s", load_path)
            return gvhmr_io.GVHMRResult.load(load_path)

        chain = self._chain
        if chain is None:
            raise RuntimeError("Validated execute source is missing its GVHMR chain")
        result = chain.predict(
            gvhmr_io.GVHMRChainRequest(
                video_path=video_path,
                max_frames=max_frames,
                num_tracks=self.config.num_tracks,
                interactive=self.config.track_selection == "interactive",
                bbox_enlarge=self.config.runtime.tracking.bbox_enlarge,
                static_cam=self.config.runtime.static_cam,
            )
        )
        if self.config.save_result:
            result.save(self.config.output_path)
        return result
