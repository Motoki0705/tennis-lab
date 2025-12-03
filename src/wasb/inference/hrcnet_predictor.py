"""HRCNet-based WASB ball detection predictor for tennis analysis.

This predictor has the same public API as WASBPredictor but uses
configs/model/hrcnet.yaml to construct the underlying model.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .wasb_predictor import _WASB_SRC_DIR, WASBPredictor


class HRCNetWASBPredictor(WASBPredictor):
    """WASB ball detection predictor using HRCNet backbone.

    The API is compatible with WASBPredictor. The difference is that this
    predictor builds the model configuration from ``configs/model/hrcnet.yaml``
    instead of ``configs/model/wasb.yaml``.
    """

    @classmethod
    def _build_config(
        cls,
        checkpoint_path: Path,
        gpus: list[int],
        score_threshold: float,
        max_disp: int,
    ) -> DictConfig:
        """Build configuration for WASB with HRCNet backbone.

        The structure is identical to WASBPredictor._build_config, but it
        reads ``hrcnet.yaml`` as the model configuration.
        """
        config_dir = _WASB_SRC_DIR / "configs"

        # Load base configs
        model_cfg = OmegaConf.load(config_dir / "model" / "hrcnet.yaml")
        detector_cfg = OmegaConf.load(config_dir / "detector" / "tracknetv2.yaml")
        tracker_cfg = OmegaConf.load(config_dir / "tracker" / "online.yaml")
        transform_cfg = OmegaConf.load(config_dir / "transform" / "default.yaml")
        dataloader_cfg = OmegaConf.load(config_dir / "dataloader" / "default.yaml")

        # Override settings
        detector_cfg["model_path"] = str(checkpoint_path)
        detector_cfg["postprocessor"]["score_threshold"] = score_threshold
        tracker_cfg["max_disp"] = max_disp

        # Build combined config (keep as DictConfig for attribute access)
        cfg = OmegaConf.create(
            {
                "model": model_cfg,
                "detector": detector_cfg,
                "tracker": tracker_cfg,
                "transform": transform_cfg,
                "dataloader": dataloader_cfg,
                "runner": {
                    "device": "cuda",
                    "gpus": gpus,
                },
            }
        )

        return cfg
