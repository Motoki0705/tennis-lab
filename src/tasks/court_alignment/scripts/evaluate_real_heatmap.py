"""Evaluate a KP14 checkpoint on an alignment_line_heatmaps_v2 archive.

Usage:
    python -m src.tasks.court_alignment.scripts.evaluate_real_heatmap \
        real_evaluation.archive_path=... real_evaluation.manifest_path=... \
        real_evaluation.alignment_path=... real_evaluation.checkpoint_path=...

Notes:
    - Hydra loads configuration from ``src/tasks/court_alignment/configs``.
    - Numeric NPZ ``mean_probability`` is the sole input authority; PNGs are ignored.
    - Reported reference errors are relative to accepted alignment, not independent GT.
"""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from src.tasks.court_alignment import configuration as _configuration  # noqa: F401
from src.tasks.court_alignment.configuration import (
    CourtAlignmentRealHeatmapRuntimeConfig,
)
from src.tasks.court_alignment.evaluation.real_heatmap import evaluate_real_heatmap
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="evaluate_real_heatmap",
    validation_boundary="court_alignment.evaluate_real_heatmap",
)
def main(cfg: DictConfig) -> None:
    """Run one explicitly configured measured-heatmap evaluation."""
    runtime = CourtAlignmentRealHeatmapRuntimeConfig.from_config(cfg)
    model = instantiate(cfg.model)
    if not isinstance(model, nn.Module):
        raise TypeError("model configuration must instantiate torch.nn.Module.")
    evaluate_real_heatmap(runtime.require_request(), model)


if __name__ == "__main__":
    main()
