"""Compare DINO-format checkpoints on one player-store split.

Runs the deployed preprocessing (``DinoPersonDetector`` resize/normalization)
and decodes class id 1, so the COCO release (person) and exported player
checkpoints are measured identically. Usage::

    .venv/bin/python -m src.tasks.player_detection.scripts.evaluate \
        '+evaluate.checkpoints.player_ft=player_detection/<exported>.pth'
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src.tasks.player_detection.configuration import (
    EvaluateConfig,
    validate_evaluate_boundary,
)
from src.tasks.player_detection.evaluation.evaluator import (
    evaluate_checkpoints,
    write_comparison_markdown,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "player_detection.evaluate"
register_boundary_validator(_BOUNDARY, validate_evaluate_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="evaluate",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    config = EvaluateConfig.from_config(cfg)
    summary = evaluate_checkpoints(config)
    write_comparison_markdown(summary, config.output_dir / "comparison.md")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {config.output_dir}")


if __name__ == "__main__":
    main()
