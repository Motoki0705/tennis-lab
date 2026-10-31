"""Select a training family; each family owns its objective and restart policy."""

from src.utils.configuration.cli import dispatch_cli


def main() -> None:
    dispatch_cli("stage", {
        "convnext_heatmap": "src.tasks.ball_detection.training.convnext_heatmap.cli",
        "dpt_pretrain": "src.tasks.ball_detection.training.dpt_pretrain.runner",
        "coordinates": "src.tasks.ball_detection.training.coordinates",
        "hybrid": "src.tasks.ball_detection.training.hybrid.runner",
        "temporal": "src.tasks.ball_detection.training.temporal.cli",
    }, description="Ball detection training; choose an explicit learning stage.")


if __name__ == "__main__":
    main()
