"""Preview supervision or augmentation without changing a training run."""

from src.utils.configuration.cli import dispatch_cli


def main() -> None:
    dispatch_cli("kind", {
        "spatial": "src.tasks.ball_detection.visualization.preview.spatial",
        "heatmaps": "src.tasks.ball_detection.visualization.preview.heatmaps",
        "video": "src.tasks.ball_detection.visualization.preview.video",
    }, description="Ball supervision and augmentation previews.")


if __name__ == "__main__":
    main()
