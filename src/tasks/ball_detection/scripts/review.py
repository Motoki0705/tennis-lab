"""Open the dataset review or model inference UI."""

from src.utils.configuration.cli import dispatch_cli


def main() -> None:
    dispatch_cli("mode", {
        "dataset": "src.tasks.ball_detection.visualization.review.cli",
        "inference": "src.tasks.ball_detection.visualization.inference.cli",
    }, description="Ball detection review UI.")


if __name__ == "__main__":
    main()
