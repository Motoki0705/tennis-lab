"""Evaluate a saved model or compare declared evaluation jobs."""

from src.utils.configuration.cli import dispatch_cli


def main() -> None:
    dispatch_cli("mode", {
        "heatmap": "src.tasks.ball_detection.evaluation.heatmap_cli",
        "manifest": "src.tasks.ball_detection.evaluation.manifest_cli",
        "coordinates": "src.tasks.ball_detection.evaluation.coordinate_cli",
        "hybrid": "src.tasks.ball_detection.evaluation.hybrid_cli",
        "temporal": "src.tasks.ball_detection.evaluation.temporal_cli",
    }, description="Ball detection evaluation; checkpoint contracts remain explicit.")


if __name__ == "__main__":
    main()
