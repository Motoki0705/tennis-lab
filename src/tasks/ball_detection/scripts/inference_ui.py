"""Run the local ball_detection inference UI."""

from src.tasks.base.visualization.detection.cli import path_boundary, serve

PATH_BOUNDARY = path_boundary("ball_detection.inference_ui")


def main() -> None:
    serve("ball_detection", "inference", port=8777, boundary=PATH_BOUNDARY)


if __name__ == "__main__":
    main()
