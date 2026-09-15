"""Run the local ball_detection review UI."""

from src.tasks.base.visualization.detection.cli import path_boundary, serve

PATH_BOUNDARY = path_boundary("ball_detection.review_dataset")


def main() -> None:
    serve("ball_detection", "review", port=8776, boundary=PATH_BOUNDARY)


if __name__ == "__main__":
    main()
