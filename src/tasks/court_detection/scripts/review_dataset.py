"""Run the local court_detection review UI."""

from src.tasks.base.visualization.detection.cli import path_boundary, serve

PATH_BOUNDARY = path_boundary("court_detection.review_dataset")


def main() -> None:
    serve("court_detection", "review", port=8774, boundary=PATH_BOUNDARY)


if __name__ == "__main__":
    main()
