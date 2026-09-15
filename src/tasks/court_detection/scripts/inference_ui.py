"""Run the local court_detection inference UI."""

from src.tasks.base.visualization.detection.cli import path_boundary, serve

PATH_BOUNDARY = path_boundary("court_detection.inference_ui")


def main() -> None:
    serve("court_detection", "inference", port=8775, boundary=PATH_BOUNDARY)


if __name__ == "__main__":
    main()
