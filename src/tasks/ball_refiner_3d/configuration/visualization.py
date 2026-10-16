"""Validated defaults for the local Dataset Review server."""

from dataclasses import dataclass

import yaml

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ReviewServerConfig:
    host: str
    port: int
    cpu_threads: int

    def __post_init__(self) -> None:
        if (
            self.host not in {"127.0.0.1", "localhost", "::1"}
            or not 1 <= self.port <= 65535
            or self.cpu_threads < 1
        ):
            raise ValueError(
                "Require a local host, valid port and positive CPU thread count"
            )


def review_server_config() -> ReviewServerConfig:
    path = (
        PROJECT_ROOT
        / "src/tasks/ball_refiner_3d/configs/visualization/dataset_review.yaml"
    )
    return parse_section(ReviewServerConfig, yaml.safe_load(path.read_text()))
