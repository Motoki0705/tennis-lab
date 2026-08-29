"""Authority scoping for isolated Court-dataset reruns."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import yaml

from src.synthetic_data_generation.pipeline.runner import (
    _configuration_authority,
    _court_report_scoped_authorities_match,
    _court_report_scoped_authority,
)

pytestmark = pytest.mark.unit


def _authority(
    *,
    court: str,
    blcs: str,
    plcs: str,
    pipeline_seed: int = 695,
) -> Mapping[str, object]:
    return _configuration_authority(
        yaml.safe_dump(
            {
                "request": {
                    "scene_id": "B00",
                    "source_video": "synthetic_data_generation/raw/video.mp4",
                    "targets": ["court"],
                    "from_stage": "court_dataset",
                },
                "dataset": {
                    "court": {"schema": court},
                    "blcs": {"schema": blcs},
                    "plcs": {"schema": plcs},
                },
                "pipeline": {"seed": pipeline_seed},
                "nht": {"backend": "public-cli"},
            },
            sort_keys=False,
        )
    )


def _scoped(authority: Mapping[str, object]) -> Mapping[str, object]:
    value = _court_report_scoped_authority(
        authority,
        require_court_target=True,
    )
    assert value is not None
    return value


def test_court_rerun_ignores_all_dataset_stage_config_changes() -> None:
    existing = _scoped(_authority(court="v2", blcs="legacy-blcs", plcs="legacy-plcs"))
    requested = _scoped(
        _authority(court="v3", blcs="current-blcs", plcs="current-plcs")
    )

    assert "dataset" not in existing
    assert "dataset" not in requested
    assert _court_report_scoped_authorities_match(existing, requested)


def test_court_rerun_still_rejects_shared_pipeline_changes() -> None:
    existing = _scoped(_authority(court="v2", blcs="legacy-blcs", plcs="legacy-plcs"))
    requested = _scoped(
        _authority(
            court="v3",
            blcs="current-blcs",
            plcs="current-plcs",
            pipeline_seed=696,
        )
    )

    assert not _court_report_scoped_authorities_match(existing, requested)
