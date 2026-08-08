"""Tests for explicit NHT GPU runtime configuration materialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.synthetic_data_generation.reconstruction.runtime_config import (
    NHTTrainingRuntime,
    resolved_nht_runtime_config,
    write_nht_runtime_config,
)


def _runtime(tmp_path: Path) -> NHTTrainingRuntime:
    python = tmp_path / "runtime/python"
    trainer = tmp_path / "runtime/simple_trainer_nht.py"
    python.parent.mkdir(parents=True)
    python.write_text("python", encoding="utf-8")
    trainer.write_text("trainer", encoding="utf-8")
    return NHTTrainingRuntime(python=python, trainer=trainer)


def _base(tmp_path: Path) -> Path:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "nht_pipeline_config_v1",
                "nht_training": {
                    "python": None,
                    "trainer": None,
                    "adapter": None,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_runtime_config_replaces_only_explicit_python_and_trainer(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    resolved = resolved_nht_runtime_config(_base(tmp_path), runtime=runtime)

    training = resolved["nht_training"]
    assert isinstance(training, dict)
    assert training == {
        "python": str(runtime.python),
        "trainer": str(runtime.trainer),
        "adapter": None,
    }


def test_runtime_config_is_written_atomically_to_the_fixed_stage_path(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    destination = (tmp_path / "B00/reconstruction/input-config.yaml").resolve()

    written = write_nht_runtime_config(
        _base(tmp_path).resolve(),
        destination,
        runtime=runtime,
    )

    assert written == destination
    assert written.is_file()
    assert not written.with_suffix(".yaml.tmp").exists()
    assert yaml.safe_load(written.read_text(encoding="utf-8"))["nht_training"][
        "trainer"
    ] == str(runtime.trainer)


def test_runtime_config_rejects_missing_typed_training_keys(tmp_path: Path) -> None:
    base = tmp_path / "production.yaml"
    base.write_text("nht_training:\n  python: null\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain trainer"):
        resolved_nht_runtime_config(base.resolve(), runtime=_runtime(tmp_path))
