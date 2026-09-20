# mypy: disallow_untyped_decorators=False
"""Tiny CPU streams cover witnessed execution and the actual legacy error handler."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def module() -> ModuleType:
    path = Path(__file__).with_name("witness.py")
    spec = importlib.util.spec_from_file_location("reuse_witness", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def fixture(module: ModuleType, tmp_path: Path) -> tuple[Any, Path, Mock]:
    source = tmp_path / "checkpoint"
    source.write_bytes(b"tiny checkpoint bytes")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    original = Mock(
        side_effect=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    )
    helper = module.load(module.HELPER, "fixture_capture")
    witness = module.Witness(
        original=original,
        helper=helper,
        checkpoint=source,
        evidence=evidence,
        expected=hashlib.sha256(source.read_bytes()).hexdigest(),
        metadata={},
    )
    helper.read_stream = Mock(wraps=helper.read_stream)
    return witness, source, original


def test_three_reads_verify_saved_bytes_and_finish(
    fixture: tuple[Any, Path, Mock],
) -> None:
    witness, source, original = fixture
    for _ in range(3):
        assert witness(source) == witness.expected
    witness.finish()
    assert witness.report["status"] == "passed"
    assert len(witness.report["comparisons"]) == 3
    assert witness.helper.read_stream.call_count == 3
    original.assert_not_called()
    saved = json.loads((witness.evidence / "ledger.json").read_text())
    assert saved["captured_calls"] == 3 and saved["legacy_run_completed"] is True


def test_prepublication_anomaly_stops_publish_and_all_future_live_reads(
    fixture: tuple[Any, Path, Mock],
) -> None:
    witness, source, _ = fixture
    witness(source)
    source.write_bytes(b"corrupt bytes")
    publish = Mock()
    with pytest.raises(ValueError, match="prepublication"):
        witness(source)
        publish()
    publish.assert_not_called()
    with pytest.raises(ValueError, match="not reopened"):
        witness(source)
    assert witness.helper.read_stream.call_count == 2
    assert witness.report["skipped_after_failure"] == 1
    assert witness.report["status"] == "failed"


def test_fourth_call_is_refused_without_read(fixture: tuple[Any, Path, Mock]) -> None:
    witness, source, _ = fixture
    for _ in range(3):
        witness(source)
    with pytest.raises(ValueError, match="Fourth"):
        witness(source)
    assert witness.helper.read_stream.call_count == 3
    assert witness.report["calls"] == 4


def test_other_path_keeps_original_hash(
    fixture: tuple[Any, Path, Mock], tmp_path: Path
) -> None:
    witness, _, original = fixture
    other = tmp_path / "other"
    other.write_bytes(b"other bytes")
    assert witness(other) == hashlib.sha256(b"other bytes").hexdigest()
    original.assert_called_once_with(other)
    witness.helper.read_stream.assert_not_called()


def test_saved_snapshot_failure_is_sticky(fixture: tuple[Any, Path, Mock]) -> None:
    witness, source, _ = fixture
    verify = witness.helper.verify_snapshot

    def corrupt_saved(report: dict[str, Any]) -> None:
        Path(report["snapshot"]).write_bytes(b"changed saved bytes")
        verify(report)

    witness.helper.verify_snapshot = corrupt_saved
    with pytest.raises(ValueError, match="snapshot gate"):
        witness(source)
    with pytest.raises(ValueError, match="not reopened"):
        witness(source)
    assert witness.helper.read_stream.call_count == 1


def test_actual_legacy_exception_handler_does_not_reread_checkpoint(
    module: ModuleType,
    fixture: tuple[Any, Path, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    witness, source, _ = fixture
    old = module.load(module.OLD, "fixture_old_reuse")
    old_inputs = old.Inputs

    # Seed the same Inputs.before entry that the initial checkpoint gate creates,
    # then trigger a failed second read before the rest of the costly fixture setup.
    def seeded_inputs() -> Any:
        inputs = old_inputs()
        inputs.before[str(source.resolve())] = witness(source)
        return inputs

    def fail_during_setup(cfg: Any) -> None:
        source.write_bytes(b"bad second stream")
        witness(source)

    monkeypatch.setattr(old, "Inputs", seeded_inputs)
    monkeypatch.setattr(old, "dual_sha256", witness)
    monkeypatch.setattr(old, "MAIN", tmp_path)
    monkeypatch.setattr(old.DatasetBuildConfig, "from_config", fail_during_setup)
    output = tmp_path / "outputs" / "legacy"
    with pytest.raises(ValueError, match="prepublication"):
        old.run(tmp_path / "source", tmp_path / "target", output)
    assert (output / "failure.json").exists()
    assert "not reopened" in (output / "inputs_after_failure.json").read_text()
    assert witness.helper.read_stream.call_count == 2
    assert witness.report["skipped_after_failure"] == 1


def test_finish_requires_exactly_three_reads(fixture: tuple[Any, Path, Mock]) -> None:
    witness, source, _ = fixture
    witness(source)
    with pytest.raises(ValueError, match="exactly three"):
        witness.finish()
    assert witness.report["status"] == "failed"


def test_wrapper_requires_explicitly_disabled_cuda(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        module.run(
            source=tmp_path / "source",
            target=tmp_path / "target",
            output=tmp_path / "output",
            evidence=tmp_path / "evidence",
        )
    assert not (tmp_path / "output").exists() and not (tmp_path / "evidence").exists()
