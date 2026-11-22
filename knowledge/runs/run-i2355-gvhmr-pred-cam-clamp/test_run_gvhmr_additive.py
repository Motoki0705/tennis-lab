"""CPU contracts for the additive-mask runtime entry point."""

import json
from pathlib import Path

import pytest
from run_gvhmr_additive import require_additive


def write_conversion(directory: Path, mode: str) -> None:
    certificate = {"graph_rewrite": {"attention_mask": {"mode": mode}}}
    (directory / "conversion.json").write_text(json.dumps(certificate))


@pytest.mark.parametrize("mode", ["additive", "additive-finite"])
def test_require_additive_accepts_additive_certificates(tmp_path: Path, mode: str):
    write_conversion(tmp_path, mode)

    assert require_additive(["--condition", "2", "--conversion", str(tmp_path)]) == mode


def test_require_additive_rejects_bool_mask_plans(tmp_path: Path):
    write_conversion(tmp_path, "drop-expand")

    with pytest.raises(ValueError, match="not additive"):
        require_additive(["--conversion", str(tmp_path)])


def test_require_additive_needs_one_conversion():
    with pytest.raises(ValueError, match="one --conversion"):
        require_additive(["--condition", "2"])
