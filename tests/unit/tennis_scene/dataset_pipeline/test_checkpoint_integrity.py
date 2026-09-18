"""Pinned model integrity uses actual bytes, strict config and stage subsets."""

from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.dataset_pipeline.checkpoint_integrity import (
    validate_checkpoint_sha256,
    verify_checkpoint_integrity,
)
from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
from src.utils.checksum import FileIntegrityError
from src.utils.paths import PROJECT_ROOT

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
ROLES = ("court", "dino", "vitpose", "plcs", "blcs", "dinov3")


def pins():
    return dict.fromkeys(ROLES, ABC_SHA256)


def test_actual_checkpoint_bytes_match_known_digest(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"abc")
    assert verify_checkpoint_integrity({"plcs": checkpoint}, pins()) == {
        "plcs": ABC_SHA256
    }


def test_mismatch_stops_without_updating_pins(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"changed")
    expected = pins()
    with pytest.raises(RuntimeError, match="Checkpoint SHA-256 mismatch for plcs"):
        verify_checkpoint_integrity({"plcs": checkpoint}, expected)
    assert expected == pins()


def test_stage_subset_does_not_read_unused_assets(tmp_path):
    checkpoint = tmp_path / "court.ckpt"
    checkpoint.write_bytes(b"abc")
    # The other five files do not exist. Their pins are still required, while
    # only the caller's stage-required court asset is opened.
    assert verify_checkpoint_integrity({"court": checkpoint}, pins()) == {
        "court": ABC_SHA256
    }
    assert not (tmp_path / "plcs.ckpt").exists()


def test_missing_required_file_propagates_integrity_error(tmp_path):
    with pytest.raises(FileIntegrityError):
        verify_checkpoint_integrity({"plcs": tmp_path / "absent.ckpt"}, pins())


def test_provider_failure_propagates_without_retry(tmp_path):
    failure = FileIntegrityError("providers disagree", details={})
    with (
        patch(
            "src.tennis_scene.dataset_pipeline.checkpoint_integrity.dual_sha256",
            side_effect=failure,
        ) as digest,
        pytest.raises(FileIntegrityError, match="providers disagree"),
    ):
        verify_checkpoint_integrity({"plcs": tmp_path / "model.ckpt"}, pins())
    digest.assert_called_once()


def test_omitted_pins_preserve_legacy_without_file_reads(tmp_path):
    assert verify_checkpoint_integrity({"plcs": tmp_path / "absent.ckpt"}, None) == {}


@pytest.mark.parametrize(
    "value", [None, 1, "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n"]
)
def test_digest_format_is_strict(value):
    expected = pins()
    expected["plcs"] = value
    with pytest.raises(ValueError, match="checkpoint_sha256.plcs"):
        validate_checkpoint_sha256(OmegaConf.create(expected))


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_six_roles_required(change):
    expected = pins()
    if change == "missing":
        del expected["blcs"]
    else:
        expected["other"] = ABC_SHA256
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_checkpoint_sha256(OmegaConf.create(expected))


def test_unknown_asset_role_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="Unknown checkpoint asset roles"):
        verify_checkpoint_integrity({"unknown": tmp_path / "absent"}, pins())


def profiles():
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        return compose(config_name="build_slcs_dataset"), compose(
            config_name="build_broadcast_slcs_dataset"
        )


def test_meiji_pins_are_fixed_and_broadcast_recipe_omits_pins():
    meiji, broadcast = profiles()
    assert validate_checkpoint_sha256(meiji.checkpoint_sha256) == {
        "court": "b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383",
        "dino": "e61688afe3af91b25955e9f9601d04b42068327a2e076ee871828089aa4ffed5",
        "vitpose": "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
        "plcs": "e851a8fe3fbc8273ed86fcf7876c34e941b77761e4ab2f264495228b5c12919a",
        "blcs": "dd0e54d296604f43f52fd33ebf53abc4ac71cbc441f86d49cea2d1dd17e0bf32",
        "dinov3": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
    }
    assert "checkpoint_sha256" not in broadcast


@pytest.mark.parametrize("value", [None, {}, {"plcs": ABC_SHA256}])
def test_config_boundary_rejects_invalid_pins_before_loading_dataset(value):
    meiji, _ = profiles()
    meiji.checkpoint_sha256 = value
    with (
        patch(
            "src.tennis_scene.dataset_pipeline.configuration.load_dataset_manifest"
        ) as load,
        pytest.raises((ValueError, TypeError), match="checkpoint_sha256"),
    ):
        DatasetBuildConfig.from_config(meiji)
    load.assert_not_called()
