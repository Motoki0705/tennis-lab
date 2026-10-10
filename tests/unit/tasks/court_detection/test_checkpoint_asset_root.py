"""The pretrained Court backbone belongs to the checkpoint asset root."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import CourtTrainingConfig


def test_backbone_uses_checkpoint_root_instead_of_external_source(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/court_detection/configs"), version_base="1.3"
    ):
        config = compose(
            config_name="train", overrides=[f"paths.project_root={tmp_path}"]
        )
    runtime = CourtTrainingConfig.from_config(config)
    weights = runtime.model.encoder.checkpoint_path
    assert weights is not None
    assert weights.is_relative_to(runtime.shared.resolver.roots.checkpoint_root)
    assert not weights.is_relative_to(runtime.shared.resolver.roots.external_asset_root)
