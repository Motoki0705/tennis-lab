"""Colab must stage the backbone at the path the Court training runtime reads."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.colab.workflow.jobs import load_job
from src.tasks.court_detection.configuration import CourtTrainingConfig


def test_court_job_stages_backbone_in_the_training_checkpoint_root(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    job = load_job(root / "scripts/colab/workflows/jobs/court_detection.toml")
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/court_detection/configs"), version_base="1.3"
    ):
        config = compose(
            config_name="train",
            overrides=[*job.default_args, f"paths.project_root={tmp_path}"],
        )
    runtime = CourtTrainingConfig.from_config(config)
    weights = runtime.model.encoder.checkpoint_path
    assert weights is not None
    inputs = [entry for entry in job.inputs if tmp_path / entry.destination == weights]
    assert len(inputs) == 1 and not inputs[0].writable
    assert weights.is_relative_to(runtime.shared.resolver.roots.checkpoint_root)
    assert not weights.is_relative_to(runtime.shared.resolver.roots.external_asset_root)
    assert "paths.checkpoint_root" in job.protected_override_keys
