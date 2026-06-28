from __future__ import annotations

from src.utils.hydra import hydra_main


def test_hydra_main_returns_a_decorator() -> None:
    decorator = hydra_main(version_base="1.3", config_path=None)
    assert callable(decorator)

    def sample(cfg: object) -> object:
        return cfg

    wrapped = decorator(sample)
    assert callable(wrapped)


def test_scripts_use_the_shared_wrapper() -> None:
    """The per-task CLI scripts now import the shared wrapper, not a local copy."""
    from src.tasks.ball_detection.scripts import eval as ball_eval
    from src.tasks.plcs.scripts import visualize as plcs_visualize

    assert ball_eval.hydra_main is hydra_main
    assert plcs_visualize.hydra_main is hydra_main
