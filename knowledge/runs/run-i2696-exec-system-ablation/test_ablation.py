"""CPU checks for the fixed experiment boundary."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from weakref import ref

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pipeline_cli import autocast_forward, pt16_forwards
from prepare_engines import require_reference_plan
from run import BUNDLE, CASES, CLIPS, MODELS, REPO, cli_arguments, read_json

from src.tennis_scene.configuration import parse_video_batch_config
from src.tennis_scene.pipeline.definition import standard_model_components
from src.tennis_scene.scripts.generate_dataset import _compose_pipeline_config
from src.utils.models.tensorrt_artifacts import EngineManifest


@pytest.mark.parametrize("case", CASES)
def test_video_batch_cli_uses_the_pinned_conditions(case: str, tmp_path: Path) -> None:
    root = tmp_path / "output"
    run = root / "i2696/test"
    pass_dir = run / case / "fixture"
    clips = CLIPS[:1] if case == CASES[0] else CLIPS
    trt = case == CASES[3]
    engines = {name: f"tensorrt/{name}/plan" for name in MODELS}
    command = cli_arguments(case, root, run, pass_dir, clips, engines)
    with initialize_config_dir(
        config_dir=str(REPO / "src/tennis_scene/configs"), version_base="1.3"
    ):
        raw = compose(
            config_name="run_video_batch",
            overrides=command[4:],
            return_hydra_config=True,
        )
        assert (
            Path(OmegaConf.select(raw, "hydra.run.dir")) == pass_dir / "progress/hydra"
        )
        with open_dict(raw):
            del raw["hydra"]
        runtime = parse_video_batch_config(raw)
        _, pipeline = _compose_pipeline_config(runtime.generation)
    assert runtime.generation.clip_ids == clips
    assert runtime.generation.dataset_directory == pass_dir / "dataset"
    assert runtime.progress_directory == pass_dir / "progress"
    assert runtime.batching.enabled == (case in CASES[2:])
    assert (
        runtime.batching.store_cache_bytes == 0
        and runtime.batching.clips_per_pass is None
    )
    assert (
        runtime.batching.sizes.dino,
        runtime.batching.sizes.vitpose,
        runtime.batching.sizes.clip,
        runtime.batching.sizes.hmr2,
        pipeline.people.runtime.gvhmr.batch_size,
    ) == (1, 8, 16, 16, 4)
    assert (
        runtime.record_cuda
        and not runtime.record_cuda_synchronize
        and runtime.record_sample_interval == 0.5
    )
    assert pipeline.people.selection == ("dino", "vitpose-h")
    assert pipeline.tracking.encoder == "clipreid_vitb16_market1501"
    assert pipeline.player_association.players_per_side == 2
    assert (
        pipeline.ball_detection is None and not pipeline.enabled["ball_reconstruction"]
    )
    assert pipeline.people.uses_tensorrt == trt
    if not trt:
        with pt16_forwards():
            components = standard_model_components(
                pipeline, ("cam0", "cam1", "cam2", "cam3")
            )
            for stage, alias in (("hmr2_features", "features"), ("gvhmr", "mesh")):
                spec = components[stage].model_specs()[alias]
                assert spec.dtype == "float16"
                assert spec.settings["compute_precision"] == "cuda_autocast_fp16"
        appearance = components["person_appearance_features"].model_specs()["encoder"]
        assert appearance.dtype == "float16"
        assert (
            appearance.scheduling_key
            == components["person_tracking"].expected_encoder.scheduling_key
        )
        assert (
            appearance.scheduling_key
            == components["player_association"].model_specs()["encoder"].scheduling_key
        )


def test_forward_precision_preserves_gvhmr_decoder_contract_and_model_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = False

    @contextmanager
    def fake_autocast(device: str, *, dtype: torch.dtype) -> Iterator[None]:
        nonlocal entered
        assert (device, dtype) == ("cuda", torch.float16)
        entered = True
        yield
        entered = False

    class Denoiser(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
            assert entered
            return {
                "prediction": values.half() + 1,
                "mask": values > 0,
                "length": torch.tensor([2]),
            }

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    network = Denoiser()
    pointer = ref(network)
    autocast_forward(network, fp32_outputs=True)
    output = network(torch.tensor([-1.0, 3.0]))
    assert output["prediction"].dtype == torch.float32
    assert output["prediction"].tolist() == [0.0, 4.0]
    assert output["mask"].dtype == torch.bool and output["length"].dtype == torch.int64
    assert not entered
    del network
    assert pointer() is None


def test_engine_receipt_rejects_a_manifest_with_another_plan_sha256() -> None:
    from types import SimpleNamespace

    reference = read_json(
        BUNDLE.parent / "run-i2363-trt-player-video/qualification-v1.json"
    )
    expected = reference["engines"]["manifests"]["dino"]
    verified = EngineManifest.model_validate_json(json.dumps(expected))
    bundle = SimpleNamespace(root=Path("fixture"), engine=verified)
    require_reference_plan(bundle, expected)
    bundle.engine = verified.model_copy(update={"plan_sha256": "0" * 64})
    with pytest.raises(ValueError, match="differs from the #2363 qualification"):
        require_reference_plan(bundle, expected)
