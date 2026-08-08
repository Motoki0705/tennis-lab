"""Cross-domain integration for the canonical dataset performance contract."""

from __future__ import annotations

from dataclasses import fields

from hydra import compose, initialize_config_dir

from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.dataset.blcs.contracts import BLCSSampleRecord
from src.synthetic_data_generation.dataset.blcs.handler import BLCSDatasetStageHandler
from src.synthetic_data_generation.dataset.court.handler import CourtDatasetStageHandler
from src.synthetic_data_generation.dataset.plcs.composition import PreparedAvatar
from src.synthetic_data_generation.dataset.plcs.handler import PLCSStageHandler
from src.synthetic_data_generation.pipeline import SceneWorkspace, StageName
from src.synthetic_data_generation.pipeline.application import build_stage_handlers
from src.synthetic_data_generation.pipeline.publication import StagePublisher
from src.synthetic_data_generation.pipeline.registry import canonical_registry
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"


def _runtime() -> ScenePipelineConfiguration:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        config = compose(config_name="run_scene_pipeline")
    return ScenePipelineConfiguration.from_config(config)


def test_composition_root_wires_config_owned_cross_domain_budgets() -> None:
    runtime = _runtime()
    handlers = build_stage_handlers(runtime)
    court = handlers["court_dataset"]
    blcs = handlers["blcs_dataset"]
    plcs = handlers["plcs_dataset"]

    assert isinstance(court, CourtDatasetStageHandler)
    assert isinstance(blcs, BLCSDatasetStageHandler)
    assert isinstance(plcs, PLCSStageHandler)
    assert court.configuration.performance is runtime.court.performance
    assert blcs.configuration.performance is runtime.blcs.performance
    assert plcs.configuration.performance is runtime.plcs.performance
    assert blcs.renderer.execution_device == runtime.blcs.performance.execution_device
    assert (
        blcs.renderer.maximum_batch_frames
        == runtime.blcs.performance.maximum_batch_frames
    )
    assert plcs.parameters.device == runtime.plcs.performance.execution_device
    assert (
        plcs.parameters.smplh_batch_size
        == runtime.plcs.performance.maximum_batch_frames
    )
    assert runtime.court.performance.maximum_nht_invocations == 8
    assert (
        runtime.blcs.performance.maximum_nht_invocations
        == runtime.blcs.trajectory_source.scene_count
        == 3
    )
    assert runtime.plcs.performance.maximum_nht_invocations == 1
    assert all(
        budget.require_cuda and budget.execution_device == "cuda:0"
        for budget in (
            runtime.court.performance,
            runtime.blcs.performance,
            runtime.plcs.performance,
        )
    )


def test_compact_contracts_have_no_dense_per_frame_compatibility_fields() -> None:
    blcs_fields = {field.name for field in fields(BLCSSampleRecord)}

    assert {"background_store", "foreground_chunk", "chunk_sample_index"} <= blcs_fields
    assert {"sample_npz", "sample_json", "rgb", "depth"}.isdisjoint(blcs_fields)
    assert hasattr(PreparedAvatar, "frame_tensors_batch")
    assert not hasattr(PreparedAvatar, "frame_tensors")
    for domain in ("blcs", "plcs"):
        sources = (
            PROJECT_ROOT / "src/synthetic_data_generation/dataset" / domain
        ).rglob("*.py")
        text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        assert "np.savez_compressed" not in text
        assert "sample_npz" not in text
        assert "sample_json" not in text


def test_stale_partial_dataset_attempts_are_discarded_for_all_domains(
    tmp_path,
) -> None:
    workspace = SceneWorkspace(scene_id="B00", root=tmp_path / "B00")
    registry = canonical_registry()

    for stage in (
        StageName.COURT_DATASET,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
    ):
        publisher = StagePublisher(workspace, registry.spec(stage))
        publisher.staging.mkdir(parents=True)
        (publisher.staging / "partial.bin").write_bytes(b"partial")

        prepared = publisher.prepare()

        assert prepared.is_dir()
        assert not any(prepared.iterdir())
        assert not (publisher.owner / "dataset.json").exists()
        publisher.abandon()
        assert not publisher.staging.exists()
