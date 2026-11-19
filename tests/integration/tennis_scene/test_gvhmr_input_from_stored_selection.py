"""Selection and HMR2 artifacts drive temporal inference after process restart."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.body_placement import (
    BodyPlacementInput,
    BodyPlacementModule,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodySelection,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput, GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import (
    HMR2FeaturesInput,
    HMR2FeaturesModule,
    HMR2FeaturesOutput,
)
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    InferenceRuntime,
    model_identities,
)
from src.tennis_scene.pipeline.input_assembly.body import (
    GVHMRInputAssembler,
    HMR2FeaturesInputAssembler,
)
from src.tennis_scene.pipeline.model_assets import file_identity
from src.tennis_scene.pipeline.runner import ComponentNode, PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.unit.tennis_scene.pipeline.components.test_gvhmr import selected
from tests.unit.tennis_scene.pipeline.components.test_hmr2_features import (
    FakeFeatures,
    asymmetric_selection,
)
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_people_config,
    materialize_model_weights,
)


def test_gvhmr_input_reconstructed_from_selected_camera_artifact(tmp_path: Path) -> None:
    value = selected(tmp_path)
    store = ClipStore(tmp_path / 'store', {'clip': 'clip'}, memory_entries=0)
    codec = ArtifactCodec(BodyViewSelectionOutput)
    ref = store.publish('body_view_selection', value.selection, codec, schema='body_view_selection', version=1,
        identity={'policy': 'coverage'}, dependencies={}, provenance={'origin': 'component'})
    features_codec = ArtifactCodec(HMR2FeaturesOutput)
    feature_ref = store.publish('hmr2_features', value.features, features_codec, schema='hmr2_features', version=1,
        identity={'weights': 'hmr2'}, dependencies={'selection': ref}, provenance={'origin': 'component'})
    restarted = ClipStore(store.root, {'clip': 'clip'}, memory_entries=0)
    inputs = GVHMRInputAssembler().assemble(AssemblyContext(value.source),
        {'selection': restarted.load(ref, codec), 'features': restarted.load(feature_ref, features_codec)})
    inputs.validate()
    request = inputs.selection.selections[0].requests[0]
    assert request.video_path == tmp_path / 'cam1.mp4'
    np.testing.assert_array_equal(request.source_frames, [0, 2, 4])
    np.testing.assert_array_equal(request.boxes_xys, value.selection.selections[0].requests[0].boxes_xys)
    np.testing.assert_array_equal(inputs.features.bodies[0].segments[0].features[:, 0], [3, 8, 17])
    assert not inputs.features.bodies[0].segments[0].features.flags.writeable
    assert inputs.features.bodies[0].segments[0].features.dtype == np.float32


class StoredSelection:
    def __init__(self, value: BodyViewSelectionOutput) -> None:
        from src.tennis_scene.pipeline.contracts import ComponentIO
        self.io = ComponentIO('body_view_selection', type(None), BodyViewSelectionOutput, {}, 'body_view_selection')
        self.value = value

    def process(self, inputs: None) -> BodyViewSelectionOutput:
        return self.value


class NoInput:
    version = 1
    def assemble(self, context: Any, artifacts: Any) -> None:
        return None


class FeatureMesh:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def load(self) -> None:
        pass
    def unload(self) -> None:
        pass
    def predict(self, request: Any) -> Any:
        # The temporal result depends on every source feature and on the 2D pose.
        values = request.f_imgseq[:, :1] + request.kp2d[:, :1, 0]
        return SimpleNamespace(smpl_params_incam={name: values.repeat(1, width) for name, width in
            [('body_pose', 63), ('global_orient', 3), ('betas', 10), ('transl', 3)]})


def body_plan(tmp_path: Path, clips: int) -> ExecutionPlan:
    cfg = make_people_config(tmp_path)
    executions = []
    for index in range(clips):
        inputs = asymmetric_selection(tmp_path, f'clip-{index}')
        # Alternate person/segment layouts and crop values; equal lengths cannot hide swaps.
        if index:
            first, second, _ = inputs.selection.selections
            request = first.requests[1]
            request = replace(request, boxes_xys=request.boxes_xys + np.array([25, 0, 0], np.float32))
            inputs = replace(inputs, selection=BodyViewSelectionOutput((second, replace(first,
                requests=(request,), observed_samples=(2,)))))
        selection = StoredSelection(inputs.selection)
        features = materialize_model_weights(HMR2FeaturesModule(cfg))
        mesh = materialize_model_weights(GVHMRModule(cfg))
        ctx = AssemblyContext(inputs.source)
        def node(name: str, module: Any, assembler: Any, bindings: dict[str, str], ctx: AssemblyContext = ctx) -> ComponentNode:
            settings = {'models': model_identities(module.model_specs(), file_identity)} if isinstance(module, InferenceComponent) else {}
            return ComponentNode(name, module, module.io, assembler, bindings, ctx, settings, 'split-v2')
        nodes = (node('body_view_selection', selection, NoInput(), {}),
            node('hmr2_features', features, HMR2FeaturesInputAssembler(), {'selection': 'body_view_selection'}),
            node('gvhmr', mesh, GVHMRInputAssembler(), {'selection': 'body_view_selection', 'features': 'hmr2_features'}))
        store = ClipStore(tmp_path / f'store-{index}', {'clip': inputs.source.clip_id}, memory_entries=0)
        executions.append(ClipExecution(inputs.source.clip_id, nodes, store))
    return ExecutionPlan(executions)


@pytest.mark.parametrize('clips,cached_features', [(1, ()), (3, ()), (3, (0,)), (3, (2,)), (3, (0, 1, 2))],
    ids=['1', '3', '3-first-cached', '3-last-cached', '3-all-features-cached'])
def test_real_body_nodes_reuse_models_across_people_clips_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: int, cached_features: tuple[int, ...],
) -> None:
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', FakeFeatures)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', FeatureMesh)
    events = []
    @contextmanager
    def recorder(action: Any, model: str, **attributes: Any) -> Iterator[None]:
        events.append((action, model, attributes['instance']))
        yield
    if cached_features:
        prefill = body_plan(tmp_path, clips)
        PlanExecutor(prefill, model_event=recorder).run(targets=[(f'clip-{i}', 'hmr2_features') for i in cached_features])
        assert {key[0] for key in prefill.references} == {f'clip-{i}' for i in cached_features}
        assert all(key[1] != 'gvhmr' for key in prefill.references)
        events.clear()
    plan = body_plan(tmp_path, clips)
    executor = PlanExecutor(plan, model_event=recorder)
    executor.run()
    expected = ([] if len(cached_features) == clips else [('load', 'hmr2'), ('unload', 'hmr2')]) + [('load', 'gvhmr'), ('unload', 'gvhmr')]
    assert [(action, model) for action, model, _ in events] == expected
    first = executor.output(('clip-0', 'gvhmr'))
    assert [b.person_id for b in first.bodies] == [9, 2, 17]
    np.testing.assert_array_equal(first.bodies[0].segments[0].parameters.transl[:, 0], [10011, 10013, 10015])
    np.testing.assert_array_equal(first.bodies[0].segments[1].parameters.transl[:, 0], [20030, 20034])
    assert [s.observed_samples for s in first.bodies[0].segments] == [3, 2]
    assert first.bodies[1].segments[0].observed_samples == 3  # Four features include one interpolated crop.
    assert not first.bodies[2].segments
    if clips > 1:
        other = executor.output(('clip-1', 'gvhmr'))
        assert [b.person_id for b in other.bodies] == [2, 9]
        np.testing.assert_array_equal(other.bodies[1].segments[0].parameters.transl[:, 0], [22530, 22534])
    events.clear()
    fresh = body_plan(tmp_path, clips)
    PlanExecutor(fresh, model_event=recorder).run()
    assert not events and set(fresh.statuses.values()) == {'cached'}
    # Explicit load must work after both image and mesh weights disappear.
    cfg = make_people_config(tmp_path)
    cfg.hmr2_checkpoint.unlink()
    cfg.gvhmr_checkpoint.unlink()
    executions = [replace(clip, nodes=tuple(replace(n, source='load') for n in clip.nodes)) for clip in fresh.clips.values()]
    PlanExecutor(ExecutionPlan(executions), model_event=recorder).run()
    assert not events


def test_body_target_does_not_run_unselected_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', FakeFeatures)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', FeatureMesh)
    plan = body_plan(tmp_path, 3)
    PlanExecutor(plan).run(targets=[('clip-0', 'gvhmr')])
    assert set(plan.references) == {('clip-0', name) for name in ('body_view_selection', 'hmr2_features', 'gvhmr')}


def test_current_gvhmr_output_can_be_opened_by_pose_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.submodules.visualization.review.store import PoseReviewStore
    from src.tennis_scene.pipeline.artifacts import json_value

    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', FakeFeatures)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', FeatureMesh)
    clip = body_plan(tmp_path, 1).clips['clip-0']
    source = clip.nodes[0].context.source
    store = ClipStore(tmp_path / 'review-store', json_value(source), memory_entries=0)
    PlanExecutor(ExecutionPlan([replace(clip, store=store)])).run()

    review = PoseReviewStore(store.root)
    assert review.metadata()['artifacts']['gvhmr']['version'] == 2
    frame = review.frame(9, 'cam1', 3)
    np.testing.assert_array_equal(frame['parameters']['transl'], [10013, 10013, 10013])
    assert review.frame(9, 'cam1', 4)['parameters'] is None


@pytest.mark.parametrize('clips', [1, 3])
def test_separated_features_preserve_body_parameters_placement_and_asset_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: int) -> None:
    from src.tennis_scene.pipeline.body_types import BodyRecoveryRequest
    from tests.unit.tennis_scene.pipeline.test_trace_player_placement import (
        FRAMES,
        _build_case,
    )
    loads: Counter[tuple[str, str]] = Counter()
    class Frames(FakeFeatures):
        def predict(self, request: Any) -> Any:
            return SimpleNamespace(features=request.frame_indices[:, None].float().repeat(1, 1024))
    class Mesh(FeatureMesh):
        def predict(self, request: Any) -> Any:
            frames = request.f_imgseq[:, 0].numpy().astype(np.float64)
            count = len(frames)
            pose: NDArray[np.float32] = np.zeros((count, 63), np.float32)
            pose[:, 0] = .02 * frames
            betas: NDArray[np.float32] = np.zeros((count, 10), np.float32)
            betas[:, 0] = .25 if frames[0] == 0 else .75
            params = dict(body_pose=pose, global_orient=np.zeros((count, 3), np.float32), betas=betas,
                transl=np.column_stack((2 + .1 * frames, np.full(count, -.5), np.full(count, 3))).astype(np.float32))
            return SimpleNamespace(smpl_params_incam={name: torch.from_numpy(value) for name, value in params.items()})
    def geometry(config: Any) -> Any:
        result = _build_case().geometry
        result.load = lambda: None
        return result
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', Frames)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', Mesh)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.body_placement.SmplGeometry', geometry)
    cfg = make_people_config(tmp_path)
    case = _build_case()
    image = materialize_model_weights(HMR2FeaturesModule(cfg))
    mesh = materialize_model_weights(GVHMRModule(cfg))
    placement = materialize_model_weights(BodyPlacementModule(('cam0', 'cam1'), cfg, case.config, reprojection_px=20))
    modules: dict[str, InferenceComponent[Any, Any]] = {'hmr2_features': image, 'gvhmr': mesh, 'body_placement': placement}
    declarations = {(str(i), name): module.model_specs() for i in range(clips) for name, module in modules.items()}
    @contextmanager
    def recorder(action: Any, model: str, **attributes: Any) -> Iterator[None]:
        loads[(action, model)] += 1
        yield
    inputs = {}
    for i in range(clips):
        source = ClipSource(str(i), tuple(SourceVideo(c, tmp_path / str(i) / f'{c}.mp4', f'{i}-{c}',
            FRAMES, 30., 1280, 720) for c in ('cam0', 'cam1')))
        requests = tuple(BodyRecoveryRequest(source.video('cam0').path, segment.source_frames,
            np.zeros((len(segment.source_frames), 17, 3), np.float32), np.ones((len(segment.source_frames), 3), np.float32),
            source.size, case.cameras[0].intrinsic) for segment in case.recovered.bodies[0].segments)
        inputs[i] = HMR2FeaturesInput(source, BodyViewSelectionOutput((BodySelection(7, 'cam0', requests, (2, 3)),)))
    saved = {}
    recovered = {}
    with InferenceRuntime(declarations, model_event=recorder) as runtime:
        for i, value in inputs.items():
            with runtime.node((str(i), 'hmr2_features')) as borrower:
                saved[i] = image.process_with_models(value, borrower)
        for i, value in inputs.items():
            with runtime.node((str(i), 'gvhmr')) as borrower:
                recovered[i] = mesh.process_with_models(GVHMRInput(value.source, value.selection, saved[i]), borrower)
            for actual, expected in zip(recovered[i].bodies[0].segments, case.recovered.bodies[0].segments, strict=True):
                for name in ('body_pose', 'global_orient', 'betas', 'transl'):
                    np.testing.assert_array_equal(getattr(actual.parameters, name), getattr(expected.parameters, name))
        for i, value in inputs.items():
            with runtime.node((str(i), 'body_placement')) as borrower:
                result = placement.process_with_models(BodyPlacementInput(value.source,
                    CameraAlignmentOutput(SimpleNamespace(cameras=case.cameras)), case.raw, case.grouped,
                    PlayerTriangulationOutput(case.skeleton), recovered[i]), borrower).players
            assert result is not None
            np.testing.assert_array_equal(np.flatnonzero(result.smpl_valid[0]), [0, 1, 2, 4, 5])
            np.testing.assert_allclose(result.position[0, 0], [1.65625, -1.6875, 1.40625], atol=1e-6)
            np.testing.assert_allclose(result.betas[0], [.55] + [0] * 9, atol=1e-7)
    assert loads == {(action, model): 1 for action in ('load', 'unload') for model in ('hmr2', 'gvhmr', 'body_geometry')}


def test_load_rejects_pre_separation_body_parameter_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', FakeFeatures)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', FeatureMesh)
    original = body_plan(tmp_path, 1)
    clip = original.clips['clip-0']
    legacy_nodes = tuple(replace(n, io=replace(n.io, version=1)) if n.name == 'gvhmr' else n for n in clip.nodes)
    PlanExecutor(ExecutionPlan([replace(clip, nodes=legacy_nodes)])).run()
    fresh = body_plan(tmp_path, 1)
    clip = fresh.clips['clip-0']
    loaded = replace(clip, nodes=tuple(replace(n, source='load') for n in clip.nodes))
    with pytest.raises(ValueError, match='Loaded component contract mismatch: gvhmr'):
        PlanExecutor(ExecutionPlan([loaded])).run()


@pytest.mark.parametrize('precision,dtype', [('fp32', 'float32'), ('fp16', 'float16')])
def test_standard_body_declarations_with_person_precision_cache_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str, dtype: str,
) -> None:
    from src.tennis_scene.pipeline.definition import (
        enabled_model_assets,
        standard_definition,
    )
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
        materialize_assets,
        runtime,
    )

    factories: list[tuple[str, str]] = []
    class Features(FakeFeatures):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            factories.append(('hmr2', kwargs['device']))
    class Mesh(FeatureMesh):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            factories.append(('gvhmr', kwargs['device']))
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', Features)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', Mesh)
    cfg = runtime(tmp_path, overrides=('device=cuda', 'gvhmr.enabled=true',
        f'people_models.runtime.dino_detector.precision={precision}',
        f'people_models.runtime.vitpose.precision={precision}'))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    value = asymmetric_selection(tmp_path, 'clip-0')
    nodes = {n.name: n for n in standard_definition(cfg, value.source, code_identity='fixed')}
    marker = TennisSceneOrchestrator(cfg).publication_identity()
    for name in ('hmr2_features', 'gvhmr', 'body_placement'):
        assert marker['models'][name] == nodes[name].settings['models']
        component = nodes[name].component
        assert isinstance(component, InferenceComponent)
        assert {spec.dtype for spec in component.model_specs().values()} == {'float32'}
    assert marker['models']['person_detection']['detector']['dtype'] == dtype
    assert marker['models']['person_pose_features']['pose']['dtype'] == dtype
    assert marker['models']['person_tracking'] == {}
    assert nodes['person_tracking/cam0'].settings['expected_features']['models']['pose']['dtype'] == dtype
    assert not factories  # Planning and publication identities cannot load a model.
    selected = StoredSelection(value.selection)
    selection = ComponentNode('body_view_selection', selected, selected.io, NoInput(), {},
        AssemblyContext(value.source), {}, 'fixed')
    body_nodes = (selection, nodes['hmr2_features'], nodes['gvhmr'])
    store = ClipStore(tmp_path / 'body-store', {'clip': value.source.clip_id}, memory_entries=0)
    plan = ExecutionPlan([ClipExecution(value.source.clip_id, body_nodes, store)])
    executor = PlanExecutor(plan)
    executor.run()
    assert factories == [('hmr2', 'cuda'), ('gvhmr', 'cuda')]
    result = executor.output(('clip-0', 'gvhmr'))
    assert [b.person_id for b in result.bodies] == [9, 2, 17]
    np.testing.assert_array_equal(result.bodies[0].segments[1].source_frames, [10, 14])
    np.testing.assert_array_equal(result.bodies[0].segments[1].parameters.transl[:, 0], [20030, 20034])
    factories.clear()
    cached = ExecutionPlan([ClipExecution(value.source.clip_id, body_nodes, store)])
    PlanExecutor(cached).run()
    assert set(cached.statuses.values()) == {'cached'} and not factories
    loaded = ExecutionPlan([ClipExecution(value.source.clip_id,
        tuple(replace(n, source='load') for n in body_nodes), store)])
    PlanExecutor(loaded).run()
    assert set(loaded.statuses.values()) == {'loaded'} and not factories


@pytest.mark.parametrize('stage,failure', [
    ('hmr2_features', 'initialize'), ('hmr2_features', 'release'),
    ('gvhmr', 'initialize'), ('gvhmr', 'release'),
])
def test_body_failure_survives_measurement_suppression_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, failure: str,
) -> None:
    import weakref

    calls: list[str] = []
    instances: list[weakref.ReferenceType[Any]] = []
    class Failing(FakeFeatures, FeatureMesh):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            instances.append(weakref.ref(self))
        def load(self) -> None:
            calls.append('load')
            if failure == 'initialize':
                raise RuntimeError(f'{stage} {failure} failed')
        def unload(self) -> None:
            calls.append('unload')
            if failure == 'release':
                raise RuntimeError(f'{stage} {failure} failed')
        def predict(self, request: Any) -> Any:
            calls.append('predict')
            return FakeFeatures.predict(self, request) if stage == 'hmr2_features' else FeatureMesh.predict(self, request)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor',
        Failing if stage == 'hmr2_features' else FakeFeatures)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery',
        Failing if stage == 'gvhmr' else FeatureMesh)
    plan = body_plan(tmp_path, 1)
    @contextmanager
    def suppressing_event(*args: Any, **kwargs: Any) -> Iterator[None]:
        with suppress(RuntimeError):
            yield
    completed: list[str] = []
    with pytest.raises(RuntimeError, match=f'{stage} {failure} failed') as caught:
        PlanExecutor(plan, model_event=suppressing_event).run(on_clip_complete=completed.append)
    assert caught.value.__traceback__ is not None
    assert calls.count('load') == calls.count('unload') == 1
    assert calls.count('predict') == (3 if failure == 'release' else 0)
    assert all(instance() is None for instance in instances)
    assert plan.statuses[('clip-0', stage)] == 'failed'
    assert ('clip-0', stage) not in plan.references
    assert plan.clips['clip-0'].store.active(stage) is None and not completed


@pytest.mark.parametrize('stage', ['hmr2_features', 'gvhmr'])
def test_body_model_declaration_mutation_is_rejected_before_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', FakeFeatures)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', FeatureMesh)
    PlanExecutor(body_plan(tmp_path, 1)).run()
    fresh = body_plan(tmp_path, 1)
    module = fresh.nodes[('clip-0', stage)].component
    assert isinstance(module, (HMR2FeaturesModule, GVHMRModule))
    module.config = replace(module.config, runtime=replace(module.config.runtime, device='cuda'))
    events: list[tuple[Any, ...]] = []
    @contextmanager
    def recorder(*args: Any, **kwargs: Any) -> Iterator[None]:
        events.append(args)
        yield
    with pytest.raises(ValueError, match='differs from bound node identity'):
        PlanExecutor(fresh, model_event=recorder).run()
    assert fresh.statuses[('clip-0', stage)] == 'failed' and not events
