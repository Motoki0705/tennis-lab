"""Feature adoption binds models/preprocessing; scheduling does not bind pose to CLIP."""
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesModule,
    PersonPoseFeaturesModule,
)
from src.tennis_scene.pipeline.definition import (
    component_override_identity,
    standard_definition,
    standard_model_components,
)
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.integration.tennis_scene.test_person_feature_runtime import harness
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import runtime


def detector_overrides(case: Any) -> dict[str, Any]:
    return {node.name: node.component for node in case.nodes if node.io.name == 'person_detection'}


@pytest.mark.parametrize('stage', ['person_detection', 'player_selection'])
def test_neighbor_person_override_settings_cannot_share_cache_or_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    from copy import copy
    case = harness(tmp_path, monkeypatch, 'fp32')
    before = standard_model_components(case.cfg, case.source.camera_ids)[stage]
    after = copy(before)
    if stage == 'person_detection':
        after.merge_duplicates = not before.merge_duplicates
    else:
        after.config = replace(before.config, max_candidates=1)
    initial = component_override_identity(case.cfg, before)
    changed = component_override_identity(case.cfg, after)
    assert initial['models'] == changed['models']
    assert initial != changed
    key = f'{stage}/cam0'
    assert (TennisSceneOrchestrator(case.cfg, components={key: before}).publication_identity()
            != TennisSceneOrchestrator(case.cfg, components={key: after}).publication_identity())


def changed_component(component: Any, change: str, tmp_path: Path) -> Any:
    if change in ('match', 'unknown'):
        return component
    if change == 'preprocessing':
        config = replace(component.config, **({'bbox_enlarge': 2.} if isinstance(component, PersonPoseFeaturesModule)
                                            else {'min_appearance_height_px': 1000.}))
        return (PersonPoseFeaturesModule(component.people, config) if isinstance(component, PersonPoseFeaturesModule)
                else PersonAppearanceFeaturesModule(component.encoder, config))
    if isinstance(component, PersonPoseFeaturesModule):
        people = component.people
        if change == 'weights':
            checkpoint = tmp_path / 'other-pose.ckpt'
            checkpoint.write_bytes(b'different pose checkpoint')
            people = replace(people, vitpose_checkpoint=checkpoint)
        else:
            people = replace(people, runtime=replace(people.runtime, vitpose=replace(people.runtime.vitpose, precision='fp16')))
        return PersonPoseFeaturesModule(people, component.config)
    spec = component.encoder
    if change == 'weights':
        checkpoint = tmp_path / 'other-clip.ckpt'
        checkpoint.write_bytes(b'different CLIP checkpoint')
        spec = replace(spec, weights={'checkpoint': checkpoint})
    else:
        spec = replace(spec, dtype='float16')
    return PersonAppearanceFeaturesModule(spec, component.config)


@pytest.mark.parametrize('stage', ['person_pose_features', 'person_appearance_features'])
@pytest.mark.parametrize('change', ['match', 'weights', 'precision', 'unknown', 'preprocessing'])
def test_import_and_load_refuse_unverified_feature_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, change: str) -> None:
    case = harness(tmp_path, monkeypatch, 'fp32')
    name = f'{stage}/cam0'
    initial = PlanExecutor(ExecutionPlan([ClipExecution('clip', case.nodes, case.store, targets=[name])]))
    initial.run()
    output = initial.output(('clip', name))
    original = next(node for node in case.nodes if node.name == name)
    component = changed_component(original.component, change, tmp_path)
    if change == 'unknown':
        output = replace(output, **{'model_identity' if stage == 'person_pose_features' else 'encoder_identity': '0' * 64})
    target = replace(original, component=component, source='load', settings=component_override_identity(case.cfg, component))
    nodes = tuple(target if node.name == name else node for node in case.nodes)
    bound = bind_import(nodes, name, case.store)
    before = case.store.active(name)
    factories = (len(case.state.factories), case.state.encoder_factories)
    if change == 'match':
        publish_import(bound, case.store, output, importer='fixed-features', version=1,
                       identity={'input_sha256': 'c' * 64}, provenance={})
        plan = ExecutionPlan([ClipExecution('clip', nodes, case.store, targets=[name])])
        PlanExecutor(plan).run()
        assert plan.statuses[('clip', name)] == 'loaded'
    else:
        with pytest.raises(ValueError, match='feature model/preprocessing identity'):
            publish_import(bound, case.store, output, importer='fixed-features', version=1,
                           identity={'input_sha256': 'c' * 64}, provenance={})
        assert case.store.active(name) == before
        # Simulate an older external writer that did not enforce publish_import.
        case.store.publish(name, output, ArtifactCodec(type(output)), schema=target.io.output_schema,
            version=target.io.version, identity={'legacy_external': True}, dependencies=bound.references,
            provenance={'origin': 'import'})
        plan = ExecutionPlan([ClipExecution('clip', nodes, case.store, targets=[name])])
        plan.references.update({('clip', producer): case.store.active(producer) for producer in target.bindings.values()})
        with pytest.raises(ValueError, match='feature model/preprocessing identity'):
            plan.decide(('clip', name))
        assert ('clip', name) not in plan.references
    assert (len(case.state.factories), case.state.encoder_factories) == factories


@pytest.mark.parametrize('stage', ['person_pose_features', 'person_appearance_features'])
def test_override_crop_change_invalidates_cache_load_and_completion_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    case = harness(tmp_path, monkeypatch, 'fp32')
    name = f'{stage}/cam0'
    component = next(node.component for node in case.nodes if node.name == name)
    overrides = {**detector_overrides(case), name: component}
    nodes = standard_definition(case.cfg, case.source, code_identity='crop-override', overrides=overrides)
    initial = PlanExecutor(ExecutionPlan([ClipExecution('clip', nodes, case.store, targets=[name])]))
    initial.run()
    before = initial.output(('clip', name))
    marker = TennisSceneOrchestrator(case.cfg, components=overrides).publication_identity()
    if stage == 'person_pose_features':
        np.testing.assert_allclose(case.state.boxes[-1][:, 2], 38.4, rtol=0, atol=1e-5)
    else:
        assert before.appearance_valid.all()
    overrides[name] = changed_component(component, 'preprocessing', tmp_path)
    updated = standard_definition(case.cfg, case.source, code_identity='crop-override', overrides=overrides)
    assert TennisSceneOrchestrator(case.cfg, components=overrides).publication_identity() != marker
    load = tuple(replace(node, source='load') if node.name == name else node for node in updated)
    with pytest.raises(ValueError, match='feature model/preprocessing identity'):
        PlanExecutor(ExecutionPlan([ClipExecution('clip', load, case.store, targets=[name])])).run()
    plan = ExecutionPlan([ClipExecution('clip', updated, case.store, targets=[name])])
    executor = PlanExecutor(plan)
    executor.run()
    assert plan.statuses[('clip', name)] == 'executed'
    if stage == 'person_pose_features':
        np.testing.assert_allclose(case.state.boxes[-1][:, 2], 64., rtol=0, atol=1e-5)
    else:
        assert not executor.output(('clip', name)).appearance_valid.any()
    tracking = ExecutionPlan([ClipExecution('clip', updated, case.store, targets=['person_tracking/cam0'])])
    PlanExecutor(tracking).run()
    assert tracking.statuses[('clip', 'person_tracking/cam0')] == 'executed'


def test_pose_precision_change_reuses_clip_and_appearance_target_needs_no_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = harness(tmp_path, monkeypatch, 'fp32')
    appearances = [f'person_appearance_features/{camera}' for camera in case.source.camera_ids]
    seed = ExecutionPlan([ClipExecution('clip', case.nodes, case.store, targets=appearances)])
    PlanExecutor(seed).run()
    assert not case.state.factories and not case.state.events
    assert not any(key[1].startswith('person_pose_features/') for key in seed.required)
    assert case.state.encoder_factories == 1
    references = {name: case.store.active(name) for name in appearances}
    cfg = runtime(tmp_path, overrides=('device=cuda', 'gvhmr.enabled=false', 'people_models.runtime.vitpose.precision=fp16'))
    case.state.expected_precision = 'fp16'
    nodes = standard_definition(cfg, case.source, code_identity='feature-runtime', overrides=detector_overrides(case))
    for source, status in [('execute', 'cached'), ('load', 'loaded')]:
        current = tuple(replace(node, source=source) if node.name in appearances else node for node in nodes)
        plan = ExecutionPlan([ClipExecution('clip', current, case.store,
                                          targets=[f'person_tracking/{camera}' for camera in case.source.camera_ids])])
        PlanExecutor(plan).run()
        for name in appearances:
            assert plan.statuses[('clip', name)] == status
            assert case.store.active(name) == references[name]
        assert len(case.state.factories) == 1 and case.state.encoder_factories == 1
    assert case.state.factories[0]['precision'] == 'fp16'
