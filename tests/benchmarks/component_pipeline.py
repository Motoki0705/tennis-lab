"""Full clip qualification with confirmed ball, side and person assignments.

Run GPU execution through the shared training queue. External annotations never
enter the ball predictor. Model Re-ID embeddings are evaluated and saved first;
the separately confirmed historical assignments are imported for downstream load.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.components.camera_geometry import (
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.imports.ball_annotations import import_ball_annotations
from src.tennis_scene.pipeline.imports.court_side import import_confirmed_side
from src.tennis_scene.pipeline.imports.person_association import (
    import_confirmed_person_association,
)
from src.tennis_scene.pipeline.input_assembly.observations import gather_balls
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import structured_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.utilts.timeline import association_frame_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    repo, clip, report = args.repo.resolve(), args.clip.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    source = structured_clip_source(clip)
    overrides = [f'paths.project_root={repo}', f'paths.data_root={repo / "data"}',
        f'paths.checkpoint_root={repo / "ckpt"}', f'paths.external_asset_root={repo / "third_party"}',
        f'paths.artifact_root={repo / "outputs"}', f'paths.output_root={repo / "outputs"}',
        f'device={args.device}', "court_kp.checkpoint='court_detection/multiscale_depth3/b863df1f01f0.ckpt'",
        'court_kp.region_search.enabled=true', 'people_models.dino_checkpoint=dino/checkpoint0029_4scale_swin.pth',
        'plcs_reid.checkpoint=plcs/player-reid-headless-v2-s42.ckpt',
        'execution.ball_detection=load', 'execution.court_side=load', 'execution.person_reid=load',
        'camera_geometry.reference_camera=cam0',
        'person_observations.sideline_margin_m=1.0', 'person_observations.baseline_margin_m=10.0']
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('src/tennis_scene/configs').resolve())):
        config = compose(config_name='pipeline', overrides=overrides)
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    (report / 'pipeline_config.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
    application = TennisSceneOrchestrator(runtime)
    store = ClipStore(clip / 'annotations/tennis_scene', json_value(source))
    imported = import_ball_annotations(source, clip / 'outsource', store)
    nodes = standard_definition(runtime, source, code_identity=application.code_identity)
    preparation_nodes = tuple(replace(node, source='execute') if node.name == 'person_reid' else node for node in nodes)
    runner = ComponentRunner(preparation_nodes, store)
    receipt: dict[str, Any] = {'source': json_value(source), 'ball_imports': json_value(imported), 'status': 'running'}
    try:
        # Preserve model Re-ID embeddings and inferred IDs as an immutable artifact
        # before the independently confirmed association takes the active slot.
        runner.run(targets=('person_reid',))
        receipt['preparation_status'] = dict(runner.statuses)
        receipt['preparation_seconds'] = dict(runner.seconds)
        model_reid_reference = runner.references['person_reid']
        receipt['model_reid_artifact'] = json_value(model_reid_reference)
        calibration: CourtCalibrationOutput = runner.output('court_calibration')
        ball_artifacts = {f'ball_{v.camera_id}': store.load(imported[f'ball_detection/{v.camera_id}'], ArtifactCodec(BallDetectionOutput)) for v in source.videos}
        raw = gather_balls(source, ball_artifacts).select_views(tuple(v.source_index for v in calibration.calibration.views))
        grouped = single_ball_observations(raw, threshold=runtime.ball_detection.score_threshold)
        sample = association_frame_indices(source.num_frames, source.fps, max_frames=runtime.inference_policy.max_frames)
        scale = float(np.hypot(*source.size) / np.hypot(1920, 1080))
        evidence = SideEvidence('ball', grouped.uv_px[:, :, sample], grouped.visibility[:, :, sample], runtime.ball_reprojection_px * scale)
        ids = calibration.calibration.camera_ids
        reference_index = ids.index(calibration.reference_camera)
        choices = [(False,) if i == reference_index else (False, True) for i in range(len(ids))]
        hypotheses = tuple(np.asarray(turns, bool) for turns in product(*choices))
        confirmed = resolve_camera_geometry(calibration.calibration, calibration.reference_camera, hypotheses, (evidence,), config=runtime.camera_geometry)
        confirmation = {'method': 'all_half_turn_hypotheses_against_current_court_and_external_observed_ball',
            'reference_camera': calibration.reference_camera, 'candidates': confirmed.document['side_candidates'],
            'independent_model_accuracy_evaluation': False}
        side_node = next(n for n in nodes if n.name == 'court_side')
        side_ref = import_confirmed_side(side_node, store, confirmed.view_half_turns, confirmation=confirmation)
        receipt['side_confirmation'] = confirmation
        receipt['side_artifact'] = json_value(side_ref)
        write_json_atomic(report / 'side_confirmation.json', confirmation)
        reid_node = next(n for n in nodes if n.name == 'person_reid')
        person_ref, person_confirmation = import_confirmed_person_association(reid_node, store, source,
            model_reference=model_reid_reference,
            historical_association=clip / 'annotations/player_association_result.json',
            legacy_gvhmr_directory=clip / 'annotations')
        receipt['person_confirmation'] = person_confirmation
        receipt['confirmed_person_artifact'] = json_value(person_ref)
        write_json_atomic(report / 'person_confirmation.json', person_confirmation)
        runner = ComponentRunner(nodes, store)
        runner.run()
        scene = runner.output('scene_assembly')
        application._export(scene, runner, store)
        restored = load_scene_result(store.index_path)
        np.testing.assert_array_equal(scene.ball_3d, restored.ball_3d)
        np.testing.assert_array_equal(scene.player_valid, restored.player_valid)
        receipt.update(status=scene.metadata['status'], stage_status=runner.statuses, stage_seconds=runner.seconds,
            artifacts=json_value(runner.references), validity=scene.metadata['validity_statistics'],
            camera_ids=source.camera_ids, frame_count=source.num_frames,
            confirmed_half_turns=confirmed.view_half_turns,
            ball_observed={k: int(v.observed.sum()) for k, v in ball_artifacts.items()},
            player_ids=scene.player_track_ids.tolist(), export=str(store.index_path))
        # Load-only resume must never call any process implementation.
        reload_runner = ComponentRunner([replace(n, source='load') for n in nodes], ClipStore(store.root, json_value(source), memory_entries=0))
        reload_runner.run()
        np.testing.assert_array_equal(reload_runner.output('scene_assembly').ball_3d, scene.ball_3d)
        receipt['load_only_resume'] = reload_runner.statuses
        if runner.statuses['court_side'] != 'loaded' or runner.statuses['person_reid'] != 'loaded' or any(runner.statuses[f'ball_detection/{v.camera_id}'] != 'loaded' for v in source.videos):
            raise AssertionError('Validation must load imported ball, side and confirmed person assignments')
    except Exception as error:
        receipt.update(status='failed', error=str(error), error_type=type(error).__name__,
            active_stage=runner.active_node, stage_status=runner.statuses, stage_seconds=runner.seconds,
            artifacts=json_value(runner.references))
        raise
    finally:
        write_json_atomic(report / 'evaluation.json', receipt)
    print(json.dumps({'status': receipt['status'], 'validity': receipt['validity'], 'scene_index': receipt['export']}))


if __name__ == '__main__':
    main()
