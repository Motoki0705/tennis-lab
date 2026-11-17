"""One fixed C1 clip/camera: pre-split inference versus persisted feature passes.

Run only through the shared training queue. This is an acceptance driver, not
an alternate production pipeline. The C1 recorder and legacy implementations
are loaded from pinned git objects and archived alongside the output.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.tasks.person_tracking.archive import save_features
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesModule, PersonAppearanceFeaturesModule,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ComponentNode, ExecutionPlan
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    PersonDetectionInputAssembler, PersonPoseFeaturesInputAssembler,
    PersonAppearanceFeaturesInputAssembler, PersonTrackingInputAssembler,
)
from src.tennis_scene.pipeline.model_resources import appearance_model
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.video import probe_video_info

LEGACY = 'f7b016700'
RECORDER = 'a1a05d0e0ea7f44da7ed7b29eee3450b1319338b'
# Fixed before inference. Coordinates are source pixels; peaks/embeddings unitless.
TOLERANCE = {'pose_xy_atol_px': .05, 'pose_confidence_atol': 1e-4,
             'embedding_atol': 1e-5, 'embedding_cosine_distance': 1e-6, 'rtol': 0.}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned_module(name, revision, path, output):
    content = subprocess.check_output(['git', 'show', f'{revision}:{path}'])
    archived = output / f'{name}.py'
    archived.write_bytes(content)
    spec = importlib.util.spec_from_file_location(name, archived)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_output(name, value, output):
    directory = output / name
    directory.mkdir()
    payload, arrays = ArtifactCodec(type(value)).dump(value, directory)
    (directory / 'artifact.json').write_text(json.dumps({'payload': payload, 'arrays': arrays}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.input_root.resolve()
    manifest = json.loads((root.parent / 'input-manifest.json').read_text())
    selected = [item for item in manifest['files'] if 'clip_037' in item['path']
                and ('cam0.mp4' in item['path'] or item['path'].endswith('clip.json'))]
    if len(selected) != 2 or any(sha(root / item['path']) != item['sha256'] for item in selected):
        raise ValueError('C1 input hash mismatch')
    path = root / 'videos/video_000/clips/clip_037/media/cam0.mp4'
    info = probe_video_info(path)
    if (info.frame_count, info.width, info.height) != (292, 1920, 1080):
        raise ValueError('Fixed source timeline changed')
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('src/tennis_scene/configs').resolve())):
        configured = compose(config_name='pipeline')
    cfg = PipelineRuntimeConfig.from_config(configured, bind_inputs=False)
    (output / 'plan.json').write_text(json.dumps({'head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'legacy': LEGACY, 'recorder': RECORDER, 'inputs': selected, 'tolerance': TOLERANCE,
        'scope': 'video_000/clip_037 cam0 full 292 frames; not full scene or all cameras',
        'vitpose_batch_size': cfg.people.runtime.vitpose.batch_size,
        'appearance_batch_size': cfg.tracking.features.appearance_batch_size, 'dtype': 'float32',
        'flip_test': cfg.people.runtime.vitpose.flip_test, 'allocator_limit_gib': 10}, indent=2))
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(10 * 2**30 / total, device=0)
    video = SourceVideo('cam0', path, sha(path), 292, info.fps, info.width, info.height)
    source = ClipSource('video_000/clip_037', (video,))
    context = AssemblyContext(source, 'cam0')
    recorder = pinned_module('c1_recorder', RECORDER, 'src/utils/inference_recording.py', output)
    legacy_features = pinned_module('legacy_features', LEGACY, 'src/tasks/person_tracking/features.py', output)
    legacy_tracking = pinned_module('legacy_tracking', LEGACY, 'src/tennis_scene/pipeline/components/person_tracking.py', output)
    captured = []
    class Capture(legacy_features.FeatureExtractor):
        def extract(self, *a, **kw):
            result = super().extract(*a, **kw)
            captured.append(result)
            return result
    legacy_tracking.FeatureExtractor = Capture
    encoder = appearance_model(cfg.tracking.encoder, cfg.tracking_encoder_weights, cfg.device,
        lambda: build_encoder(cfg.tracking.encoder, checkpoint_root=cfg.roots.checkpoint_root,
                              external_root=cfg.roots.external_asset_root, device=cfg.device))
    hook = recorder.model_event
    with recorder.InferenceRecorder(output / 'recording.json', devices=('cuda:0',)):
        # Freeze detector outputs once; both feature variants consume identical rows.
        detector = PersonDetectionModule(cfg.people, merge_duplicates=cfg.merge_duplicate_person_boxes)
        detector_node = ComponentNode('person_detection/cam0', detector, detector.io, PersonDetectionInputAssembler(), {}, context, {}, 'acceptance')
        detection_plan = ExecutionPlan([ClipExecution(source.clip_id, (detector_node,), ClipStore(output / 'detection-store', {}))])
        detection_executor = PlanExecutor(detection_plan, model_event=hook)
        detection_executor.run()
        detections = detection_executor.output((source.clip_id, detector_node.name))
        if not len(detections.confidence):
            raise ValueError('Fixed clip produced no detections; cannot establish inference acceptance')
        with recorder.recording_context(variant='pre_split'):
            old_tracker = legacy_tracking.PersonTrackingModule(cfg.tracking, people=cfg.people, encoder=encoder,
                aflink_checkpoint=cfg.aflink_checkpoint, show_progress=False)
            from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
            with InferenceRuntime({('legacy','tracking'): old_tracker.model_specs()}, model_event=hook) as runtime:
                with runtime.node(('legacy','tracking')) as models:
                    old_tracks = old_tracker.process_with_models(legacy_tracking.PersonTrackingInput(video,detections), models)
            save_features(output / 'pre-split-features.npz', captured, {'source': video.sha256, 'revision': LEGACY})
            save_output('pre-split-tracks', old_tracks, output)
        gc.collect()
        torch.cuda.empty_cache()
        with recorder.recording_context(variant='split'):
            fixed = SimpleNamespace(process=lambda _: detections)
            pose = PersonPoseFeaturesModule(cfg.people, cfg.tracking.features)
            appearance = PersonAppearanceFeaturesModule(encoder, cfg.tracking.features)
            tracker = PersonTrackingModule(cfg.tracking, aflink_checkpoint=cfg.aflink_checkpoint, show_progress=False)
            nodes = (
                ComponentNode('person_detection/cam0',fixed,detector.io,PersonDetectionInputAssembler(),{},context,{},'acceptance'),
                ComponentNode('person_pose_features/cam0',pose,pose.io,PersonPoseFeaturesInputAssembler(),
                    {'detections':'person_detection/cam0'},context,{},'acceptance'),
                ComponentNode('person_appearance_features/cam0',appearance,appearance.io,PersonAppearanceFeaturesInputAssembler(),
                    {'detections':'person_detection/cam0','poses':'person_pose_features/cam0'},context,{},'acceptance'),
                ComponentNode('person_tracking/cam0',tracker,tracker.io,PersonTrackingInputAssembler(),
                    {'detections':'person_detection/cam0','poses':'person_pose_features/cam0','appearance':'person_appearance_features/cam0'},context,{},'acceptance'),
            )
            plan = ExecutionPlan([ClipExecution(source.clip_id,nodes,ClipStore(output/'split-store',{}))])
            executor = PlanExecutor(plan,model_event=hook)
            executor.run()
            poses = executor.output((source.clip_id,'person_pose_features/cam0'))
            appearances = executor.output((source.clip_id,'person_appearance_features/cam0'))
            new_tracks = executor.output((source.clip_id,'person_tracking/cam0'))
    before_poses = np.concatenate([f.poses for f in captured])
    before_embeddings = np.concatenate([f.embeddings for f in captured])
    before_valid = np.concatenate([f.appearance_valid for f in captured])
    differences = {'pose_xy_max_abs_px':float(np.abs(poses.keypoints[...,:2]-before_poses[...,:2]).max()),
        'pose_confidence_max_abs':float(np.abs(poses.keypoints[...,2]-before_poses[...,2]).max()),
        'embedding_max_abs':float(np.abs(appearances.embeddings-before_embeddings).max()),
        'embedding_cosine_distance_max':float(np.maximum(0,1-(appearances.embeddings[before_valid]*before_embeddings[before_valid]).sum(1)).max()),
        'appearance_mask_equal':bool(np.array_equal(appearances.appearance_valid,before_valid)),
        'track_ids_equal':bool(np.array_equal(new_tracks.track_ids,old_tracks.track_ids)),
        'track_rows_equal':bool(np.array_equal(new_tracks.evidence.detection_rows,old_tracks.evidence.detection_rows)),
        'track_observed_equal':bool(np.array_equal(new_tracks.observed,old_tracks.observed)),
        'gsi_mask_equal':bool(np.array_equal(new_tracks.reconstruction.interpolated,old_tracks.reconstruction.interpolated)),
        'detection_rows':len(detections.confidence)}
    differences['features_within_tolerance'] = all((
        differences['pose_xy_max_abs_px']<=TOLERANCE['pose_xy_atol_px'],
        differences['pose_confidence_max_abs']<=TOLERANCE['pose_confidence_atol'],
        differences['embedding_max_abs']<=TOLERANCE['embedding_atol'],
        differences['embedding_cosine_distance_max']<=TOLERANCE['embedding_cosine_distance'],
        differences['appearance_mask_equal']))
    recorded = json.loads((output/'recording.json').read_text())
    (output/'comparison.json').write_text(json.dumps(differences,indent=2)+'\n')
    (output/'resources.json').write_text(json.dumps(recorded['resources'],indent=2)+'\n')
    events = [event for event in recorded['events'] if event['stage'] in {'model_load','model_unload'}]
    (output/'model-events.json').write_text(json.dumps(events,indent=2)+'\n')
    print(json.dumps(differences,indent=2),flush=True)
    if any(sha(root/item['path'])!=item['sha256'] for item in selected):
        raise ValueError('Fixed input changed during inference')
    if not differences['features_within_tolerance']:
        raise ValueError('Feature differences exceeded predeclared tolerance')


if __name__=='__main__':
    main()
