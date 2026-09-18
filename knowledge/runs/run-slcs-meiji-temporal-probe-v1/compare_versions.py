"""Compare frozen teacher versions; masks are not measured 3D accuracy."""
from pathlib import Path
import json
import numpy as np
from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest, ClipManifest
from src.tasks.slcs.data.annotation import load_slcs_annotation, has_slcs_annotation
from src.tasks.slcs.data.quality import QualityConfig, build_label_masks
from src.tennis_scene.dataset_pipeline.quality_report import longest_gap
from src.tennis_scene.reference_pipeline.observations import sha256

clips = ['video_000/clip_007','video_001/clip_001','video_000/clip_001','video_000/clip_009','video_001/clip_000']
results, pending = [], []
quality = QualityConfig(.3, 1, 1., .5)
for key in clips:
    roots = {ver: Path(f'data/slcs/meiji_rgb_{ver}') for ver in ['v6','v7']}
    manifests = {ver: ClipManifest.load(root/load_dataset_manifest(root).clips[key].path) for ver,root in roots.items()}
    if not has_slcs_annotation(manifests['v7'].clip_dir):
        pending.append(key)
        continue
    scenes = {ver: load_slcs_annotation(clip) for ver,clip in manifests.items()}
    versions = {}
    for ver, scene in scenes.items():
        q = scene.metadata['label_quality']
        pw, bw = np.asarray(q['player_weight']), np.asarray(q['ball_weight'])
        masks = build_label_masks(human_kp_vis=scene.human_kp_vis,ball_vis=scene.ball_vis,player_position=scene.player_position,player_yaw=scene.player_yaw,ball_3d=scene.ball_3d,config=quality,teacher_quality=q)
        pm = masks['player_label_weight'] > 0
        versions[ver] = {'player_teacher_fraction':(pw>0).mean(axis=1).tolist(), 'player_slcs_frame_fraction':pm.mean(axis=1).tolist(), 'longest_unsupported_slcs_player_gap':[longest_gap(x) for x in pm], 'ball_teacher_fraction':float((bw>0).mean()), 'scene_sha256':sha256(manifests[ver].clip_dir/'annotations/tennis_scene/scene.npz')}
    a,b = scenes['v6'],scenes['v7']
    controls = {'ball_max_absolute_difference_m':float(np.max(np.abs(a.ball_3d-b.ball_3d))),'ball_uv_identical':bool(np.array_equal(a.ball_uv,b.ball_uv)),'court_identical':bool(np.array_equal(a.court_kp,b.court_kp)),'teacher_checkpoints_identical':a.metadata['dataset_producer_identity']['checkpoints']==b.metadata['dataset_producer_identity']['checkpoints']}
    changed_pose_producers = []
    for cam in manifests['v7'].camera_ids:
        receipts = [json.loads((Path('outputs/tennis_scene/precompute/meiji_dino_vitpose')/run/key/f'{cam}_people.metadata.json').read_text()) for run in ['s42-003','s42-004']]
        if receipts[0]['pose_sha256'] != receipts[1]['pose_sha256']:
            changed_pose_producers.append({'camera':cam,'old_sha256':receipts[0]['pose_sha256'],'new_sha256':receipts[1]['pose_sha256']})
    entry = {'clip_id':key,'versions':versions,'controls':controls,'changed_pose_checkpoint_receipts':changed_pose_producers}
    if key == 'video_000/clip_007':
        entry['p0_supported_frames_571_through_651'] = {v:int(np.sum(np.asarray(s.metadata['label_quality']['player_weight'])[0,571:652] > 0)) for v,s in scenes.items()}
    results.append(entry)
    print(json.dumps(entry),flush=True)
out = Path(__file__).with_name('comparison.json')
out.write_text(json.dumps({'status':'incomplete' if pending else 'complete','pending_clips':pending,'scope':'positive teacher and final SLCS frame weights; no independent measured 3D accuracy; a changed producer receipt confounds attribution to association alone','quality':vars(quality),'results':results},indent=2)+'\n')
print(out)
