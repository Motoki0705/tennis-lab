"""Reassess immutable run-78 measurements under the human-approved policy.

No RGB decode, stripe extraction or common-H fit is performed. Review references
existing native sources/crops and displays current policy on its canvas page.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from hydra import compose, initialize_config_dir

ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.video_court.component import SharedCourtInput, SharedCourtModule
from src.tennis_scene.pipeline.video_court.diagnostics import clip_report
from src.tennis_scene.pipeline.video_court.policy import apply_drift_policy
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.artifacts import document_digest
from evidence import assert_sources


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic()
    previous=args.previous.resolve();output=args.output.resolve()
    if previous==output:raise ValueError('Keep original measurement evidence separate')
    output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((previous/'evaluation-summary.json').read_text())
    pinned=json.loads((previous/'immutable-inputs.json').read_text())
    assert_sources(json.loads((previous/'frozen-profile.json').read_text()))
    assert all(sha(Path(p))==digest for p,digest in pinned['files'].items())
    drift_paths={p:sha(p) for p in (previous/'drift-store/objects').glob('*.json')}
    artifact_root=(ROOT/'outputs').resolve()
    base_directory=Path(pinned['base_store']).resolve().relative_to(artifact_root)
    drift_directory=(previous/'drift-store').relative_to(artifact_root)
    overrides=['device=cpu','court_kp.mode=video_shared','court_kp.video_shared.action=load',
        f'paths.artifact_root={artifact_root}',f'court_kp.video_shared.directory={base_directory}',
        'court_kp.video_shared.dataset_id=meiji_4cam','court_kp.video_shared.video_id=video_000',
        f'court_kp.video_shared.artifact_digest={summary["base_bundle_digest"]}',
        f'court_kp.video_shared.drift_directory={drift_directory}',
        f'court_kp.video_shared.drift_digest={summary["drift_bundle_digest"]}']
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'src/tennis_scene/configs')):
        cfg=compose(config_name='pipeline',overrides=overrides)
    runtime=PipelineRuntimeConfig.from_config(cfg,bind_inputs=False)
    policy=runtime.video_court_unobservable_policy
    assert policy=='warn'
    artifact=TennisSceneOrchestrator.from_runtime_config(runtime).prepare_video_court()
    assert artifact is not None
    human={'date':'2026-11-17','common_h_reviewed':True,'accepted_for_downstream':True,
        'source':'Parent instruction to worker #2191, turn c4c7bed22d114a2da39b7a77252fca41',
        'quote':'緑の線は完全にコートを示している。これを下流で用いていけばよい',
        'scope':{'clips':3,'cameras':4,'native_resolution_crops':True,'clip_ids':'not specified in parent instruction',
                 'all_69_clip_images_reviewed':False},
        'rationale':['共通Hが承認済み参照と一致した','人間が共通Hを目視確認した',
                     '検査不成立の主因は隣接線の誤抽出であり、10px級の整合した動きの証拠はない'],
        'limitation':'clipの途中でcameraが動いた場合、自動では検出されないことがある'}
    save(output/'human-review.json',human)
    effective={};clips=[];frames=[];camera_summary={};component_counts=Counter();strict_counts=Counter();points={}
    for camera,doc in artifact.cameras.items():
        effective[camera]={}
        for clip,source in doc['sources'].items():
            measured=clip_report(artifact,camera,clip)
            adopted=apply_drift_policy(measured,policy,common_status=doc['status'])
            strict=apply_drift_policy(measured,'fail',common_status=doc['status'])
            strict_counts[strict['status']]+=1
            effective[camera][clip]=adopted
            for before,after in zip(measured['frames'],adopted['frames'],strict=True):
                assert {k:v for k,v in after.items() if k not in {'status','measurement_status'}} == {k:v for k,v in before.items() if k!='status'}
                frames.append({'camera':camera,'clip_id':clip,'frame_index':after['frame_index'],
                    'measurement_status':before['status'],'status':after['status'],'reason':after['reason'],
                    'geometry_reasons':'|'.join(after.get('geometry_reasons',[])),
                    'mean_px':after.get('mean_displacement_px'),'max_px':after.get('max_displacement_px'),
                    'max_projection_std_px':after.get('max_visible_projection_std_px'),'unobservable_policy':policy})
            warned=[f for f in adopted['frames'] if f['status']=='warning']
            clips.append({'camera':camera,'clip_id':clip,'status':adopted['status'],
                'measurement_status':measured['status'],'reason':adopted['reason'],'worst_frame_index':adopted['worst_frame_index'],
                'warning_frames':'|'.join(str(f['frame_index']) for f in warned),
                'geometry_reasons':'|'.join(sorted({r for f in warned for r in f.get('geometry_reasons',[])})),
                'unobservable_policy':policy,'base_camera_digest':artifact.manifest['cameras'][camera]})
            video=SourceVideo(camera,(ROOT/'data/tennis_multiview/processed/meiji_4cam/dataset'/f'videos/video_000/clips/{clip.split("/")[1]}/media/{camera}.mp4').resolve(),
                source['sha256'],source['num_frames'],source['fps'],*doc['size_wh'])
            inputs=SharedCourtInput(video,clip,doc,artifact.manifest['cameras'][camera],measured,summary['drift_camera_digests'][camera])
            result=SharedCourtModule(policy).process(inputs)
            assert result.diagnostics['drift']==adopted
            if camera in points:np.testing.assert_array_equal(points[camera],result.keypoints)
            else:points[camera]=result.keypoints.copy()
            component_counts['shared_output']+=1
            try:old=SharedCourtModule('fail').process(inputs)
            except ReconstructionUnavailable:
                assert strict['status']=='failed'
            else:np.testing.assert_array_equal(old.keypoints,result.keypoints)
        selected=[c for c in clips if c['camera']==camera]
        sampled=[f for f in frames if f['camera']==camera]
        camera_summary[camera]={'clips':len(selected),'usable':sum(c['status']!='failed' for c in selected),
            'ok':sum(c['status']=='ok' for c in selected),'warning':sum(c['status']=='warning' for c in selected),
            'failed':sum(c['status']=='failed' for c in selected),'frames':dict(Counter(f['status'] for f in sampled)),
            'keypoint_bytes_sha256':hashlib.sha256(points[camera].tobytes()).hexdigest()}
    for name,rows in [('clips',clips),('frames',frames)]:
        with (output/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    four=sum(all(effective[c][clip]['status']!='failed' for c in effective) for clip in next(iter(effective.values())))
    artifact.verify()
    assert all(sha(Path(p))==digest for p,digest in pinned['files'].items())
    assert all(sha(p)==digest for p,digest in drift_paths.items())
    final={'schema':'video_court_policy_reassessment_v1','policy':policy,
        'common_h_bundle':artifact.digest,'common_h_cameras':artifact.manifest['cameras'],
        'drift_bundle':summary['drift_bundle_digest'],'drift_camera_digests':summary['drift_camera_digests'],
        'base_artifact_bytes_unchanged':True,'drift_artifact_bytes_unchanged':True,'measurement_values_unchanged':True,
        'common_h_refit':False,'rgb_decoded':False,'cameras':camera_summary,'camera_clips':len(clips),'frames':len(frames),
        'complete_four_camera_clips':four,'strict_policy_counts':dict(strict_counts),'component_results':dict(component_counts),
        'runtime_overrides':overrides,'default_policy_from_yaml':True,'human_review':human,
        'policy_identity':document_digest({'common_h':artifact.digest,'drift':summary['drift_bundle_digest'],'unobservable_policy':policy}),
        'seconds':time.monotonic()-started}
    save(output/'summary.json',final)
    # Derive a new review from the previous gallery; native images stay in place.
    page=(previous/'review/index.html').read_text()
    marker='<script id="data" type="application/json">';start=page.index(marker)+len(marker);end=page.index('</script>',start)
    gallery=json.loads(page[start:end])
    image_prefix=os.path.relpath(previous/'review',output/'review')+'/'
    for row in gallery:
        adopted=effective[row['camera']][row['clip_id']]
        row.update(status=adopted['status'],reason=adopted['reason'],worst_frame_index=adopted['worst_frame_index'],unobservable_policy=policy)
        by_frame={f['frame_index']:f for f in adopted['frames']}
        for frame in row['frames']:
            frame['source']=image_prefix+frame['source'];frame.pop('overlay',None)
            frame['report']=by_frame.get(frame['index'])
            for paths in frame['crops'].values():
                if paths:
                    for key in paths:paths[key]=image_prefix+paths[key]
    page=page[:start]+json.dumps(gallery,ensure_ascii=False).replace('<','\\u003c')+page[end:]
    page=page.replace('人間の品質確認は未完了です。','共通Hは人間が3clip×4cameraと原寸cropで確認し、下流での利用を承認しました（2026-11-17）。')
    page=page.replace('検査不能は失敗です。未測定frameの安全を保証しません。',
        'unobservable_policy=warn：検査不能と線の矛盾は警告として共通Hを使います。信頼できるmean 10px超／max 25px超だけdrift失敗です。clipの途中でcameraが動いた場合、自動では検出されないことがあります。')
    page=page.replace("$('download').href=currentFrame.overlay;",'')
    page=page.replace('selected_frame:f','unobservable_policy:current.unobservable_policy,selected_frame:f')
    page=page.replace('<p><a id="download">選択frameの重ね描き</a></p>','')
    page=page.replace('../evaluation-summary.json','../summary.json')
    (output/'review').mkdir(exist_ok=True);(output/'review/index.html').write_text(page)
    print(json.dumps({k:v for k,v in final.items() if k not in {'runtime_overrides','human_review'}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,default=Path('outputs/tennis_scene/evaluate/court-video-drift-lines-20261117'))
    p.add_argument('--output',type=Path,default=Path('outputs/tennis_scene/evaluate/court-video-drift-lines-20261117/policy'))
    main(p.parse_args())
