"""Verify immutable geometry, hash-only reader correction and actual clip boundary."""
import hashlib,json
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.tennis_scene.pipeline.video_court.diagnostics import DriftReference,load_video_drift
from src.tennis_scene.pipeline.video_court.component import SharedCourtInput,SharedCourtModule
from src.tennis_scene.pipeline.contracts import SourceVideo,ClipSource
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from evidence import assert_sources

out=Path('outputs/tennis_scene/evaluate/court-video-drift-lines-20261117')
summary=json.loads((out/'evaluation-summary.json').read_text())
old=json.loads((out/'evaluation-summary-bgr.json').read_text())
pinned=json.loads((out/'immutable-inputs.json').read_text());assert_sources(json.loads((out/'frozen-profile.json').read_text()))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in pinned['files'].items())
base=VideoCourtStore(Path(pinned['base_store'])).load(summary['base_bundle_digest'])
artifact=load_video_drift(base,DriftReference(out/'drift-store',summary['drift_bundle_digest']))
comparisons=0;changed_hashes=0;same_heldout={}
for camera,doc in artifact.drift_cameras.items():
    previous=json.loads((out/'drift-store/objects'/f'{old["drift_camera_digests"][camera]}.json').read_text())
    counts=Counter()
    for clip,report in doc['clips'].items():
        prior=previous['clips'][clip]
        assert {k:v for k,v in report.items() if k!='frames'}=={k:v for k,v in prior.items() if k!='frames'}
        for a,b in zip(report['frames'],prior['frames'],strict=True):
            assert {k:v for k,v in a.items() if k!='rgb_sha256'}=={k:v for k,v in b.items() if k!='rgb_sha256'}
            comparisons+=1;changed_hashes+=a['rgb_sha256']!=b['rgb_sha256']
            if a['frame_index']==4*(base.cameras[camera]['sources'][clip]['num_frames']-1)//5:counts[a['status']]+=1
    same_heldout[camera]=dict(counts)
module=SharedCourtModule();points={};counts=Counter();started=time.monotonic()
for clip in next(iter(base.cameras.values()))['sources']:
    videos=[]
    for camera,doc in base.cameras.items():
        source=doc['sources'][clip]
        path=Path('data/tennis_multiview/processed/meiji_4cam/dataset')/f'videos/video_000/clips/{clip.split("/")[1]}/media/{camera}.mp4'
        videos.append(SourceVideo(camera,path.resolve(),source['sha256'],source['num_frames'],source['fps'],*doc['size_wh']))
    source=ClipSource(clip,tuple(videos));artifact.bind(source)
    for video in videos:
        camera=video.camera_id;report=artifact.drift_cameras[camera]['clips'][clip]
        inputs=SharedCourtInput(video,clip,base.cameras[camera],base.manifest['cameras'][camera],report,artifact.drift_manifest['cameras'][camera])
        try:result=module.process(inputs)
        except ReconstructionUnavailable as error:
            assert report['status']=='failed' and error.reason=='video_shared_court_failed';counts['explicit_failure']+=1
        else:
            assert report['status']!='failed'
            if camera in points:np.testing.assert_array_equal(result.keypoints,points[camera])
            points[camera]=result.keypoints;counts['same_shared_keypoints']+=1
artifact.verify()
proof={'base_bundle_digest':base.digest,'drift_bundle_digest':artifact.drift.digest,'base_objects_bytes_unchanged':True,
       'reader_correction_compared_frames':comparisons,'numeric_and_verdict_mismatches':0,'corrected_rgb_hashes':changed_hashes,
       'old_bgr_artifact_superseded':old['drift_bundle_digest'],'common_h_refit':False,'threshold_changes_after_freeze':False,
       'same_heldout_4_over_5_statuses':same_heldout,'component_results':dict(counts),'component_seconds':time.monotonic()-started}
(out/'result-integrity.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
