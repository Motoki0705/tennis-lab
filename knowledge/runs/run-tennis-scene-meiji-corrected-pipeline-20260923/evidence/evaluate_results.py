"""Measure the two fresh scene outputs and their rendered movies."""
from pathlib import Path
from fractions import Fraction
import hashlib, json, subprocess, sys
import cv2
import numpy as np
from omegaconf import OmegaConf

root=Path(__file__).resolve().parent
requests={phase:json.loads((root/f'{phase}.json').read_text()) for phase in ('pipeline','dataset')}
clip=Path(requests['pipeline']['clip'])
manifest=json.loads((clip/'clip.json').read_text())
paths={'pipeline':Path(requests['pipeline']['output'])/'scene.npz','dataset':clip/'annotations/tennis_scene/scene.npz'}
if len(sys.argv)>1:
    phase=sys.argv[1]
    if phase not in paths:raise ValueError(phase)
    paths={phase:paths[phase]}
partial_reports={phase:json.loads((root/f'evaluation-{phase}.json').read_text()) for phase in paths if (root/f'evaluation-{phase}.json').is_file()}
if (root/'evaluation.json').is_file():
    completed_report=json.loads((root/'evaluation.json').read_text())
    for phase in paths:
        if phase in completed_report.get('scenes',{}):partial_reports[phase]=completed_report

def summarize(values):
    arr=np.asarray(values)
    finite=arr[np.isfinite(arr)]
    if finite.size==0:return {'count':0,'nonfinite':int(arr.size)}
    return {'count':int(finite.size),'nonfinite':int(arr.size-finite.size),'min':float(finite.min()),'max':float(finite.max()),'mean':float(finite.mean()),'median':float(np.median(finite)),'p95':float(np.percentile(finite,95))}

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): result.update(block)
    return result.hexdigest()

manual=json.loads((clip/'annotations/manual_court_kp_result.json').read_text())
manual_kp=np.asarray(manual['keypoints'])[:,:,:14]
manual_vis=np.asarray(manual['visibility'])[:,:,:14]>0
annotations={c:json.loads((clip/'outsource'/f'{c}_annotations.json').read_text()) for c in manifest['camera_ids']}
media_hashes={c:digest(clip/'media'/f'{c}.mp4') for c in manifest['camera_ids']}
for c in manifest['camera_ids']:
    if annotations[c]['source']['sha256']!=media_hashes[c]:raise ValueError(f'Ball annotation source bytes differ: {c}')
receipt=json.loads((root/'input_receipt.json').read_text())
for role,item in receipt.items():
    if digest(Path(item['path']))!=item['sha256']:raise ValueError(f'Input changed during validation: {role}')
report={'input_receipt':receipt,'manifest':manifest,'source_sha256':media_hashes,'scenes':{},'comparison':{},'videos':{},'limits':['No measured 3D ground truth. Court/ball annotations are reference observations, not proof of 3D accuracy.','Court uses static manual points across the full clip; repeated frames are correlated, not independent accuracy evidence.','Ball distances use observed annotation frames only and report detection coverage separately.']}
def config_differences(left,right,path=''):
    if isinstance(left,dict) and isinstance(right,dict):
        return [difference for key in sorted(set(left)|set(right)) for difference in config_differences(left.get(key),right.get(key),f'{path}.{key}'.lstrip('.'))]
    return [] if left==right else [{'setting':path,'pipeline':left,'dataset':right}]
configs={phase:OmegaConf.to_container(OmegaConf.load(root/f'{phase}.expanded_pipeline.yaml'),resolve=True) for phase in ('pipeline','dataset')}
report['configuration_comparison']={'differences':config_differences(configs['pipeline'],configs['dataset']),'note':'generate_dataset takes actual video_paths and camera_ids from clip.json; the saved base pipeline config retains its example video_paths. Actual scene metadata inputs are compared independently.'}
for phase,path in paths.items():
    metadata=json.loads(path.with_suffix('.metadata.json').read_text())
    with np.load(path,allow_pickle=False) as scene:
        inspection=[]
        entry={'path':str(path),'sha256':digest(path),'arrays':{},'court':{},'ball':{},'dynamics':{},'pose_visibility_conversion':metadata.get('pose_visibility_conversion')}
        entry['court_comparison_contract']='Camera-local KP14 from the stage JSON, before the reference-camera permutation, compared in the manual annotation channel order; no fitted permutation is chosen from GT.'
        for key in scene.files:
            a=scene[key]
            entry['arrays'][key]={'shape':list(a.shape),'dtype':str(a.dtype),'nonfinite':int((~np.isfinite(a)).sum())}
            assert np.isfinite(a).all(),(phase,key,'nonfinite')
        assert int(scene['num_frames'])==manifest['num_frames']
        assert int(scene['width'])==manifest['width'] and int(scene['height'])==manifest['height']
        assert metadata['camera_ids']==manifest['camera_ids']
        assert metadata['court_kp_frame_indices']==list(range(1010))
        assert receipt['court']['sha256'] in json.dumps(metadata['court_detection']['checkpoint'])
        entry['metadata']={'path':str(path.with_suffix('.metadata.json')),'sha256':digest(path.with_suffix('.metadata.json')),'enabled_stages':metadata['enabled_stages'],'court_reference':metadata['court_reference'],'player_association':metadata['player_association'],'gvhmr_alignment':metadata['gvhmr_alignment']}
        assert all(player['fit']['success'] for player in metadata['gvhmr_alignment']['players'])
        assert scene['player_position'].shape==(2,1010,3)
        assert scene['ball_uv'].shape==(3,1010,2)
        assert scene['human_kp_vis'].min()>=0 and scene['human_kp_vis'].max()<=1
        for key in ('player_position','gvhmr_aligned_player_position','ball_3d'):
            a=scene[key]
            speed=np.linalg.norm(np.diff(a,axis=-2),axis=-1)*float(scene['fps'])
            entry['dynamics'][key]={'position_m':{'x':summarize(a[...,0]),'y':summarize(a[...,1]),'z':summarize(a[...,2])},'speed_m_s':summarize(speed)}
            for index in np.argsort(speed.reshape(-1))[-3:][::-1]:
                axis=np.unravel_index(index,speed.shape)
                inspection.append({'frame':int(axis[-1]+1),'reason':f'{key}: largest interframe speed','value_m_s':float(speed[axis]),'player':int(axis[0]) if speed.ndim==2 else None})
            z=a[...,2]
            for mode,index in [('minimum',int(np.argmin(z))),('maximum',int(np.argmax(z)))]:
                axis=np.unravel_index(index,z.shape)
                inspection.append({'frame':int(axis[-1]),'reason':f'{key}: {mode} height','value_m':float(z[axis]),'player':int(axis[0]) if z.ndim==2 else None})
        residual=np.linalg.norm(scene['gvhmr_aligned_player_position']-scene['player_position'],axis=-1)
        entry['alignment_position_difference_m']=summarize(residual)
        entry['yaw_step_deg']=summarize(np.abs(np.rad2deg(np.angle(np.exp(1j*np.diff(scene['player_yaw'],axis=-1))))))
        entry['inspection_candidates']=inspection
        if phase=='dataset':
            marker=json.loads((path.parent/'annotation.json').read_text())
            assert marker['clip_id']=='video_000/clip_000'
            assert marker['clip_manifest_sha256']==digest(clip/'clip.json')
            for key,spec in marker['arrays'].items():
                assert list(scene[key].shape)==spec['shape'] and str(scene[key].dtype)==spec['dtype'],key
            assert (path.parent/marker['pipeline_config']).is_file()
            entry['completion_marker']=marker
        court=json.loads((Path(requests[phase]['output'])/'court_kp_result.json').read_text())
        kp=np.asarray(court['keypoints'])[:,:,:14]; vis=np.asarray(court['visibility'])[:,:,:14]>0
        for n,c in enumerate(manifest['camera_ids']):
            mask=vis[n]&manual_vis[n]
            error=np.linalg.norm((kp[n]-manual_kp[n])*[manifest['width']-1,manifest['height']-1],axis=-1)
            entry['court'][c]={'visible_points':int(mask.sum()),'distance_px':summarize(error[mask]),'first_frame_distance_px':summarize(error[0][mask[0]]),'frames_with_all_points':int(mask.all(axis=-1).sum())}
            observed=np.array([f['status']=='observed' for f in annotations[c]['frames']],bool)
            predicted=scene['ball_vis'][n].astype(bool)
            target=np.array([[f['center_px']['x'],f['center_px']['y']] if f['status']=='observed' else [0,0] for f in annotations[c]['frames']])
            uv=scene['ball_uv'][n]*[manifest['width']-1,manifest['height']-1]
            matched=observed&predicted
            entry['ball'][c]={'observed_frames':int(observed.sum()),'predicted_on_observed':int(matched.sum()),'missing_on_observed':int((observed&~predicted).sum()),'coverage':float(matched.sum()/observed.sum()),'distance_px':summarize(np.linalg.norm(uv[matched]-target[matched],axis=-1))}
        report['scenes'][phase]=entry
if len(paths)==2:
    scene_metadata={phase:json.loads(path.with_suffix('.metadata.json').read_text()) for phase,path in paths.items()}
    assert scene_metadata['pipeline']['video_paths']==scene_metadata['dataset']['video_paths']
    assert scene_metadata['pipeline']['camera_ids']==scene_metadata['dataset']['camera_ids']
    report['configuration_comparison']['actual_video_paths_equal']=True
    report['configuration_comparison']['actual_camera_ids_equal']=True
report['slcs_reader']={'accepted':None,'reason':'Dataset scene is not part of this partial evaluation.'}
if 'dataset' in paths:
    from hydra import compose, initialize_config_dir
    from src.tasks.slcs.configuration import SLCSPrecomputeConfig
    from src.tasks.slcs.data.dataset import load_clip_arrays
    from src.tennis_scene.generate_dataset.manifest import ClipManifest
    from src.utils.paths import PROJECT_ROOT
    with initialize_config_dir(version_base='1.3',config_dir=str(PROJECT_ROOT/'src/tasks/slcs/configs')):
        cfg=compose(config_name='precompute_dino_tokens',overrides=['paths.data_root=/home/kamimura/projects/tennis-lab/data','data.dataset_root=tennis_multivew/processed/meiji_3cam/dataset','data.split_file=tennis_multivew/processed/meiji_3cam/dataset/splits.json','precompute.device=cpu'])
        data_config=SLCSPrecomputeConfig.from_config(cfg).data.pipeline
    try:
        loaded=load_clip_arrays(ClipManifest.load(clip),config=data_config)
        report['slcs_reader']={'accepted':True,'player_valid_frames':loaded.player_label_valid.sum(axis=-1).tolist(),'ball_valid_frames':int(loaded.ball_label_valid.sum())}
    except Exception as error:
        report['slcs_reader']={'accepted':False,'error':f'{type(error).__name__}: {error}'}
if len(paths)==2:
    with np.load(paths['pipeline']) as a,np.load(paths['dataset']) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            x,y=a[key],b[key]
            assert x.shape==y.shape,(key,x.shape,y.shape)
            diff=np.abs(x.astype(np.float64)-y.astype(np.float64))
            report['comparison'][key]={'equal':bool(np.array_equal(x,y)),'absolute_difference':summarize(diff)}
            if key in ('player_yaw','gvhmr_aligned_player_yaw'):
                circular=np.abs(np.angle(np.exp(1j*(x.astype(np.float64)-y.astype(np.float64)))))
                report['comparison'][key]['circular_difference_degrees']=summarize(np.rad2deg(circular))
            del x,y,diff
for command in json.loads((root/'render_commands.json').read_text()):
    if command['phase'] not in paths:continue
    movie_paths=sorted(Path(command['output']).glob('*.mp4'))
    expected={'scene.mp4'} if command['module']=='visualization' else {f'{task}_viz.mp4' for task in ('ball_detection','court_kp','gvhmr','plcs','gvhmr_alignment','blcs')}
    assert {path.name for path in movie_paths}==expected,(command['phase'],command['module'],movie_paths)
    for path in movie_paths:
        movie_sha=digest(path)
        cached_report=partial_reports.get(command['phase'],{})
        cached=cached_report.get('videos',{}).get(str(path))
        if cached is not None and cached.get('sha256')==movie_sha and cached_report['scenes'][command['phase']]['sha256']==report['scenes'][command['phase']]['sha256']:
            if all(Path(cached[field]).is_file() and digest(Path(cached[field]))==cached.get(field+'_sha256') for field in ('contact_sheet','extreme_frame_contact_sheet')):
                report['videos'][str(path)]={**cached,'verification_reused_after_hash_check':True}
                continue
        output=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(path)]))['streams'][0]
        cap=cv2.VideoCapture(str(path)); count=0; blank=0; panels=[]
        candidate_frames={row['frame'] for row in report['scenes'][command['phase']]['inspection_candidates']}
        anomaly_panels=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            if np.std(cv2.resize(frame,(64,36)))<1:blank+=1
            if count in (0,505,1009):
                height=round(frame.shape[0]*640/frame.shape[1])
                panel=cv2.resize(frame,(640,height)); cv2.putText(panel,f'frame {count}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2); panels.append(panel)
            if count in candidate_frames:
                panel=cv2.resize(frame,(480,270));cv2.putText(panel,f'frame {count}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);anomaly_panels.append(panel)
            count+=1
        cap.release()
        assert count==1010,(path,count)
        contact=root/f'{command["phase"]}-{path.stem}-contact.jpg'
        cv2.imwrite(str(contact),np.concatenate(panels,axis=1))
        assert int(output['nb_frames'])==count
        recorded_fps=float(Fraction(output['avg_frame_rate']))
        assert abs(recorded_fps-manifest['fps'])<0.001,(path,recorded_fps,manifest['fps'])
        anomaly_contact=root/f'{command["phase"]}-{path.stem}-extremes.jpg'
        while len(anomaly_panels)%4:anomaly_panels.append(np.zeros_like(anomaly_panels[0]))
        cv2.imwrite(str(anomaly_contact),np.concatenate([np.concatenate(anomaly_panels[i:i+4],axis=1) for i in range(0,len(anomaly_panels),4)],axis=0))
        report['videos'][str(path)]={**output,'sha256':movie_sha,'contact_sheet_sha256':digest(contact),'extreme_frame_contact_sheet_sha256':digest(anomaly_contact),'decoded_frames':count,'blank_frames':blank,'contact_sheet':str(contact),'extreme_frame_contact_sheet':str(anomaly_contact),'fps_difference':recorded_fps-manifest['fps'],'duration_difference_seconds':float(output['duration'])-manifest['num_frames']/manifest['fps']}
report_path=root/('evaluation.json' if len(paths)==2 else f'evaluation-{next(iter(paths))}.json')
report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({'report':str(report_path),'scenes':list(report['scenes']),'videos':len(report['videos'])},indent=2))

if report['slcs_reader']['accepted'] is False:raise SystemExit(1)
