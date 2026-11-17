"""CPU replay of preserved observations; explicit controlled correspondences, no new inference."""
import os,runpy,time,json,hashlib,dataclasses,threading,resource,statistics,subprocess
from pathlib import Path
import numpy as np
import psutil
ns=runpy.run_path(str(Path(__file__).with_name('probe_real.py')))
source,runtime,side,inputs,calibration=(ns['source'],ns['runtime'],ns['side'],ns['inputs'],ns['artifacts']['calibration'])
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentModule,CameraAlignmentInput
from src.tennis_scene.pipeline.components.body_view_selection import BodyViewSelectionModule,BodyViewSelectionInput
from src.tennis_scene.pipeline.observation_types import ObjectObservations,group_observations
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
process=psutil.Process()
def frozen(x):
    if isinstance(x,np.ndarray):return {'dtype':str(x.dtype),'shape':x.shape,'sha256':hashlib.sha256(x.tobytes()).hexdigest(),'all_finite':bool(np.isfinite(x).all())}
    if dataclasses.is_dataclass(x):return {f.name:frozen(getattr(x,f.name)) for f in dataclasses.fields(x)}
    if isinstance(x,dict):return {k:frozen(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [frozen(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.generic):return x.item()
    return x
def summarize(x):
    data=frozen(x);encoded=json.dumps(data,sort_keys=True,allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(),data

def align(people,cap):
    return CameraAlignmentModule(source.camera_ids,runtime.camera_geometry,player_reprojection_px=runtime.player_reprojection_px,ball_reprojection_px=runtime.ball_reprojection_px,joint_confidence=runtime.joint_confidence,max_frames=cap).process(CameraAlignmentInput(source,calibration,side,people,inputs.balls))
# Physically consistent dense two-person projection for the CPU worst-coverage path.
t=source.num_frames;p=2;j=17
xyz=np.zeros((p,t,j,3),np.float64)
for person in range(p):
    xyz[person,:,:,0]=(person*4-2)+.2*np.sin(np.arange(t)[:,None]/100)+np.arange(j)[None,:]%3*.1
    xyz[person,:,:,1]=person*8-4+np.arange(j)[None,:]%4*.1
    xyz[person,:,:,2]=1+np.arange(j)[None,:]*.03
uv=np.stack([c.project(xyz)[0] for c in ns['outputs'][2048].geometry.cameras]).transpose(0,2,1,3,4).astype(np.float32)
lo,hi=uv.min(-2),uv.max(-2)
boxes=np.concatenate(((lo+hi)/2,(np.max(hi-lo,axis=-1)+50)[...,None]),-1).astype(np.float32)
raw=ObjectObservations(source.camera_ids,source.size,source.fps,uv,np.ones(uv.shape[:-1],np.float32),np.ones(uv.shape[:3],bool),np.tile(np.arange(p,dtype=np.int64),(len(source.camera_ids),1)),boxes)
ids=np.broadcast_to(np.arange(p,dtype=np.int64),raw.observed.shape).copy()
dense=group_observations(raw,ids,threshold=runtime.human_vis_threshold)
dense_alignment=align(dense,2048)
scopes={
 'court_side_saved':('saved calibrated ball observations',lambda cap:CourtSideModule(source.camera_ids,runtime.court_side,max_frames=cap).process(inputs)),
 'camera_alignment_saved_ball_only':('saved calibration/side/ball, explicit empty people evidence',lambda cap:align(ns['empty_people'],cap)),
 'body_view_selection_saved_provisional':('saved pose/boxes, provisional carrier-index IDs, not an association result',lambda cap:BodyViewSelectionModule(source.camera_ids,max_frames=cap).process(BodyViewSelectionInput(source,ns['outputs'][2048],ns['raw'],ns['grouped']))),
 'camera_alignment_dense':('controlled dense two-person projections with real calibrated cameras and saved ball',lambda cap:align(dense,cap)),
 'body_view_selection_dense':('controlled fully observed two-person trajectories, 1055-frame body requests',lambda cap:BodyViewSelectionModule(source.camera_ids,max_frames=cap).process(BodyViewSelectionInput(source,dense_alignment,raw,dense))),
}
records=[]
for name,(scope,function) in scopes.items():
    for cap in (2048,4096):
        function(cap) # one additional warmup; setup probe has also run before timed cases
        seconds=[];hashes=[];peaks=[];baselines=[]
        for repeat in range(5):
            before=process.memory_info().rss;peak=[before];done=threading.Event()
            def sampler():
                while not done.wait(.002):peak[0]=max(peak[0],process.memory_info().rss)
            thread=threading.Thread(target=sampler);thread.start();start=time.perf_counter()
            output=function(cap)
            elapsed=time.perf_counter()-start;peak[0]=max(peak[0],process.memory_info().rss);done.set();thread.join()
            digest,document=summarize(output)
            seconds.append(elapsed);hashes.append(digest);peaks.append(peak[0]);baselines.append(before)
        if len(set(hashes))!=1:raise ValueError('Repeated numerical outputs changed')
        row={'case':name,'scope':scope,'cap':cap,'repeats':5,'seconds':seconds,'mean_seconds':statistics.mean(seconds),'sample_stdev_seconds':statistics.stdev(seconds),'rss_before_bytes':baselines,'rss_sampled_peak_bytes':peaks,'rss_max_observed_increase_bytes':max(p-b for p,b in zip(peaks,baselines)),'process_lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'output_sha256':hashes[0]}
        if name.startswith('body_view'):
            row['request_lengths']=[[len(r.source_frames) for r in s.requests] for s in output.selections]
        records.append(row);print(json.dumps(row),flush=True)
for name in scopes:
    pair=[x for x in records if x['case']==name]
    assert pair[0]['output_sha256']==pair[1]['output_sha256'],name
assert max(n for row in records if row['case']=='body_view_selection_dense' for seq in row['request_lengths'] for n in seq)==1055
import torch,scipy,platform
assert not torch.cuda.is_initialized()
result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_frames':source.num_frames,'sampled_frames':len(sampled_frame_indices(source.num_frames,source.fps,max_frames=2048)),'scopes':list(scopes),'all_caps_outputs_bitwise_equal':True,'cuda_initialized':False,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}},'records':records}
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
