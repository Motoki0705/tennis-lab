import json,sys,time,resource,threading
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import psutil
from hydra import compose,initialize_config_dir
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.contracts import ClipSource,SourceVideo,AssemblyContext
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.components.court_calibration import CourtCalibrationOutput
from src.tennis_scene.pipeline.components.ball_points import BallPointsOutput
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
from src.tennis_scene.pipeline.input_assembly.reconstruction import CourtSideInputAssembler
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
root=Path('/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/datasets/repeat-1/videos/video_000/clips/clip_035/annotations/tennis_scene')
index=json.loads((root/'scene.json').read_text())
source=ClipSource(index['source']['clip_id'],tuple(SourceVideo(**{**v,'path':Path(v['path'])}) for v in index['source']['videos']))
def load(name,kind):
    ref=index['artifacts'][name]
    doc=read_component_descriptor(root,ref,node=name,source_sha256=index['source_sha256'])
    return ArtifactCodec(kind).load(doc['payload'],(root/ref['path']).parent,doc['arrays'])
artifacts={'calibration':load('court_calibration',CourtCalibrationOutput)}
for camera in source.camera_ids:artifacts['ball_'+camera]=load('ball_points/'+camera,BallPointsOutput)
inputs=CourtSideInputAssembler(0.).assemble(AssemblyContext(source),artifacts)
with initialize_config_dir(config_dir=str(Path('src/tennis_scene/configs').resolve()),version_base='1.3'):
    cfg=compose(config_name='pipeline',overrides=['device=cpu'])
    runtime=PipelineRuntimeConfig.from_config(cfg,bind_inputs=False)
rows=[];process=psutil.Process()
for limit in (1024,2048,4096):
    peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.005):peak[0]=max(peak[0],process.memory_info().rss)
    thread=threading.Thread(target=sample);thread.start();started=time.perf_counter()
    try:
        output=CourtSideModule(source.camera_ids,runtime.court_side,max_frames=limit).process(inputs)
        row={'status':'complete','camera_ids':output.camera_ids,'turns':output.view_half_turns,'frames':output.frames,'margin':output.margin,'hypotheses':[vars(h) for h in output.hypotheses]}
    except Exception as error:
        row={'status':'failed','error_type':type(error).__name__,'reason':getattr(error,'reason',None),'error':str(error),'diagnostics':getattr(error,'diagnostics',{})}
    finally:
        seconds=time.perf_counter()-started;stop.set();thread.join()
    row.update(limit=limit,seconds=seconds,rss_sampled_peak=peak[0],process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    rows.append(row);print(json.dumps(row,default=str),flush=True)
Path('outputs/tennis_scene/evaluate/frame-sampling-limit-20261117/real-side.json').write_text(json.dumps(rows,indent=2,default=str)+'\n')
import numpy as np
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentInput,CameraAlignmentModule
from src.tennis_scene.pipeline.components.body_view_selection import BodyViewSelectionInput,BodyViewSelectionModule
from src.tennis_scene.pipeline.observation_types import GroupedObservations,group_observations,ObjectObservations
from src.tennis_scene.pipeline.input_assembly.observations import gather_people
side=CourtSideModule(source.camera_ids,runtime.court_side,max_frames=2048).process(inputs)
views=len(side.camera_ids);frames=source.num_frames
empty_people=GroupedObservations(np.empty(0,np.int64),np.empty((0,views,frames,17,2),np.float32),np.empty((0,views,frames,17),np.float32),np.zeros((0,views,frames,17),bool),np.empty((0,views,frames),np.int64))
outputs={};more=[]
for limit in (2048,4096):
    module=CameraAlignmentModule(source.camera_ids,runtime.camera_geometry,player_reprojection_px=runtime.player_reprojection_px,ball_reprojection_px=runtime.ball_reprojection_px,joint_confidence=runtime.joint_confidence,max_frames=limit)
    start=time.perf_counter()
    try:
        alignment=module.process(CameraAlignmentInput(source,artifacts['calibration'],side,empty_people,inputs.balls))
        outputs[limit]=alignment
        row={'stage':'camera_alignment','limit':limit,'status':'complete','scope':'real calibration+ball observations, people evidence omitted explicitly','document':alignment.geometry.document,'camera_ids':alignment.geometry.camera_ids}
    except Exception as error:row={'stage':'camera_alignment','limit':limit,'status':'failed','error':str(error),'reason':getattr(error,'reason',None),'diagnostics':getattr(error,'diagnostics',{})}
    row.update(seconds=time.perf_counter()-start,rss=process.memory_info().rss,process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);more.append(row);print(json.dumps(row,default=str),flush=True)
if 2048 in outputs:
    for c in source.camera_ids:artifacts['pose_'+c]=load('pose_estimation/'+c,ObjectObservations)
    raw=gather_people(source,artifacts)
    active=tuple(v.source_index for v in artifacts['calibration'].calibration.views)
    raw=raw.select_views(active)
    # Controlled identity correspondence only; no person association or new image inference is claimed.
    object_ids=np.broadcast_to(np.arange(raw.uv_px.shape[2],dtype=np.int64),(views,frames,raw.uv_px.shape[2])).copy();object_ids[~raw.observed]=-1
    grouped=group_observations(raw,object_ids,threshold=runtime.human_vis_threshold)
    for limit in (2048,4096):
        start=time.perf_counter()
        selection=BodyViewSelectionModule(source.camera_ids,max_frames=limit).process(BodyViewSelectionInput(source,outputs[2048],raw,grouped))
        counts=[{'person_id':s.person_id,'camera_id':s.camera_id,'request_lengths':[len(r.source_frames) for r in s.requests],'observed_samples':s.observed_samples} for s in selection.selections]
        all_finite=all(np.isfinite(r.keypoints).all() and np.isfinite(r.boxes_xys).all() for s in selection.selections for r in s.requests)
        row={'stage':'body_view_selection','limit':limit,'scope':'saved poses/boxes with explicit provisional carrier-index correspondence, not estimated player identity','status':'complete','seconds':time.perf_counter()-start,'selections':counts,'all_finite':all_finite,'rss':process.memory_info().rss,'process_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};more.append(row);print(json.dumps(row,default=str),flush=True)
Path('outputs/tennis_scene/evaluate/frame-sampling-limit-20261117/real-downstream.json').write_text(json.dumps(more,indent=2,default=str)+'\n')
