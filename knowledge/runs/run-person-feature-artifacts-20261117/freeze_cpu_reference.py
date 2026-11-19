import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import types
test_package = types.ModuleType("tests")
test_package.__path__ = [str(Path("tests").resolve())]
sys.modules["tests"] = test_package
import numpy as np
import torch

from src.tasks.person_tracking.archive import save_features
from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionInput, PlayerSelectionModule
from src.tennis_scene.pipeline.contracts import SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config, fake_encoder_spec, materialize_model_weights
from tests.unit.tennis_scene.pipeline.components.test_player_selection import _input

REV='f7b016700'
root=Path('outputs/person_feature_acceptance')

def legacy(name, path):
    code=subprocess.check_output(['git','show',f'{REV}:{path}'],text=True)
    mod=importlib.util.module_from_spec(importlib.util.spec_from_loader(name,loader=None))
    sys.modules[name]=mod
    exec(compile(code,path,'exec'),mod.__dict__)
    return mod
features=legacy('legacy_features','src/tasks/person_tracking/features.py')
tracking=legacy('legacy_tracking','src/tennis_scene/pipeline/components/person_tracking.py')
selection=legacy('legacy_selection','src/tennis_scene/pipeline/components/player_selection.py')
frames=64
calibrated=_input([(0.,5.)]*2,np.ones((2,frames),bool))
box=calibrated.tracks.boxes_xyxy[0,0]
all_boxes=[]; rows=[]; offsets=[0]
for frame in range(frames):
    count=0 if frame in {15,16,63} else 2
    all_boxes.extend([box]*count)
    rows.extend([frame*7+2,frame*7+5][:count])
    offsets.append(len(rows))
detections=PersonDetectionOutput('cam0',np.array(offsets,np.int64),np.array(all_boxes,np.float32),
    np.full(len(rows),.9,np.float32),np.array(rows,np.int64))
class Pose:
    def __init__(self,*a,**kw): pass
    def load(self): pass
    def unload(self): pass
    def predict(self,request):
        frame,_=next(iter(request.frames_bgr))
        fractions=np.array([.25,.75] if frame<10 else [.75,.25],np.float32)
        result=np.zeros((len(request.bbx_xys),17,3),np.float32)
        result[...,:2]=box[None,None,:2]+fractions[:,None,None]*(box[None,None,2:]-box[None,None,:2])
        result[...,2]=1.4
        return SimpleNamespace(keypoints=torch.from_numpy(result))
class Encoder:
    name=TrackingConfig().encoder
    input_size=(8,4)
    def embed(self,pixels):
        result=torch.zeros((len(pixels),1280));result[:,0]=1
        return result
class Link:
    def __init__(self,path): pass
    def links(self,boxes,observed):return {i:i for i in range(len(boxes))},[]
recorded=[]
class Capture(features.FeatureExtractor):
    def extract(self,*a,**kw):
        value=super().extract(*a,**kw);recorded.append(value);return value
tracking.FeatureExtractor=Capture
tracking.ViTPosePose2D=Pose
tracking.AFLink=Link
tracking.OpenCVVideoFrameReader=lambda *a,**kw:(SimpleNamespace(index=i,frame=np.zeros((1080,1920,3),np.uint8)) for i in range(frames))
video=SourceVideo('cam0',root/'not-opened.mp4','fixed-model-fakes',frames,30.,1920,1080)
tracker=materialize_model_weights(tracking.PersonTrackingModule(TrackingConfig(),people=make_people_config(root),
    encoder=fake_encoder_spec(Encoder),aflink_checkpoint=root/'aflink.pth',show_progress=False))
result=tracker.process(tracking.PersonTrackingInput(video,detections))
sampler=calibrated
module=selection.PlayerSelectionModule(sampler and __import__('src.tasks.person_tracking.court_linking',fromlist=['LinkingConfig']).LinkingConfig(),
    footpoints=__import__('src.tasks.player_association.geometry.footpoints',fromlist=['FootpointConfig']).FootpointConfig(),
    sampling=__import__('src.tasks.player_association.appearance.sampling',fromlist=['CropSamplingConfig']).CropSamplingConfig(),
    encoder=fake_encoder_spec(Encoder),encoder_name=Encoder.name,device='cpu',enabled=True)
selected=module.process(selection.PlayerSelectionInput(video,calibrated.calibration,result))
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
out=Path('tests/integration/tennis_scene/fixtures')
save_features(out/'person-features-before-split.npz',recorded,{'revision':REV,'contract':'pre-split FeatureExtractor + PersonTrackingModule, deterministic CPU model fakes'})
# Full numerical output and diagnostics, including GSI and regrouped evidence.
codec=ArtifactCodec(type(result));payload,arrays=codec.dump(result,out/'unused') if False else ({},{})
np.savez_compressed(out/'person-tracking-before-split.npz',
    track_ids=result.track_ids,boxes=result.boxes_xyxy,observed=result.observed,
    detection_rows=result.evidence.detection_rows,poses=result.evidence.poses,
    embeddings=result.evidence.embeddings,appearance_valid=result.evidence.appearance_valid,
    reconstructed_boxes=result.reconstruction.boxes,interpolated=result.reconstruction.interpolated,
    selected=selected.selected,group_ids=selected.tracks.track_ids,group_boxes=selected.tracks.boxes_xyxy,
    group_observed=selected.tracks.observed,origin_rows=selected.origin_rows,
    group_detection_rows=selected.tracks.evidence.detection_rows,
    source_track_ids=np.asarray(result.source_track_ids,np.int64))
(out/'person-selection-before-split.json').write_text(json.dumps({'revision':REV,'diagnostics':selected.diagnostics},sort_keys=True,indent=2)+'\n')
print(json.dumps({'frames':len(recorded),'rows':len(rows),'track_ids':result.track_ids.tolist(),'selected_observations':int(selected.selected.sum()),'synthetic':int(result.reconstruction.interpolated.sum())}))
