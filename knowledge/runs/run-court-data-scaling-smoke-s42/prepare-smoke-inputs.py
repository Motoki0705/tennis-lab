from pathlib import Path
import json
from src.synthetic_data_generation.dataset.court.sample_store import open_court_store,read_court_manifest,finish_court_store
from src.tasks.court_detection.scaling.real_split import real_source,make_registry
from src.utils.data.image_record_store import ImageRecordWriter
base=Path('/home/kamimura/projects/tennis-lab')
local=Path.cwd()/'tmp/court-scale/smoke-data'
root=base/'data/synthetic_data_generation/scenes/B00/datasets/court'
store=open_court_store(root)
manifest=read_court_manifest(root,store=store)
selected=manifest['samples'][:8]
manifest['samples']=selected
with ImageRecordWriter(local/'synthetic/samples') as writer:
 for sample in selected:
  row=sample['image_index'];writer.append(store.jpeg(row),store.record(row))
 finish_court_store(local/'synthetic',writer,manifest)
source=real_source(base/'data/court_detection/tennis_court_detector-v1')
records=[*source.records('train'),*source.records('val')]
# Four existing videos and the excluded annotation, copied into a smoke-only store.
ids=[];videos=set()
for split in ('train','val'):
 count=0
 for record in source.records(split):
  video=record.sample_id.rsplit('_',1)[0]
  if video not in videos:
   videos.add(video);ids.append(record.payload['image_index']);count+=1
  if count==2:break
for row in range(len(source.image_store)):
 if source.image_store.record(row)['id']=='QszoUKyCOHo_600':ids.append(row);break
with ImageRecordWriter(local/'real') as writer:
 for row in ids:writer.append(source.image_store.jpeg(row),source.image_store.record(row))
 storage=writer.finish({'source_schema':'tennis_court_detector_annotations_v1'})
(local/'real/dataset.json').write_text(json.dumps({'schema':'tennis_court_detector_store_v1','storage':storage}))
# Use at least one record from each original split, required by the unchanged real loader.

registry=make_registry(real_source(local/'real'),version='smoke-real-v1',seed=42)
out=Path.cwd()/'outputs/court_detection/evaluate/smoke-real-video-v1/s42'
out.mkdir(parents=True,exist_ok=False)
(out/'registry.json').write_text(json.dumps(registry,indent=2))
print('synthetic generation seed',manifest['seed'])
print(registry['summary'])
