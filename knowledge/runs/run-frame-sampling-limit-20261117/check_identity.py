import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from hydra import compose,initialize_config_dir
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.contracts import ClipSource,SourceVideo
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
root=Path('/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/datasets/repeat-1/videos/video_000/clips/clip_035/annotations/tennis_scene')
index=json.loads((root/'scene.json').read_text()); source=ClipSource(index['source']['clip_id'],tuple(SourceVideo(**{**v,'path':Path(v['path'])}) for v in index['source']['videos']))
nodes={};runtime={}
for limit in (1024,2048):
    with initialize_config_dir(config_dir=str(Path('src/tennis_scene/configs').resolve()),version_base='1.3'):
        cfg=compose(config_name='pipeline',overrides=['device=cpu',f'frame_sampling.max_frames={limit}'])
    runtime[limit]=PipelineRuntimeConfig.from_config(cfg,bind_inputs=False)
    nodes[limit]={n.name:n for n in standard_definition(runtime[limit],source,code_identity='same-src-python-tree')}
changed={name for name in nodes[1024] if nodes[1024][name].settings!=nodes[2048][name].settings}
assert changed=={'court_side','camera_alignment','body_view_selection'},changed
affected=set(changed)
while True:
    more={name for name,n in nodes[2048].items() if set(n.bindings.values())&affected}
    if more<=affected:break
    affected|=more
manifest=Path('data/tennis_multiview/processed/meiji_4cam/dataset/dataset.json')
dataset=json.loads(manifest.read_text())
lengths=[{'clip_id':r['clip_id'],'source_frames':r['num_frames'],'fps':r['fps'],'sampled_frames':len(sampled_frame_indices(r['num_frames'],r['fps'],max_frames=4096))} for r in dataset['clips']]
result={'direct_settings_changes':sorted(changed),'invalidated_by_dependencies':sorted(affected),'unchanged_upstream':sorted(set(nodes[1024])-affected),'publication_processing_settings_changed':runtime[1024].processing_settings!=runtime[2048].processing_settings,'dataset_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'clip_count':len(lengths),'lengths':lengths,'over_1024':[r['clip_id'] for r in lengths if r['sampled_frames']>1024],'over_2048':[r['clip_id'] for r in lengths if r['sampled_frames']>2048]}
assert len(lengths)==69 and result['over_1024']==['video_000/clip_035'] and not result['over_2048']
Path(__file__).with_name('identity-and-lengths.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='lengths'},flush=True)
