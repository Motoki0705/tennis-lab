"""Fixed C3 acceptance run: immutable PNG checkpoints, then validated videos."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

p=argparse.ArgumentParser()
p.add_argument('--blender',required=True)
p.add_argument('--package',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
work=a.output/'work'
artifacts=a.output/'artifacts'
checkpoints=a.output/'checkpoints'
for path in (work,artifacts,checkpoints):
    path.mkdir(parents=True)
package=json.loads((a.package/'manifest.json').read_text())
ids=[r['player_id'] for r in package['players']]
assert ids==[0,1] and package['num_frames']==1010
source_hash=hashlib.sha256((a.package/'manifest.json').read_bytes()).hexdigest()
start={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'started_at':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,
       'package_sha256':source_hash,'player_ids':ids,'human_acceptance':'pending'}
(artifacts/'run_start.json').write_text(json.dumps(start,indent=2))
blocks=[]
for identity in ids:
    destination=work/'renders'/f'player_{identity}'
    saved=0
    def snapshot(force=False):
        global saved
        report_path=destination/'render.json'
        if not report_path.exists():
            return
        report=json.loads(report_path.read_text())
        n=len(report['frames'])
        if n<=saved or not (force or (saved==0 and n>=4) or n//128>saved//128):
            return
        rows=report['frames'][saved:n]
        name=f'player_{identity}_{saved:05d}_{n-1:05d}.tar'
        staging=work/(name+'.partial')
        meta={**report,'frames':rows,'complete':False,'checkpoint_indices':[saved,n]}
        payload=json.dumps(meta,indent=2).encode()
        with tarfile.open(staging,'w') as archive:
            header=tarfile.TarInfo(f'player_{identity}/checkpoint.json')
            header.size=len(payload)
            archive.addfile(header,io.BytesIO(payload))
            for row in rows:
                frame=f"{row['output_index']:05d}.png"
                archive.add(destination/'frames'/frame,arcname=f'player_{identity}/frames/{frame}',recursive=False)
        os.replace(staging,checkpoints/name)
        blocks.append({'player_id':identity,'path':name,'output_indices':[saved,n],
                       'source_frames':[r['source_frame'] for r in rows]})
        saved=n
        print('C3_CHECKPOINT_READY '+json.dumps(blocks[-1]),flush=True)
    command=[a.blender,'--background','--factory-startup','--python-exit-code','1','--threads','4',
             '--python','src/tennis_scene/visualization/blender/shots/player_view.py','--',
             '--package',str(a.package),'--output',str(destination),'--player-id',str(identity),'--gpu-backend','OPTIX']
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line,end='',flush=True)
            if 'Saved:' in line or 'PLAYER_VIEW_COMPLETE' in line:
                snapshot()
        code=process.wait()
        snapshot(force=True)
        if code:
            raise RuntimeError(f'Blender ID{identity} failed: {code}')
    report=json.loads((destination/'render.json').read_text())
    assert report['complete'] and len(report['frames'])==1010 and saved==1010
    report_delivery=work/f'render_player_{identity}.json.delivery'
    shutil.copyfile(destination/'render.json',report_delivery)
    os.replace(report_delivery,artifacts/f'render_player_{identity}.json')
video_output=work/'videos'
subprocess.run(['.venv/bin/python','-m','src.tennis_scene.scripts.publish_player_views',
    '--package',str(a.package),'--renders',str(work/'renders'),'--output',str(video_output)],check=True)
subprocess.run(['ffmpeg','-v','error','-n','-i',str(video_output/'comparison.mp4'),'-frames:v','240',
    '-an','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
    str(video_output/'comparison_short.mp4')],check=True)
# The publication reader performs an independent full decode and PTS check.
subprocess.run(['.venv/bin/python','-c',
    'import json,sys; from pathlib import Path; from src.tennis_scene.visualization.player_view.video import verify_video; '
    'p=Path(sys.argv[1]); r=verify_video(p/"comparison_short.mp4",frames=240,fps=59.94006,width=2560,height=720); '
    '(p/"short_comparison.json").write_text(json.dumps({"source_frames":[0,240],"verification":r,"human_acceptance":"pending"},indent=2))',
    str(video_output)],check=True)
for source in video_output.iterdir():
    temporary=work/(source.name+'.delivery')
    shutil.copyfile(source,temporary)
    os.replace(temporary,artifacts/source.name)
manifest_delivery=work/'checkpoint_manifest.json.delivery'
manifest_delivery.write_text(json.dumps({'package_sha256':source_hash,'blocks':blocks},indent=2))
os.replace(manifest_delivery,artifacts/'checkpoint_manifest.json')
print('C3_FULL_DELIVERY_COMPLETE',flush=True)
