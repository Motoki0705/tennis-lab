"""One command to admit the smoke, benchmark or five-model campaign exclusively."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,str(Path.cwd()))
from protocol import REPO, ROOT, write

p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','measure','train'));p.add_argument('--attempt',type=int,default=1)
a=p.parse_args()
session=os.environ.get('CODEX_SESSION_ID')
if not session:raise RuntimeError('Codex session attribution is unavailable; do not invent an id')
base=Path(__file__).resolve().parent
entry={'smoke':[base/'smoke.py'],'measure':[base/'measure.py','matrix'],'train':[base/'serial.py','run']}[a.mode]
if a.mode in ('measure','train') and not (ROOT/'smoke/COMPLETED.json').exists():raise RuntimeError('Five-condition smoke must complete first')
if a.mode=='train' and not (ROOT/'benchmark-receipt.json').exists():raise RuntimeError('Benchmarks must be verified first')
command=['env','TENNIS_LAB_DINO_EXTENSION='+str(ROOT/'msda-build/lib/MultiScaleDeformableAttention.so'),
         'OMP_NUM_THREADS=2','TORCHINDUCTOR_COMPILE_THREADS=2',sys.executable,*map(str,entry)]
queue=Path.cwd()/'.agents/skills/training-queue/scripts/training_queue.sh'
env=dict(os.environ,TRAINING_QUEUE_DIR=str(REPO/'.training_queue'))
request=['bash',str(queue),'add',shlex.join(command),'--name','arch-efficiency-'+a.mode,'--provider','codex',
         '--session',session,'--issue','1859','--resource','all','--idempotency-key',f'arch-efficiency-s42-20261111/{a.mode}/attempt-{a.attempt}']
result=subprocess.run(request,env=env,check=True,capture_output=True,text=True)
print(result.stdout.strip())
record=dict(mode=a.mode,attempt=a.attempt,resource='all',queue_result=result.stdout.strip(),command=command,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),queue=str(REPO/'.training_queue'))
write(ROOT/'queue'/f'{a.mode}-attempt-{a.attempt}.json',record)
subprocess.run(['bash',str(queue),'start','--idle-timeout','300'],env=env,check=True)
