"""Promote bounded evidence; full images and per-frame predictions stay local."""
from pathlib import Path
import csv,hashlib,json,platform,shutil,subprocess

run=Path('knowledge/runs/run-court-video-drift-lines-20261117')
out=Path('outputs/tennis_scene/evaluate/court-video-drift-lines-20261117')
files=['development-protocol.json','development-summary.json','heldout-protocol.json','heldout-summary.json',
       'recovered-synthetic-fixtures.json','evaluation-protocol.json','evaluation-summary.json','evaluation-summary-bgr.json',
       'causes-summary.json','motion-distributions.json','result-integrity.json','review-summary.json','gallery-browser-check.json',
       'clips.csv','drift-timeline.svg','conflicting-lines.png','verification.log','configuration-verification.log',
       'reader-and-motion-final.log','multi-clip-final.log','runtime-reference-final.log','implementation-commit.log']
for name in files:shutil.copy2(out/name,run/name)
rows=list(csv.DictReader((out/'frames.csv').open()))
for camera in ['cam0','cam1','cam2','cam3']:
    with (run/f'frames-{camera}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(r for r in rows if r['camera']==camera)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
summary={}
for name in ['review','drift-store','causes']:
    inventory=out/f'{name}-inventory.jsonl';count=size=0
    with inventory.open('w') as f:
        for p in sorted((out/name).rglob('*')):
            if not p.is_file():continue
            row={'path':str(p.relative_to(out)), 'size':p.stat().st_size,'sha256':sha(p)}
            f.write(json.dumps(row)+'\n');count+=1;size+=row['size']
    summary[name]={'files':count,'bytes':size,'inventory_path':str(inventory),'inventory_sha256':sha(inventory)}
summary['retained_locally']=['development-detail.json','heldout-detail.json','causes-detail.json','evaluation-summary-bgr.json','evaluated-builder-bgr.py','gallery-warning.png']
summary['deleted_files']=[]
save(run/'artifact-inventories.json',summary)
save(run/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'opencv':__import__('cv2').__version__,'numpy':__import__('numpy').__version__,'gpu_used':False})
print(json.dumps(summary),flush=True)
