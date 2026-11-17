"""Small summaries in Git; per-image matrices stay in inventoried outputs."""
import hashlib
import json
import platform
from pathlib import Path
import shutil

import cv2
import numpy
import scipy
import torch

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-legacy-default-20261117')


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['images']==3545 and summary['regions']==8530 and summary['limit'] is None
    assert summary['mismatches']==[]
    rows=['|入力|画像|領域|H成立（旧=既定）|採用（旧=既定）|全比較完全一致|',
          '|---|---:|---:|---:|---:|---:|']
    for name,r in summary['datasets'].items():
        rows.append(f"|{name}|{r['images']}|{r['regions']}|{r['any_h']}|{r['accepted']}|{r['all_exact']}|")
    (BASE/'table.md').write_text('\n'.join(rows)+'\n')
    for name in ('summary.json','input-integrity.json','mutations.json'):
        shutil.copyfile(OUT/name,BASE/name)
    (BASE/'logs').mkdir(exist_ok=True)
    for name in ('targeted-tests.log','related-tests.log','legacy-known-h.log','mypy.log','mutations.log','input-integrity.log',
                 'final-tests.log','pre-commit.log','knowledge-validation.log'):
        path=OUT/'logs'/name
        if path.exists():
            assert path.stat().st_size<=256*1024
            shutil.copyfile(path,BASE/'logs'/name)
    lines=[]
    for path in sorted((OUT/'evaluation').rglob('comparison.json')):
        data=path.read_bytes()
        lines.append(json.dumps({'path':str(path.relative_to(OUT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))
    assert len(lines)==3545
    dest=OUT/'inventories/comparisons.jsonl';dest.parent.mkdir(exist_ok=True)
    dest.write_text('\n'.join(lines)+'\n')
    (BASE/'artifact-inventories.json').write_text(json.dumps({'comparisons':{'root':str(OUT),
        'files':len(lines),'inventory_path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},
        'inputs':'knowledge/runs/run-court-fixed-support-rgb-20261117/artifact-inventories.json'},indent=2)+'\n')
    environment={'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,
                 'opencv':cv2.__version__,'torch':torch.__version__,'gpu_used':False}
    (BASE/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print('\n'.join(rows))

if __name__=='__main__':main()
