"""Small review tables plus content inventories; dense arrays stay in outputs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
OLD=Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')
MODES=('legacy','fixed_support','fixed_support_rgb')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    target=BASE/'report';target.mkdir(exist_ok=True)
    shutil.copyfile(OUT/'report/summary.json',target/'summary.json')
    for name in ('rgb-rejection-analysis.json','historical-transitions.json','api-parity.json','api-parity-before-diagnostics.json','api-parity-accepted.json',
                 'evidence-immutability.json','rgb-complete.json','complete-all.json'):
        shutil.copyfile(OUT/name,target/name)
    with (OUT/'report/cases.csv').open() as handle:cases=list(csv.DictReader(handle))
    meiji=[row for row in cases if row['dataset']=='meiji_4cam']
    with (target/'meiji4-cases.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(meiji[0]),lineterminator='\n');writer.writeheader();writer.writerows(meiji)
    by_key={}
    for row in cases:
        if row['dataset'] not in ('tennis_court_detector_val','synthetic_court_test'):continue
        item=by_key.setdefault(row['key'],{'key':row['key']})
        prefix={'legacy':'L','fixed_support':'A','fixed_support_rgb':'B'}[row['condition']]
        for key in ('geometry_valid','accepted','mean_error_px','max_error_px'):
            value=row[key]
            if value in ('True','False'):value=int(value=='True')
            elif value:value=f'{float(value):.6g}'
            item[f'{prefix}_{key}']=value
    for dataset,prefix in (('tennis_court_detector_val','val'),('synthetic_court_test','synthetic')):
        rows=[row for key,row in by_key.items() if key.startswith(dataset+'/')]
        for offset in range(0,len(rows),800):
            with (target/f'{prefix}-cases-{offset//800:02d}.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows[offset:offset+800])
    logs=BASE/'verification';logs.mkdir(exist_ok=True)
    for name in ('unit-tests.log','integration-tests.log','rgb-coordinate-contract.log','diagnostic-contracts.log','mutation.log','mutation-restored.log'):
        shutil.copyfile(OUT/name,logs/name)
    inventories=OUT/'inventories';inventories.mkdir(exist_ok=True)
    groups={f'input-{group}':(OLD/group,'**/raw_*.npz') for group in ('raw','val_raw','test_raw')}
    groups.update({'line-evidence':(OUT/'line-evidence','**/*.npz'),'rgb':(OUT/'rgb','**/*.png'),
                   'evaluation':(OUT/'evaluation','**/evaluation.json'),'review':(OUT/'review','*.jpg')})
    inventory={}
    for group,(root,pattern) in groups.items():
        total=0;count=0;tree=hashlib.sha256();destination=inventories/f'{group}.jsonl'
        with destination.open('w') as handle:
            for path in sorted(root.glob(pattern)):
                size=path.stat().st_size;record={'path':str(path.relative_to(root)),'bytes':size,'sha256':digest(path)}
                line=json.dumps(record,separators=(',',':'),ensure_ascii=False)+'\n'
                handle.write(line);tree.update(line.encode());total+=size;count+=1
        inventory[group]={'root':str(root),'files':count,'bytes':total,'jsonl_sha256':tree.hexdigest(),'inventory_path':str(destination)}
        print(group,count,total,flush=True)
    (BASE/'artifact-inventories.json').write_text(json.dumps(inventory,indent=2))


if __name__=='__main__':main()
