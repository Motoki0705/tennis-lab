"""Consume atomic geometry results as they complete; score every record exactly once."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import time

from summarize_final import BASE, process, summarize


def main():
    parser=argparse.ArgumentParser()
    for name in ['raw','legacy','current','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    references=json.loads((BASE/'references.json').read_text());protocol=json.loads((BASE/'protocol.json').read_text())
    assert references['approved_for_evaluation'] and hashlib.sha256((BASE/'references.json').read_bytes()).hexdigest()==protocol['reference_sha256']
    records=json.loads((args.raw/'manifest.json').read_text())['records'];pending={r['key']:r for r in records}
    args.output.mkdir(parents=True,exist_ok=True)
    assert not list(args.output.glob('**/evaluation.json')), 'Use a fresh destination; do not mix evaluation identities'
    started=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        while pending or futures:
            ready=[key for key in pending if all((root/key/'geometry.json').is_file() for root in [args.legacy,args.current])]
            for key in ready:
                record=pending.pop(key);f=pool.submit(process,(record,args.raw,args.legacy,args.current,references,args.output));futures[f]=key
            if not futures:
                if all((root/'complete.json').is_file() for root in [args.legacy,args.current]):raise FileNotFoundError(sorted(pending))
                time.sleep(1);continue
            done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for f in done:
                rows.append(f.result());del futures[f]
            if done and len(rows)%100==0:print('scored',len(rows),'of',len(records),flush=True)
    assert len(rows)==len(records)
    summary={'protocol':protocol,'partial':False,'summary':summarize(rows)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (args.output/'complete.json').write_text(json.dumps({'records':len(rows),'seconds':time.time()-started,'workers':args.workers,'includes_dependency_wait':True},indent=2))
    print('complete',len(rows),flush=True)


if __name__=='__main__':main()
