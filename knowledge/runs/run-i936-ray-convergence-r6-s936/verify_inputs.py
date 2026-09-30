"""Check unchanged source arrays and compare old/new saved 3D conditions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser()
for name in ('source','dataset','old_audit','output'):parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args()
source=json.loads((args.source/'manifest.json').read_text())
components,mixtures,rows=[],[],[]
for record in source['rallies']:
    rid=record['rally_id'];path=args.source/(rid+'.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['npz_sha256']
    with np.load(path) as z:a={k:z[k] for k in z.files}
    with np.load(args.dataset/(rid+'.npz')) as z:b={k:z[k] for k in z.files}
    changed={'gmm3d_means_m','gmm3d_covariance_m2','gmm3d_weights','gmm3d_method_codes'}
    keep=set(a)-changed
    assert all(np.array_equal(a[k],b[k]) for k in keep)
    old=[]
    for p in sorted(args.old_audit.glob(rid+'-*.npz')):
        with np.load(p) as z:old.append({k:z[k] for k in ('frames','means','weights')})
    old={k:np.concatenate([r[k] for r in old]) for k in old[0]}
    np.testing.assert_array_equal(old['frames'],np.arange(record['frames']))
    comp=np.linalg.norm(b['gmm3d_means_m'].astype(float)-old['means'],axis=-1).max(-1)
    mix=np.linalg.norm(np.einsum('tm,tmi->ti',b['gmm3d_weights'].astype(float),b['gmm3d_means_m'].astype(float))-np.einsum('tm,tmi->ti',old['weights'],old['means']),axis=-1)
    components.extend(comp);mixtures.extend(mix)
    rows.append({'rally_id':rid,'unchanged_fields':sorted(keep),'source_sha256':record['npz_sha256'],'new_sha256':hashlib.sha256((args.dataset/(rid+'.npz')).read_bytes()).hexdigest()})
mixtures=np.array(mixtures)
result={'frames':len(mixtures),'rallies':rows,'maximum_component_mean_shift_quantiles_m':np.quantile(components,[.5,.95,1]).tolist(),'mixture_mean_shift_quantiles_m':np.quantile(mixtures,[.5,.95,1]).tolist(),'mixture_mean_shift_above_2cm_frames':int((mixtures>.02).sum()),'all_preserved_source_arrays_match':True}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
