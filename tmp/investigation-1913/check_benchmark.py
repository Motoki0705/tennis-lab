import json,statistics
from pathlib import Path
r=Path('tmp/investigation-1913');files=list((r/'evidence/arch-benchmark').glob('*block-*__full-*.json'));drive=json.loads(next((r/'evidence/arch-benchmark').glob('*summary.json')).read_text());pr=json.load(open(r/'prs/1910/knowledge/runs/run-ball-arch-efficiency-s42-20261111/benchmark-summary.json'));out={}
for key in ['C0','C1','C2','C3','CA','T1','T2']:
 ds=[json.loads(p.read_text()) for p in files if p.name.endswith('__full-'+key+'.json')];assert len(ds)==3
 for d in ds:assert d['kind']=='untrained benchmark' and d['warmup']>=10 and d['repetitions']==30
 infer=[t['cuda_ms']/d['frames_per_inference'] for d in ds for t in d['raw']['inference']];updates=[t['total']['cuda_ms'] for d in ds for t in d['raw']['training']]
 assert len(infer)==len(updates)==90
 allocated=max(t['peak_allocated'] for d in ds for t in d['raw']['training'])/(1024**3)
 mac_records=[d['macs'] for d in ds if 'macs' in d];assert mac_records
 macs=mac_records[0]['conv_linear_attention_macs_per_frame']/1e9;assert all(m['conv_linear_attention_macs_per_frame']/1e9==macs for m in mac_records)
 vals=dict(counted_gmac_per_frame=macs,inference_ms_per_frame_p50=statistics.median(infer),update_ms_p50=statistics.median(updates),peak_allocated_gib=allocated)
 for s in [drive,pr]:
  c=s['cases']['full-'+key];assert vals['inference_ms_per_frame_p50']==c['inference_ms_per_frame']['p50'];assert vals['update_ms_p50']==c['update_cuda_ms']['p50'];assert vals['counted_gmac_per_frame']==c['macs']['conv_linear_attention_macs_per_frame']/1e9
 out[key]=vals
for key,case in pr['cases'].items():
 for k,v in case.items():assert drive['cases'][key][k]==v,(key,k)
node=__import__('yaml').safe_load((r/'prs/1910/knowledge/nodes/ball_detection/000069-run-ball-arch-efficiency-s42-20261111.md').read_text().split('---',2)[1]);assert node['metrics']==out
print('21 raw timing JSONs -> 7 full-model metrics and node: EXACT MATCH. All 48 PR case fields equal Drive summary (PR omits cold_start/compilation).')
for k,v in out.items():print(k,v)
print('C2 MAC reduction %',100*(1-out['C2']['counted_gmac_per_frame']/out['C0']['counted_gmac_per_frame']))
print('C2 update reduction %',100*(1-out['C2']['update_ms_p50']/out['C0']['update_ms_p50']))
(r/'analysis/benchmark-recomputed.json').write_text(json.dumps({'cases':out,'raw_full_model_jsons':21,'samples_per_case':90,'pr_48_cases_equal_drive_summary':True,'note':'Drive cold_start/compilation retained in raw evidence, omitted in small PR summary'},indent=2)+'\n')
