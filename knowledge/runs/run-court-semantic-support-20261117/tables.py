"""Format final JSON measurements without fitting or selecting from GT."""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')
MODES=('legacy','fixed_support','fixed_support_rgb','new_support','new_rgb')
NAMES={'legacy':'旧処理','fixed_support':'前回(a)','fixed_support_rgb':'前回(a)+(b)','new_support':'今回(a)','new_rgb':'今回(a)+(b)'}

def f(value,digits=2):return '—' if value is None else f'{value:.{digits}f}'
def confusion(row):return ' / '.join(str(row[k]) for k in ('TP','FN','FP','TN'))

def main():
 r=json.loads((OUT/'report/summary.json').read_text());parts=[]
 for title,values in [('全276',r['datasets']['meiji_4cam']['all']),('参照・pilot未使用224',r['holdout']['all'])]:
  parts += [f'## {title}','|条件|TP|FN|FP|TN|採用|正しい推定の採用率|正解候補がある画像|','|---|---:|---:|---:|---:|---:|---:|---:|']
  for m in MODES:
   v=values[m];parts.append(f'|{NAMES[m]}|{v["TP"]}|{v["FN"]}|{v["FP"]}|{v["TN"]}|{v["accepted"]}|{f(100*v["correct_acceptance_rate"] if v["correct_acceptance_rate"] is not None else None,1)}%|{v["correct_candidate_available"]}|')
 parts+=['## Camera別（TP / FN / FP / TN）','|camera|'+'|'.join(NAMES[m] for m in MODES)+'|','|---|'+'---|'*len(MODES)]
 for cam in ('cam0','cam1','cam2','cam3'):parts.append('|'+cam+'|'+'|'.join(confusion(r['datasets']['meiji_4cam'][cam][m]) for m in MODES)+'|')
 parts+=['## 成立・採用（成立 / 採用）','|入力|'+'|'.join(NAMES[m] for m in MODES)+'|','|---|'+'---|'*len(MODES)]
 for ds in ('tennis_court_detector_val','synthetic_court_test','meiji_3cam','tennis_clip'):
  parts.append('|'+ds+'|'+'|'.join(str(r['datasets'][ds]['all'][m]['geometry_valid'])+' / '+str(r['datasets'][ds]['all'][m]['accepted']) for m in MODES)+'|')
 parts+=['## 同じGT点での誤差比較','|入力・条件|共通画像|旧mean px|新mean px|成立喪失/獲得|採用喪失/獲得|','|---|---:|---:|---:|---:|---:|']
 for ds in ('tennis_court_detector_val','synthetic_court_test'):
  for m in MODES[1:]:
   v=r['regression'][ds][m];parts.append(f'|{ds} {NAMES[m]}|{v["paired_geometry_cases"]}|{f(v["legacy_common_error_px"]["mean"],4)}|{f(v["new_common_error_px"]["mean"],4)}|{v["lost_geometry"]}/{v["gained_geometry"]}|{v["lost_acceptance"]}/{v["gained_acceptance"]}|')
 parts+=['## camera別の誤差・ばらつき（frame mean中央値 / 可視KPのclip変位p95、px）','|camera|'+'|'.join(NAMES[m] for m in MODES)+'|','|---|'+'---|'*len(MODES)]
 for cam in ('cam0','cam1','cam2','cam3'):
  values=r['datasets']['meiji_4cam'][cam];parts.append('|'+cam+'|'+'|'.join(f(values[m]['frame_mean_error_px']['median'])+' / '+f((values[m]['clip_dispersion_reference_visible_px'] or {}).get('p95')) for m in MODES)+'|')
 parts+=['## 旧正解候補184画像の救済','|条件|救済|順位負け|全正解gate拒否|今の候補に正解なし|','|---|---:|---:|---:|---:|']
 for m in MODES:
  v=r['frozen_184_pool_rescue']['all'][m];parts.append(f'|{NAMES[m]}|{v["recovered"]}|{v["ranking"]}|{v["gate"]}|{v["no_correct_candidate"]}|')
 lines=[]
 for part in parts:
  if part.startswith('## '):lines.extend(['',part,''])
  else:lines.append(part)
 (OUT/'report/tables.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
