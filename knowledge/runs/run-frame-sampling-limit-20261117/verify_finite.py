import json,hashlib
from pathlib import Path
p=Path(__file__).with_name('benchmark_limit.py')
ns={'__file__':str(p),'__name__':'validation_setup'}
exec(compile(p.read_text().split('records=[]',1)[0],str(p),'exec'),ns)
prior=json.loads(p.with_name('results.json').read_text())
def ok(value):
    if isinstance(value,dict):return value.get('all_finite',True) is not False and all(ok(v) for v in value.values())
    if isinstance(value,list):return all(ok(v) for v in value)
    return True
rows=[]
for row in prior['records']:
    output=ns['scopes'][row['case']][1](row['cap'])
    sha,doc=ns['summarize'](output)
    assert ok(doc) and sha==row['output_sha256']
    rows.append({'case':row['case'],'cap':row['cap'],'all_arrays_finite':True,'output_sha256':sha})
p.with_name('numerical-validation.json').write_text(json.dumps(rows,indent=2)+'\n')
