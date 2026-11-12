import base64,hashlib,json
from pathlib import Path
root=Path('tmp/investigation-1913');entries=[]
for p in sorted((root/'_responses').glob('*.json')):
 x=json.loads(p.read_text());b=base64.b64decode(x.pop('b64_string'),validate=True)
 target=root/x['local_path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 x['size']=len(b);x['sha256']=hashlib.sha256(b).hexdigest();x['md5']=hashlib.md5(b).hexdigest()
 if x['source_size'] is not None:assert x['source_size']==len(b),(p,x['source_size'],len(b))
 if x['expected_sha256']:assert x['sha256']==x['expected_sha256'],p
 entries.append(x)
(root/'evidence/index.json').write_text(json.dumps({'schema_version':1,'files':entries},ensure_ascii=False,indent=2)+'\n')
print('decoded',len(entries),'files',sum(x['size'] for x in entries),'bytes')
