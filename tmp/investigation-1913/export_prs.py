import hashlib,json,subprocess
from pathlib import Path
import yaml
root=Path('tmp/investigation-1913')
allprs={}
for n in [1805,1831,1910,1489,1897,1858,1855,1892,1896,1903]:
 p=json.loads((root/f'github/pr-{n}.json').read_text())
 head=p['head_sha'];base=p['base_sha']
 paths=subprocess.check_output(['git','diff','--name-only',base+'...'+head],text=True).splitlines()
 files=[];nodes=[]
 for path in paths:
  b=subprocess.run(['git','show',head+':'+path],capture_output=True,check=True).stdout
  dest=root/f'prs/{n}'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  files.append(dict(path=path,size=len(b),sha256=hashlib.sha256(b).hexdigest(),source=f'https://github.com/Motoki0705/tennis-lab/blob/{head}/{path}'))
  if path.startswith('knowledge/nodes/') and path.endswith('.md'):
   s=b.decode();fm=yaml.safe_load(s.split('---',2)[1]);nodes.append(dict(path=path,frontmatter=fm))
 (root/f'github/pr-{n}.diff').write_bytes(subprocess.check_output(['git','diff',base+'...'+head]))
 allprs[str(n)]=dict(head=head,base=base,state=p['state'],draft=p['draft'],merged=p['merged'],files=files,nodes=nodes,non_knowledge=[q for q in paths if not q.startswith('knowledge/')])
 print(n,'files',len(files),'nodes',len(nodes),'non-knowledge',allprs[str(n)]['non_knowledge'])
(root/'analysis/pr-inventory.json').write_text(json.dumps(allprs,ensure_ascii=False,indent=2,default=str))
main=json.loads((root/'github/main.json').read_text())['commit']['sha']
for path in ['knowledge/nodes/ball_detection/.sequence','knowledge/nodes/court_detection/.sequence','knowledge/summary.md']:
 b=subprocess.check_output(['git','show',main+':'+path]);dest=root/'main'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 print('main',path,len(b),b.decode().strip() if path.endswith('.sequence') else '')
