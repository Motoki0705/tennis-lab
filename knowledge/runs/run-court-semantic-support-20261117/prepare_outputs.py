"""Prepare only this run's local report paths; frozen source assets stay in run73."""
from pathlib import Path
import json

OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')
PREVIOUS=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')

def main():
 data=(PREVIOUS/'manifest.json').read_bytes();records=json.loads(data)['records'];assert len(records)==3545
 OUT.mkdir(parents=True,exist_ok=True);manifest=OUT/'manifest.json'
 if manifest.exists():assert manifest.read_bytes()==data
 else:manifest.write_bytes(data)
 (OUT/'evaluation').mkdir(exist_ok=True)
 for dataset in sorted({r['dataset'] for r in records}):
  dest=OUT/'evaluation'/dataset;target=(OUT/('meiji4' if dataset=='meiji_4cam' else 'regression')/dataset).resolve()
  if dest.exists() or dest.is_symlink():assert dest.is_symlink() and dest.resolve()==target
  else:dest.symlink_to(target,target_is_directory=True)
if __name__=='__main__':main()
