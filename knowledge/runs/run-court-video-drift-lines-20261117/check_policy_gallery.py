"""Check current policy, source-frame labels and unchanged native crops."""
import argparse
import importlib.util
from pathlib import Path
import subprocess

path=Path(__file__).with_name('check_gallery.py')
spec=importlib.util.spec_from_file_location('previous_gallery_check',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
script=previous.SCRIPT
script=script.replace("data.find(r=>r.status==='warning')","data.find(r=>r.reason==='drift_unobservable')")
script=script.replace("currentFrame.report.status==='warning'","currentFrame.report.status==='warning' && currentFrame.report.measurement_status==='failed'")
script=script.replace("$('details').textContent.includes('camera_moved_possible')","$('details').textContent.includes('drift_unobservable') && $('details').textContent.includes('warn')")
script=script.replace("final.status!=='failed'","final.status!=='warning'")
needle="  const initial=await evaluate"
extra=r'''
  if(!await evaluate("data.every(r=>r.status!=='failed') && data.filter(r=>r.status==='warning').length===159 && document.body.textContent.includes('3clip×4camera') && document.body.textContent.includes('自動では検出されないことがあります')"))throw new Error('Policy or review-scope mismatch');
'''
script=script.replace(needle,extra+needle)
script=script.replace("result:'passed',gpu_disabled:true","result:'passed',gpu_disabled:true,unobservable_policy:'warn',human_review_scope:'3 clips x 4 cameras + native crops'")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chrome',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    subprocess.run(['node','-e',script,str(args.chrome),(args.output/'review/index.html').resolve().as_uri(),str(args.output.resolve())],check=True,timeout=50)
