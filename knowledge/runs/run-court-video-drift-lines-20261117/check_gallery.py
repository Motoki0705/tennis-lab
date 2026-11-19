"""Exercise native frame selection, per-frame table and diagnostic overlays."""
import argparse
import importlib.util
from pathlib import Path
import subprocess

PREVIOUS=Path('knowledge/runs/run-court-video-shared-acceptance-20261117/check_gallery.py')
spec=importlib.util.spec_from_file_location('previous_browser_check',PREVIOUS)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
SCRIPT=previous.SCRIPT
start=SCRIPT.index('  const initial=')
end=SCRIPT.index('}finally{',start)
SCRIPT=SCRIPT[:start]+r'''
  const initial=await evaluate("({rows:data.length,cameras:$('camera').options.length,clips:$('clip').options.length,width:$('canvas').width,height:$('canvas').height,frames:$('frame').options.length,table:$('table').rows.length})");
  if(initial.rows!==276||initial.cameras!==4||initial.clips!==69||initial.width!==1920||initial.height!==1080||initial.frames!==6||initial.table!==5)throw new Error(JSON.stringify(initial));
  await evaluate("{const row=data.find(r=>r.status==='warning');$('camera').value=row.camera;$('clip').value=row.clip_id;load();}");
  await until("currentImage && current.status==='warning' && currentFrame.report.status==='warning'");
  await evaluate("$('zoom').value='2';$('reference').checked=true;paint()");
  if(!await evaluate("$('canvas').style.width==='3840px' && $('details').textContent.includes('camera_moved_possible')"))throw new Error('Warning/zoom not visible');
  await evaluate("$('zoom').value='0.5';paint()");
  const warning=await evaluate("({camera:current.camera,clip:current.clip_id,status:current.status,frame:currentFrame.index,source:currentImage.src})");
  const screenshot=await call('Page.captureScreenshot',{format:'png'});
  fs.writeFileSync(path.join(output,'gallery-warning.png'),Buffer.from(screenshot.data,'base64'));
  await evaluate("$('frame').value='0';frame()");
  await until("currentImage && currentFrame.index===0");
  if(!await evaluate("$('status').textContent.includes('目視のみ')"))throw new Error('Frame zero mislabeled as measured');
  await evaluate("$('camera').value='cam0';load();$('camera').value='cam3';load();$('clip').selectedIndex=68;load();");
  await until("currentImage && current.camera==='cam3' && current.clip_id.endsWith('clip_068') && currentImage.src.includes('/cam3/clip_068/')");
  const final=await evaluate("({status:current.status,crops:document.querySelectorAll('#crops img').length,source:currentImage.src,frame:currentFrame.index,worst:current.worst_frame_index})");
  if(final.status!=='failed'||final.crops!==4||final.frame!==final.worst)throw new Error(JSON.stringify(final));
  await evaluate("$('table').rows[0].click()");
  await until("currentImage && String(currentFrame.index)===$('table').rows[0].dataset.frame");
  fs.writeFileSync(path.join(output,'gallery-browser-check.json'),JSON.stringify({initial,warning,final,result:'passed',gpu_disabled:true},null,2)+'\n');
  console.log(JSON.stringify({initial,warning,final,result:'passed'}));
''' + SCRIPT[end:]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chrome',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    subprocess.run(['node','-e',SCRIPT,str(args.chrome),(args.output/'review/index.html').resolve().as_uri(),str(args.output.resolve())],check=True,timeout=50)
