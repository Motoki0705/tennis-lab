"""Exercise the generated local gallery in an installed CPU headless browser."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

SCRIPT = r'''
const {spawn} = require('node:child_process');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const {once} = require('node:events');
const [chrome, url, output] = process.argv.slice(1);
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'court-review-'));
const browser=spawn(chrome,['--no-sandbox','--disable-gpu','--use-gl=disabled','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;
(async()=>{try{
  let stderr='';
  const port=await new Promise((resolve,reject)=>{
    browser.on('error',reject); browser.on('exit',code=>reject(new Error('browser exited '+code)));
    browser.stderr.on('data',chunk=>{stderr+=chunk.toString();const match=stderr.match(/ws:\/\/127\.0\.0\.1:(\d+)\//);if(match)resolve(match[1]);});
  });
  const pages=await (await fetch('http://127.0.0.1:'+port+'/json/list')).json();
  socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
  await once(socket,'open');let serial=0;const pending=new Map();
  socket.addEventListener('message',event=>{const v=JSON.parse(event.data);if(pending.has(v.id)){const [resolve,reject]=pending.get(v.id);pending.delete(v.id);v.error?reject(new Error(JSON.stringify(v.error))):resolve(v.result);}});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,[resolve,reject]);socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
  const until=async expression=>{for(let i=0;i<300;i++){try{if(await evaluate(expression))return;}catch{}await new Promise(r=>setTimeout(r,50));}throw new Error('Timed out: '+expression);};
  await call('Emulation.setDeviceMetricsOverride',{width:1280,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url});
  await until("typeof currentImage !== 'undefined' && currentImage && currentImage.complete");
  const initial=await evaluate("({rows:data.length,cameras:$('camera').options.length,clips:$('clip').options.length,width:$('canvas').width,height:$('canvas').height})");
  if(initial.rows!==276||initial.cameras!==4||initial.clips!==69||initial.width!==1920||initial.height!==1080)throw new Error(JSON.stringify(initial));
  await evaluate("{const row=data.find(r=>r.status==='warning');$('camera').value=row.camera;$('clip').value=row.clip_id;$('frame').value='heldout';load();}");
  await until("currentImage && current.status==='warning' && currentImage.src.endsWith('/heldout/source.jpg')");
  await evaluate("$('zoom').value='2';$('reference').checked=true;paint()");
  if(!await evaluate("$('canvas').style.width==='3840px' && $('details').textContent.includes('camera_moved_possible')"))throw new Error('Warning/zoom not visible');
  await evaluate("$('zoom').value='0.5';paint()");
  const warning=await evaluate("({camera:current.camera,clip:current.clip_id,status:current.status,source:currentImage.src})");
  const screenshot=await call('Page.captureScreenshot',{format:'png'});
  fs.writeFileSync(path.join(output,'gallery-warning.png'),Buffer.from(screenshot.data,'base64'));
  await evaluate("$('camera').value='cam0';load();$('camera').value='cam3';load();$('clip').selectedIndex=68;load();");
  await until("currentImage && current.camera==='cam3' && current.clip_id.endsWith('clip_068') && currentImage.src.endsWith('/cam3/clip_068/heldout/source.jpg')");
  const final=await evaluate("({status:current.status,crops:document.querySelectorAll('#crops img').length,source:currentImage.src})");
  if(final.status!=='failed'||final.crops!==4)throw new Error(JSON.stringify(final));
  fs.writeFileSync(path.join(output,'gallery-browser-check.json'),JSON.stringify({initial,warning,final,result:'passed',gpu_disabled:true},null,2)+'\n');
  console.log(JSON.stringify({initial,warning,final,result:'passed'}));
}finally{if(socket)socket.close();browser.kill('SIGTERM');await once(browser,'exit').catch(()=>{});fs.rmSync(profile,{recursive:true,force:true});}})().catch(error=>{console.error(error);process.exitCode=1});
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrome', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    process = subprocess.Popen(['node', '-e', SCRIPT, str(args.chrome), (args.output / 'review/index.html').resolve().as_uri(),
                                str(args.output.resolve())], start_new_session=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        raise
    print(stdout, end=''); print(stderr, end='')
    if process.returncode:
        raise SystemExit(process.returncode)
