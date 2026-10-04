import {trajectorySegments,reasonName,frameRows,frameFromTime,videoTimeForFrame} from './model.mjs';

const data=JSON.parse(document.getElementById('review-data').textContent);
const $=id=>document.getElementById(id);
const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors=['#ed8151','#4bb985','#9b70d7','#4c92d5'];
const statusLabels={rendered:'描画可能',unsupported:'履歴schema / reader対象外',missing:'未生成',stale:'古い依存',unavailable:'RGB不在 / 図未描画'};
const yes=(value,empty='無効')=>`<span class="${value?'valid':'rejected'}">${value?'有効':empty}</span>`;
let frame=0,timer=null,request=0;

$('snapshot-path').textContent=data.index_path;
const counts=data.nodes.reduce((result,node)=>{result[node.status]=(result[node.status]??0)+1;return result;},{});
$('snapshot-summary').innerHTML=[
  [`${data.sources.length} camera / ${data.frames}f`,data.clip_id,false],
  [`${data.nodes.filter(n=>n.artifact_id).length} 採用`,`${counts.rendered??0} 描画可能 · ${counts.unsupported??0} 履歴schema`,false],
  [`${counts.missing??0} 未生成`,`${counts.stale??0} 古い依存（保存index内の比較）`,!!counts.missing],
  [data.export.status==='verified'?'統合exportあり':'統合exportなし',data.export.message,data.export.status!=='verified']
].map(([title,body,warn])=>`<div class="summary-card ${warn?'warn':''}"><strong>${escape(title)}</strong>${escape(body)}</div>`).join('');
$('source-table').innerHTML=`<table><tr><th>camera</th><th>source / 採用source hash</th><th>保存時間軸</th><th>RGB</th></tr>${data.sources.map(v=>`<tr><td>${escape(v.camera_id)}</td><td>${escape(v.path)}<br><code>${escape(v.sha256)}</code></td><td>${v.num_frames}f / ${v.fps} fps<br>${v.width} × ${v.height}</td><td>${v.checksum_verified?'checksum確認済み':'未保存 / 不在'}</td></tr>`).join('')}</table><p class="small">index SHA256: ${escape(data.index_sha256)}<br>source identity: ${escape(data.source_sha256)}</p>`;
$('artifact-table').innerHTML=`<table><tr><th>component</th><th>状態 / 保存schema</th><th>採用artifact / 出自</th><th>入力依存（port → producer / artifact）</th></tr>${data.nodes.map(n=>`<tr><td><a href="#${escape(n.node.replaceAll('/','-'))}">${escape(n.node)}</a></td><td><span class="state ${n.status}">${escape(statusLabels[n.status])}</span><br>${n.schema?escape(`${n.schema} v${n.version}`):'未保存'}<br>${escape((n.reasons??[]).join('; '))}</td><td>${n.artifact_id?`<details><summary><code>${escape(n.artifact_id.slice(0,12))}</code> · ${escape(n.origin.origin)}</summary><code>${escape(n.artifact_id)}</code><p>${escape(n.path)}</p><p>SHA256 ${escape(n.sha256)}</p><pre>${escape(JSON.stringify(n.origin,null,2))}</pre></details>`:'—'}</td><td>${n.dependencies.length?n.dependencies.map(d=>`${escape(d.port)} → ${d.producer?`<a href="#${escape(d.producer.replaceAll('/','-'))}">${escape(d.producer)}</a>`:'<span class="rejected">採用版に存在しない</span>'}<br><code title="${escape(d.artifact_id)}">${escape(d.artifact_id.slice(0,12))}</code>`).join('<br>'):'入力artifactなし'}</td></tr>`).join('')}</table>`;
$('scene-note').textContent=data.scene_reason;
$('scene-status').textContent=data.scene?`SceneResult v2 · ${data.scene.status}`:'3D / mask未保存';
$('frame-number').max=$('frame-slider').max=String(data.frames-1);
$('media-note').textContent=data.online?'全source frameのRGBをCPUで読取。時刻表示は保存FPSによる目安で、動画PTSではありません。':'静的gallery: 撮影済みsample frameのみRGBを表示。任意frameのRGBには --serve を使います。';

const viewCanvases=new Map();
for(const v of data.sources){
  const tile=document.createElement('div');tile.className='rgb-tile';
  tile.innerHTML=`<div class="view-label">${escape(v.camera_id)} · source frame <span class="rgb-frame">0</span></div><canvas width="960" height="540"></canvas><div class="rgb-error"></div>`;
  $('rgb-views').append(tile);
  const canvas=tile.querySelector('canvas');viewCanvases.set(v.camera_id,{canvas,tile});
  canvas.addEventListener('click',()=>{if(data.online)window.open(`/api/frame?camera=${encodeURIComponent(v.camera_id)}&frame=${frame}`,'_blank','noopener');else if(data.samples[v.camera_id][frame])window.open(data.samples[v.camera_id][frame],'_blank','noopener');});
}

function line(ctx,a,b,color,width=2){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
function dot(ctx,p,color,r=3,outline=false){ctx.beginPath();ctx.arc(...p,r,0,Math.PI*2);if(outline){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}else{ctx.fillStyle=color;ctx.fill();}}
function project3D(point,fit,v){
  const camera=fit.R.map((row,i)=>row.reduce((n,x,j)=>n+x*point[j],fit.t[i]));
  if(camera[2]<=1e-6)return null;
  const pixel=fit.K.map(row=>row.reduce((n,x,j)=>n+x*camera[j],0));
  return [pixel[0]/pixel[2]/v.width*960,pixel[1]/pixel[2]/v.height*540];
}
function drawOverlay(ctx,camera,f){
  if(!data.scene){
    if(!$('show-2d').checked)return;
    const saved=data.observations[camera],v=data.sources.find(s=>s.camera_id===camera),uv=p=>[p[0]/v.width*960,p[1]/v.height*540];
    if(saved.pose){const pose=saved.pose;for(let p=0;p<pose.local_track_ids.length;p++)if(pose.observed[f][p]){
      const kp=pose.uv_px[f][p],conf=pose.confidence[f][p],color=colors[p%colors.length];
      for(const [i,j] of data.skeleton)if(conf[i]>.15&&conf[j]>.15)line(ctx,uv(kp[i]),uv(kp[j]),color,2);
      for(let j=0;j<17;j++)if(conf[j]>.15)dot(ctx,uv(kp[j]),color,2);
    }}
    if(saved.ball&&saved.ball.point_kind[f])dot(ctx,uv(saved.ball.uv_px[f]),saved.ball.observed[f]?'#ffd03e':'#ed8151',3,!saved.ball.observed[f]);
    return;
  }
  const a=data.scene.arrays,c=data.scene.camera_ids.indexOf(camera),v=data.sources.find(s=>s.camera_id===camera);
  if(c<0)return;
  const uv=p=>[p[0]*960,p[1]*540];
  if($('show-2d').checked){
    for(let j=0;j<a.court_kp[c][f].length;j++)if(a.court_vis[c][f][j]>0)dot(ctx,uv(a.court_kp[c][f][j]),'#c2e6d5',2);
    for(let p=0;p<data.scene.player_ids.length;p++){
      if(!a.player_observed[p][f])continue;
      const kp=a.human_kp_2d[p][c][f],vis=a.human_kp_vis[p][c][f],color=colors[p%colors.length];
      for(const [i,j] of data.scene.skeleton)if(vis[i]>0&&vis[j]>0)line(ctx,uv(kp[i]),uv(kp[j]),color,2);
      for(let j=0;j<kp.length;j++)if(vis[j]>0)dot(ctx,uv(kp[j]),color,2);
    }
    if(a.ball_vis[c][f])dot(ctx,uv(a.ball_uv[c][f]),'#ffd03e',3);
  }
  if($('show-reprojection').checked){
    const fitIndex=data.scene.reference?.camera_ids.indexOf(camera)??-1;
    const fit=fitIndex<0?null:data.scene.reference.camera_fits[fitIndex];if(!fit)return;
    for(let p=0;p<data.scene.player_ids.length;p++)for(let j=0;j<17;j++)if(a.player_kp_3d_vis[p][f][j]){
      const pixel=project3D(a.player_kp_3d[p][f][j],fit,v);if(pixel)dot(ctx,pixel,'#52acff',3,true);
    }
    if(a.ball_3d_valid[f]){const pixel=project3D(a.ball_3d[f],fit,v);if(pixel)dot(ctx,pixel,'#52acff',5,true);}
  }
}
async function drawRGB(f,token){
  await Promise.all(data.sources.map(async v=>{
    const {canvas,tile}=viewCanvases.get(v.camera_id),ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,960,540);tile.querySelector('.rgb-frame').textContent=f;
    const note=tile.querySelector('.rgb-error');note.textContent='RGB読取中…';
    if(!v.available){note.textContent='元RGB未保存 / 不在';return;}
    const src=data.online?`/api/frame?camera=${encodeURIComponent(v.camera_id)}&frame=${f}`:data.samples[v.camera_id][f];
    if(!src){note.textContent='このframeのRGBは静的galleryに未収録（--serveで確認）';return;}
    const img=new Image();
    await new Promise(resolve=>{img.onload=()=>{if(token===request){ctx.drawImage(img,0,0,960,540);drawOverlay(ctx,v.camera_id,f);note.textContent='';canvas.dataset.frame=String(f);}resolve();};img.onerror=()=>{if(token===request)note.textContent='RGB読取失敗。source/index変更やdecodeを確認してください。';resolve();};img.src=src;});
  }));
}
function drawCourt(f){
  const canvas=$('court-scene'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,440,390);
  if(!data.scene){ctx.fillStyle='#637983';ctx.font='16px system-ui';ctx.fillText('保存3D・品質maskなし',120,185);return;}
  const oblique=$('scene-view').value==='oblique';
  const project=p=>oblique?[220+(p[0]-.36*p[1])*12,195+p[1]*8-p[2]*18]:[220+p[0]*16,195+p[1]*13];
  const court=[[-5.485,-11.885,0],[5.485,-11.885,0],[5.485,11.885,0],[-5.485,11.885,0],[-5.485,-11.885,0]];
  for(let i=1;i<court.length;i++)line(ctx,project(court[i-1]),project(court[i]),'#98b1a8',1);
  line(ctx,project([-5.485,0,0]),project([5.485,0,0]),'#739289',2);
  const a=data.scene.arrays,start=Math.max(0,f-59);
  function trail(points,valid,color){
    for(const segment of trajectorySegments(points,valid,start,f+1)){
      for(let i=1;i<segment.length;i++)line(ctx,project(segment[i-1].point),project(segment[i].point),color,1);
      if(segment.length===1)dot(ctx,project(segment[0].point),color,2);
    }
  }
  for(let p=0;p<data.scene.player_ids.length;p++){
    const color=colors[p%colors.length];trail(a.player_position[p],a.player_valid[p],color);
    for(const [i,j] of data.scene.skeleton)if(a.player_kp_3d_vis[p][f][i]&&a.player_kp_3d_vis[p][f][j])line(ctx,project(a.player_kp_3d[p][f][i]),project(a.player_kp_3d[p][f][j]),color,2);
    if(a.player_valid[p][f])dot(ctx,project(a.player_position[p][f]),color,4);
  }
  trail(a.ball_3d,a.ball_3d_valid,'#b48b12');if(a.ball_3d_valid[f])dot(ctx,project(a.ball_3d[f]),'#e8ba26',4);
  ctx.font='12px system-ui';ctx.fillStyle='#456259';ctx.fillText(`court m · source frame ${f}`,12,18);
  ctx.fillText(oblique?'x / y / height':'x / y · heightは斜め表示へ',12,376);
}
function drawQuality(f){
  const rows=frameRows(data.scene,f);
  if(!rows.length){$('frame-quality').innerHTML='<p class="notice">3D・品質mask・棄却理由は未保存。2D task結果は下のcomponent galleryから確認できます。</p>';return;}
  $('frame-quality').innerHTML=`<table><tr><th>対象</th><th>2D観測</th><th>3D root / 球</th><th>heading / mesh</th><th>有効3D関節</th><th>保存棄却理由</th></tr>${rows.map(r=>`<tr><th>${escape(r.name)}</th><td>${r.name==='ball'?`${r.observed} view`:yes(r.observed,'未観測')}</td><td>${yes(r.valid)}</td><td>${r.name==='ball'?'—':`${yes(r.heading)} / ${yes(r.mesh)}`}</td><td>${r.name==='ball'?'—':`${r.joints}/17`}</td><td class="${r.valid?'valid':'rejected'}">${escape(r.reason)}${r.jointReasons?`<br><span class="small">関節: ${escape(r.jointReasons)}</span>`:''}</td></tr>`).join('')}</table>`;
}
const timelineCanvases=[];
if(data.scene){
  const a=data.scene.arrays;
  for(const [label,mask] of [...a.player_observed.map((m,p)=>[`P${data.scene.player_ids[p]} 2D観測`,m]),...a.player_valid.map((m,p)=>[`P${data.scene.player_ids[p]} root`,m]),['ball 3D',a.ball_3d_valid]]){
    const row=document.createElement('div');row.className='timeline-row';const labelEl=document.createElement('span');labelEl.textContent=label;const canvas=document.createElement('canvas');canvas.width=data.frames;canvas.height=15;row.append(labelEl,canvas);$('quality-timeline').append(row);timelineCanvases.push({canvas,mask});canvas.addEventListener('click',e=>setFrame(Math.floor((e.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*data.frames)));
  }
  for(const gap of data.scene.gaps){const b=document.createElement('button');b.textContent=`${gap.name} [${gap.start}, ${gap.end})`;b.dataset.start=gap.start;b.dataset.end=gap.end;b.addEventListener('click',()=>setFrame(gap.start));$('gap-buttons').append(b);}
  const fits=data.scene.reference?.camera_fits??[];
  $('calibration-details').innerHTML=fits.map((fit,i)=>`${escape(data.scene.reference.camera_ids[i])}: frame ${fit.calibration_frame_index} / RMSE ${fit.rmse_px.toFixed(2)} px`).join('<br>')+'<br>RMSEは保存平面fitの指標。独立3D精度ではありません。';
}
function drawTimeline(f){for(const {canvas,mask} of timelineCanvases){const ctx=canvas.getContext('2d');ctx.fillStyle='#c5573d';ctx.fillRect(0,0,canvas.width,15);ctx.fillStyle='#439d74';for(const segment of trajectorySegments(mask,mask))ctx.fillRect(segment[0].frame,0,segment.length,15);ctx.fillStyle='#162f3c';ctx.fillRect(f,0,2,15);}}

const movies=[...document.querySelectorAll('video')];let videoSync=false;
function syncMovies(f,leader=null){
  videoSync=true;
  for(const video of movies)if(video!==leader&&video.readyState>=1&&Math.abs(video.currentTime*data.fps-f-.5)>.75)video.currentTime=videoTimeForFrame(f,data.fps);
  videoSync=false;
}
for(const video of movies){video.addEventListener('play',()=>{if(timer){clearInterval(timer);timer=null;$('play-frames').textContent='▶ 4 fpsで確認';}for(const other of movies)if(other!==video)other.pause();});for(const event of ['timeupdate','seeked'])video.addEventListener(event,()=>{const next=frameFromTime(video.currentTime,data.fps,data.frames);if(!videoSync&&next!==frame&&(event==='seeked'||!video.paused))setFrame(next,video);});}
if(movies.length)$('media-note').textContent+=' component動画も同じ保存FPSのframeへ連動します。動画再生は選択した1本が基準です。';
async function setFrame(value,leader=null){
  frame=Math.max(0,Math.min(data.frames-1,Math.trunc(Number(value)||0)));const token=++request;
  $('frame-slider').value=$('frame-number').value=String(frame);$('frame-time').textContent=`/ ${data.frames-1} · ${(frame/data.fps).toFixed(3)} s（FPS目安）`;
  for(const b of $('gap-buttons').children)b.classList.toggle('active',frame>=Number(b.dataset.start)&&frame<Number(b.dataset.end));
  drawCourt(frame);drawQuality(frame);drawTimeline(frame);syncMovies(frame,leader);await drawRGB(frame,token);
}
$('frame-number').addEventListener('change',e=>setFrame(e.target.value));$('frame-slider').addEventListener('input',e=>setFrame(e.target.value));
$('previous-frame').addEventListener('click',()=>setFrame(frame-1));$('next-frame').addEventListener('click',()=>setFrame(frame+1));
for(const id of ['show-2d','show-reprojection','scene-view'])$(id).addEventListener('change',()=>setFrame(frame));
$('play-frames').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;$('play-frames').textContent='▶ 4 fpsで確認';}else{for(const v of movies)v.pause();$('play-frames').textContent='■ 停止';timer=setInterval(()=>{if(frame>=data.frames-1){clearInterval(timer);timer=null;$('play-frames').textContent='▶ 4 fpsで確認';}else setFrame(frame+1);},250);}});
window.addEventListener('pagehide',()=>{if(timer)clearInterval(timer);});
setFrame(0);
