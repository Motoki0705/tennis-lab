const $ = id => document.getElementById(id);
const labels = {not_saved:'採点frame ID・frame別costは未保存',not_sampled:'sample対象外',duplicate_removed:'保存mask上で重複除外',fewer_than_two_views:'2 view未満で未採点',scored:'保存済み採点frame'};
const kinds = {unresolved:'未解決（存在不明）',observed:'観測点',interpolated:'補間点・根拠外',occlusion_estimated:'遮蔽推定・根拠外'};
const colors = ['#257e9b','#c78a32','#9b77b6','#899aa6'];
let catalog, current, timeline, snapshot, requestId=0, selected=0, playing=false, centers=new Map();
const fmt = v => v === null || v === undefined ? '未保存' : v.toFixed(3);
const turns = values => values.map(v=>v?'180°':'0°').join(' / ');
const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path){ const r=await fetch(path); if(!r.ok){const j=await r.json();throw new Error(j.detail || `HTTP ${r.status}`);}return r.json(); }
function error(e){$('error').hidden=false;$('error').textContent=e.message;playing=false;$('play').textContent='再生';}
function pointText(point){if(!point)return '観測artifactなし';if(!point.uv_px)return kinds[point.kind];const [x,y]=point.uv_px;return `${kinds[point.kind]}${point.in_frame?'':'・画像外'}<br>(${x.toFixed(1)}, ${y.toFixed(1)}) px<br>保存値 ${point.confidence.toFixed(3)}`;}
function renderDecision(){
 const d=current.decision, h=d?.hypotheses;
 $('decision').textContent=d ? d.decided?'保存実験で採用':'保存実験で停止 · '+d.reason:'side判定 未保存';
 $('notice').classList.toggle('good',!!d?.decided);
 const src=Object.values(current.point_sources).filter(Boolean);
 const imported=src.some(s=>s.semantics==='annotation_acceptance_not_probability');
 $('notice').textContent=imported?'注釈インポートの観測点。モデル支援レビューを含み、独立したhuman GTではありません。採用は保存実験の判定です。補間・遮蔽推定は根拠から区別します。':src.some(s=>s.schema==='ball_points v2')?'現行ball_points v2：全frameの最大weight成分平均点。presenceは診断値で、観測の除外に使いません。表示判定は保存artifactの結果です。':'保存済みdetector経路の観測。過去の暫定モデル・gate設定によるartifactで、現行refinerの成績・現在の本番選択を示しません。';
 $('metrics').innerHTML=`<div><strong>${fmt(d?.margin)}</strong><span>margin</span></div><div><strong>${d?.frames??'—'}</strong><span>採点frame</span></div><div><strong>${h?fmt(h[0].support):'—'}</strong><span>最良support率</span></div>`;
 $('threshold').textContent=d?.thresholds?`保存閾値: margin ≥ ${d.thresholds.min_margin} / cost ≤ ${d.thresholds.max_cost} / support ≥ ${d.thresholds.min_support} / 最低 ${d.thresholds.min_frames} frame`:'判定閾値・採点frame IDはstoreに未保存。現在の設定値を過去artifactに代入しません。';
 $('source').textContent=`${current.clip} · ${current.cameras.length} view · ${current.frames} frame · ${current.fps.toFixed(2)} fps`;
 $('timeline-note').textContent=current.frame_scores_saved?'各cameraの帯は保存観測mask。点はCSVに保存された採点frameのcost（0–1）、欠測を跨いで結びません。クリックで同時刻を開きます。':'各cameraの帯は保存観測mask。採点frame ID・frame別costが未保存のため、支持frameとして表示しません。';
 $('timeline-legend').innerHTML=current.frame_scores_saved?`<span class="badge-best">H1 最良</span> / <span class="badge-other">H2 次点</span> / 薄色 H3・H4 · cost ↓`:'観測あり ／ 観測なしは空白';
 $('provenance').textContent=JSON.stringify({store:current.store,index_sha256:current.index_sha256,source_sha256:current.source_sha256,point_sources:current.point_sources,diagnostic_sources:current.diagnostic_sources,calibration:current.calibration},null,2);
 $('calibration').textContent=current.calibration?`reference ${current.calibration.reference_camera} · frame 0からの近似pinhole校正 / RMSE: `+current.calibration.views.map(v=>`${v.camera_id} ${v.rmse_px.toFixed(2)} px`).join(' / ')+(Object.keys(current.calibration.excluded).length?' / 校正除外: '+Object.keys(current.calibration.excluded).join(', '):''):'校正artifact 未保存';
}
function renderHypotheses(){
 const d=current.decision, h=d?.hypotheses;
 if(!h){$('hypotheses').innerHTML=`<div class="blank">仮説cost・supportは未保存${d?.view_half_turns?'。保存side: '+esc(turns(d.view_half_turns)):''}</div>`;$('score-note').textContent='再計算は行いません。sideのみの旧schemaと、仮説診断を保存するschemaを区別します。';return;}
 const scores=snapshot.evidence.scores;
 $('hypotheses').innerHTML=`<table><thead><tr><th>仮説</th><th>${d.camera_ids.map(esc).join(' / ')}</th><th>集計cost</th><th>支持率 / 支持数換算</th><th>このframe cost</th><th>このframe support</th></tr></thead><tbody>${h.map((v,i)=>`<tr data-hypothesis="${i}" class="${i===selected?'selected':''}"><td class="hyp ${i===0?'best':i===1?'runner':''}">H${i+1}${i===0?' 最良':i===1?' 次点':''}</td><td class="turns">${turns(v.view_half_turns)}</td><td>${fmt(v.cost)}</td><td>${fmt(v.support)} · ${(v.support*v.frames).toFixed(0)}/${v.frames}</td><td>${scores?fmt(scores.costs[i]):'未保存／未採点'}</td><td>${scores?`<span class="pill ${scores.supports[i]?'support':'fail'}">${scores.supports[i]?'支持':'支持なし'}</span>`:'—'}</td></tr>`).join('')}</tbody></table>`;
 $('hypotheses').querySelectorAll('tr[data-hypothesis]').forEach(row=>row.addEventListener('click',()=>{selected=Number(row.dataset.hypothesis);renderHypotheses();drawGeometry();drawTimeline();}));
 $('score-note').textContent=`costは保存された正規化再投影cost、supportは物理条件と全観測viewの閾値を満たす割合。支持数は保存率×frame数の換算。${scores?` H2 − H1 のこのframe差: ${(scores.costs[1]-scores.costs[0]).toFixed(3)}。`:' この時刻の仮説差は保存情報がなく判定できません。'}`;
}
function drawTimeline(){
 const c=$('timeline');c.width=Math.round(c.clientWidth*devicePixelRatio);c.height=140*devicePixelRatio;const ctx=c.getContext('2d');ctx.scale(devicePixelRatio,devicePixelRatio);const w=c.clientWidth,pad=49,plot=w-pad-8;
 current.cameras.forEach((camera,i)=>{let mask=timeline.observed[camera];ctx.fillStyle='#5c7681';ctx.font='10px system-ui';ctx.fillText(camera,0,13+i*16);ctx.fillStyle='#eaf0f3';ctx.fillRect(pad,4+i*16,plot,10);if(mask){ctx.fillStyle='#6597ac';mask.forEach((v,f)=>{if(v)ctx.fillRect(pad+f/(current.frames-1)*plot,4+i*16,Math.max(1,plot/current.frames),10);});}});
 const top=current.cameras.length*16+8,height=65;
 if(timeline.evidence.length){ctx.strokeStyle='#e7edf0';ctx.beginPath();ctx.moveTo(pad,top);ctx.lineTo(w,top);ctx.moveTo(pad,top+height);ctx.lineTo(w,top+height);ctx.stroke();ctx.fillStyle='#82949c';ctx.fillText('cost 1',0,top+5);ctx.fillText('0',26,top+height);
 [2,3,1,0].forEach(i=>{if(i>=timeline.evidence[0].costs.length)return;ctx.strokeStyle=colors[i];ctx.lineWidth=i===selected?1.8:.9;ctx.globalAlpha=i<2?1:.5;timeline.evidence.forEach(row=>{ctx.beginPath();ctx.arc(pad+row.frame/(current.frames-1)*plot,top+(1-row.costs[i])*height,i===selected?1.7:1,0,Math.PI*2);ctx.fillStyle=colors[i];ctx.fill();});});ctx.globalAlpha=1;
 }else{ctx.fillStyle='#8699a1';ctx.font='11px system-ui';ctx.fillText('frame別cost・採点mask 未保存',pad,top+32);}
 ctx.strokeStyle='#183e51';ctx.lineWidth=1;const x=pad+snapshot.frame/(current.frames-1)*plot;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,top+height);ctx.stroke();ctx.font='10px system-ui';ctx.fillStyle='#183e51';ctx.fillText(`f${snapshot.frame}`,Math.min(w-45,x+4),138);
}
function drawGeometry(){
 const c=$('geometry'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
 if(!current.calibration){ctx.fillStyle='#708792';ctx.fillText('校正artifact 未保存',20,50);return;}
 const views=current.calibration.views, h=current.decision?.hypotheses, assignment=h?h[selected].view_half_turns:current.decision?.view_half_turns;
 if(!assignment){ctx.fillStyle='#708792';ctx.fillText('比較可能なside仮説 未保存',20,50);return;}
 const xmax=Math.max(10,...views.map(v=>Math.abs(v.center_m[0])+3)),ymax=Math.max(16,...views.map(v=>Math.abs(v.center_m[1])+3)),scale=Math.min(140/xmax,105/ymax);const xy=(x,y)=>[170+x*scale,123-y*scale];
 ctx.strokeStyle='#bed3d9';ctx.lineWidth=1;const [x0,y0]=xy(-5.485,11.885);ctx.strokeRect(x0,y0,10.97*scale,23.77*scale);[-11.885,-6.4,0,6.4,11.885].forEach(y=>{ctx.beginPath();ctx.moveTo(...xy(-5.485,y));ctx.lineTo(...xy(5.485,y));ctx.stroke();});ctx.fillStyle='#8b9fa7';ctx.font='10px system-ui';ctx.fillText('reference court',xy(-5.485,11.885)[0],Math.max(12,y0-5));
 const draw=(turnValues,color,offset)=>views.forEach(v=>{const i=current.decision.camera_ids.indexOf(v.camera_id),sign=turnValues[i]?-1:1;const [x,y]=xy(v.center_m[0]*sign,v.center_m[1]*sign);const length=Math.hypot(v.direction[0],v.direction[1]);const dx=v.direction[0]*sign/Math.max(.01,length)*20,dy=-v.direction[1]*sign/Math.max(.01,length)*20;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();ctx.font='10px system-ui';ctx.fillText(`${v.camera_id} ${turnValues[i]?'180°':'0°'}`,Math.min(275,Math.max(4,x+(i===1?-63:6))),Math.min(239,Math.max(14,y+offset+(offset>0?i*6:0))));});
 if(h&&h.length>1)draw(h[selected===1?0:1].view_half_turns,'#c78a32',14);draw(assignment,'#257e9b',-6);
 $('geometry-legend').innerHTML=h?`<span class="badge-best">青: H${selected+1}</span> / <span class="badge-other">橙: H${selected===1?1:2}</span> · 図からcostは計算しません`:'保存sideの校正位置・向き';
}
function mark(ctx,point,x,y,scale){if(!point?.uv_px)return;ctx.beginPath();ctx.arc(x,y,Math.max(2,3*scale),0,Math.PI*2);ctx.fillStyle=point.observed?'#ffda4a':'#bdcbd2';ctx.fill();}
function drawView(node,image,camera){
 const full=node.querySelector('.full'),crop=node.querySelector('.crop');full.width=camera.width;full.height=camera.height;let ctx=full.getContext('2d');ctx.fillStyle='#192d37';ctx.fillRect(0,0,full.width,full.height);
 if(image){ctx.drawImage(image,0,0);if(camera.point?.uv_px)mark(ctx,camera.point,...camera.point.uv_px,1);}
 else{ctx.fillStyle='#dce7eb';ctx.font='50px system-ui';ctx.fillText('RGB 未保存 / unavailable',50,full.height/2);}
 crop.width=224;crop.height=224;ctx=crop.getContext('2d');ctx.fillStyle='#192d37';ctx.fillRect(0,0,224,224);
 const center=centers.get(camera.id)??camera.point?.uv_px;
 if(image&&center){const [x,y]=center;ctx.drawImage(image,x-64,y-64,128,128,0,0,224,224);if(camera.point?.uv_px)mark(ctx,camera.point,(camera.point.uv_px[0]-x+64)*1.75,(camera.point.uv_px[1]-y+64)*1.75,1.75);}
 else{ctx.fillStyle='#bacbd3';ctx.font='19px system-ui';ctx.fillText('クリックで拡大',35,115);}
 full.onclick=e=>{const r=full.getBoundingClientRect();centers.set(camera.id,[(e.clientX-r.left)/r.width*camera.width,(e.clientY-r.top)/r.height*camera.height]);drawView(node,image,camera);};
}
async function showFrame(frame){
 const token=++requestId,caseId=current.id;frame=Math.max(0,Math.min(current.frames-1,Math.round(frame)));$('error').hidden=true;document.body.dataset.loading='true';
 try{const data=await api(`/api/frame?case=${caseId}&frame=${frame}`);const results=await Promise.all(data.cameras.map(async camera=>{if(!camera.image_available)return null;const img=new Image();img.src=`/api/image/${caseId}/${encodeURIComponent(camera.id)}/${frame}`;await img.decode();return img;}));if(token!==requestId)return;
 snapshot=data;$('frame').value=String(frame);$('scrub').value=String(frame);$('clock').textContent=`${data.seconds.toFixed(3)} s`;
 $('views').innerHTML=data.cameras.map(c=>`<div class="view" data-camera="${esc(c.id)}"><div class="view-head"><span>${esc(c.id)} · f${frame}</span><span class="state">${c.point?.observed?'観測あり':c.point?'根拠なし':'未保存'}</span></div><canvas class="full" aria-label="${esc(c.id)} 全画面"></canvas><div class="detail"><canvas class="crop" aria-label="${esc(c.id)} ボール周辺拡大"></canvas><div class="meta"><strong>128 px範囲の拡大</strong><br>${pointText(c.point)}<br>${c.point?'presence / confidenceの意味は出典参照':''}</div></div></div>`).join('');
 [...$('views').children].forEach((node,i)=>drawView(node,results[i],data.cameras[i]));
 const e=data.evidence;$('evidence').innerHTML=`<strong>f${frame} · ${labels[e.state]}</strong>保存観測: ${data.observing_cameras.map(esc).join(' + ')||'なし'}<br>${e.scores?`採点camera: ${e.scores.cameras.map(esc).join(' + ')}<br>H1支持: ${e.scores.supports[0]?'あり':'なし'} / H2支持: ${e.scores.supports[1]?'あり':'なし'}`:'この時刻のsupportは保存情報から確認できません。'}`;
 const pairs=current.decision?.pair_frames;$('pairs').innerHTML=pairs?'保存pair共通frame数<br>'+Object.entries(pairs).map(([key,value])=>`${esc(key)}: ${value}`).join(' / '):'camera対ごとの支持frame数: 未保存';
 renderHypotheses();drawGeometry();drawTimeline();document.body.dataset.frame=String(frame);document.body.dataset.case=String(caseId);document.body.dataset.loading='false';
 }catch(e){if(token===requestId){error(e);document.body.dataset.loading='error';}}
}
async function chooseCase(id){playing=false;$('play').textContent='再生';const token=++requestId;current=catalog.cases.find(c=>c.id===id);selected=0;centers=new Map();$('frame').max=String(current.frames-1);$('scrub').max=String(current.frames-1);const loaded=await api(`/api/timeline?case=${id}`);if(token!==requestId)return;timeline=loaded;renderDecision();await showFrame(0);}
$('frame').addEventListener('change',()=>showFrame(Number($('frame').value)));
$('scrub').addEventListener('input',()=>showFrame(Number($('scrub').value)));
$('case').addEventListener('change',()=>chooseCase(Number($('case').value)).catch(error));
$('back').onclick=()=>showFrame((snapshot?.frame??0)-1);$('next').onclick=()=>showFrame((snapshot?.frame??0)+1);
$('timeline').onclick=e=>{const r=$('timeline').getBoundingClientRect();showFrame((e.clientX-r.left-49)/(r.width-57)*(current.frames-1));};
$('play').onclick=async()=>{playing=!playing;$('play').textContent=playing?'停止':'再生';while(playing){const start=performance.now();if(snapshot.frame>=current.frames-1){playing=false;$('play').textContent='再生';break;}await showFrame(snapshot.frame+1);await new Promise(r=>setTimeout(r,Math.max(0,100-(performance.now()-start))));}};
window.addEventListener('resize',()=>{if(snapshot){drawTimeline();drawGeometry();}});
try{catalog=await api('/api/catalog');$('case').innerHTML=catalog.cases.map(c=>`<option value="${c.id}">${esc(c.clip)} · ${esc(c.store.split('/').slice(-2).join('/'))}${c.frame_scores_saved?' · frame診断あり':''}</option>`).join('');await chooseCase(0);}catch(e){error(e);}
