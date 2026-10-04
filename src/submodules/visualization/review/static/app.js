const $ = (id) => document.getElementById(id);
let meta, current, timeline, requestSerial = 0, timer = null, image = null;
let orbit = {yaw: -0.7, pitch: 0.13, zoom: 1.4}, drag = null;
const colors = {observed: '#6be0b6', input: '#ffcc6b', low: '#f07e8e', mesh: '#79b9ef', joints: '#d695f4'};
async function api(url) { const response = await fetch(url); if (!response.ok) { const error = await response.json(); throw new Error(error.detail || response.statusText); } return response.json(); }
function showError(error) { $('error').hidden = false; $('error').textContent = error.message; stop(); }
function frameValue() { return Math.max(0, Math.min(meta.frame_count - 1, Number($('frame').value))); }
function query() { return new URLSearchParams({person: $('person').value, camera: $('camera').value, frame: frameValue()}); }
function badge(text, state = '') { const element = document.createElement('span'); element.className = `badge ${state}`; element.textContent = text; return element; }
function metric(label, value) { const element = document.createElement('div'); element.className = 'metric'; const name = document.createElement('span'); name.textContent = label; const detail = document.createElement('b'); detail.textContent = value; element.append(name, detail); return element; }
function blank(canvas, text) { const ctx = canvas.getContext('2d'); ctx.fillStyle = '#101821'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = '#9fb5c8'; ctx.font = '14px system-ui'; ctx.textAlign = 'center'; ctx.fillText(text, canvas.width / 2, canvas.height / 2); ctx.textAlign = 'left'; }
function stop() { clearTimeout(timer); timer = null; $('play').textContent = '再生'; }
async function load(refreshTimeline = false) {
  const serial = ++requestSerial; $('error').hidden = true;
  const frame = frameValue(); $('frame').value = frame; $('scrub').value = frame;
  const selectedQuery = query();
  try {
    const result = await api(`/api/frame?${selectedQuery}`);
    if (serial !== requestSerial) return;
    const nextTimeline = refreshTimeline || !timeline ? await api(`/api/timeline?${selectedQuery}`) : timeline;
    if (serial !== requestSerial) return;
    current = result; timeline = nextTimeline;
    updateStatus(); drawTimeline(); drawBody(); drawCourt(); updateConfidence();
    const source = new Image();
    source.onload = () => { if (serial !== requestSerial) return; image = source; drawImages(); };
    source.onerror = () => { if (serial !== requestSerial) return; image = null; blank($('image'), 'RGBを取得できません（source欠損 / decodeエラー）'); blank($('crop'), 'RGBなし'); };
    source.src = `/api/image?${new URLSearchParams({camera: current.camera, frame: current.frame})}`;
    image = null; blank($('image'), '同じsource frameを読み込み中…'); blank($('crop'), '読み込み中…');
    history.replaceState(null, '', `/?${selectedQuery}`);
  } catch (error) { if (serial === requestSerial) { current = null; blank($('image'), '読込エラー'); blank($('crop'), '読込エラー'); blank($('body'), '読込エラー'); showError(error); } }
}
function updateStatus() {
  const p = current.placement, observed = current.observation.state === 'observed';
  const statuses = [badge(`2D観測 ${observed ? 'あり' : current.observation.state === 'unobserved' ? 'なし / mask=false' : '未保存 / ID不明'}`, observed ? 'good' : 'warn'), badge(`GVHMR入力 ${current.request ? 'sampleあり' : current.request_camera ? '未sample' : '未保存'}`, current.request ? 'good' : 'warn'), badge(`SMPL parameter ${current.parameters ? '保存sample' : current.recovered_camera ? '未sample' : '未保存'}`, current.parameters ? 'good' : 'warn'), badge(`court mesh ${p ? p.smpl_valid ? '有効' : '無効 / 描画なし' : '未保存'}`, p ? p.smpl_valid ? 'good' : 'bad' : 'warn')];
  $('status').replaceChildren(...statuses);
  $('time').textContent = `${current.seconds.toFixed(3)} s / FPS`;
  $('imageLabel').textContent = `${current.camera} · frame ${current.frame}`;
  const joints = current.request?.joints;
  $('inputStatus').textContent = `入力view ${current.request_camera || '未保存'} · ${current.request ? `保存request ${joints.filter(j => j[2] > 0).length}/17 joint confidence > 0` : 'このsource frameのrequestはありません（近傍sampleの代用なし）'}${current.request_camera && current.camera !== current.request_camera ? ' · 選択cameraと異なるため入力poseを重ねません' : ''}`;
  const nodes = p ? [metric('root / heading mask', `${p.root_valid} / ${p.heading_valid}`), metric('mesh / 配置理由', `${p.smpl_valid} · ${p.reason} (${p.reason_code})`), metric('court root (m)', p.position ? p.position.map(v => v.toFixed(3)).join(' / ') : '無効（座標0を表示しません）'), metric('yaw / mesh reprojection', `${p.yaw === null ? '無効' : `${(p.yaw * 180 / Math.PI).toFixed(1)}°`} / ${p.reprojection_px === null ? '無効' : `${p.reprojection_px.toFixed(2)} px`}`)] : [metric('body placement', '成果物未保存')];
  const valid = current.joints_3d?.filter(Boolean).length;
  nodes.push(metric('三角測量COCO17', valid === undefined ? '未保存' : `${valid}/17 有効joint（モデル推定）`));
  nodes.push(metric('各cameraの実観測', Object.entries(current.observations_by_camera).map(([camera,obs]) => `${camera}: ${obs.state === 'observed' ? `${obs.joints.filter(j=>j[2]>=.3).length}/17` : obs.state === 'unobserved' ? '未観測' : '未保存/ID不明'}`).join(' · ')));
  if (current.joint_reasons_3d) { const counts = {}; for(const reason of current.joint_reasons_3d){if(reason===0)continue;counts[reason]=(counts[reason]||0)+1;} if(Object.keys(counts).length)nodes.push(metric('保存3D joint拒否',Object.entries(counts).map(([code,count])=>`${meta.triangulation_rejection_codes[code]||`unknown_code_${code}`} (${code}) ×${count}`).join(' / '))); }
  $('placement').replaceChildren(...nodes);
  $('parameters').replaceChildren();
  for (const text of [current.parameters ? `in-camera transl (m): ${current.parameters.transl.map(v => v.toFixed(2)).join(', ')}` : 'camera-local parameter: このframeは未保存', current.parameters ? `body_pose norm: ${Math.hypot(...current.parameters.body_pose).toFixed(3)} rad` : '保存placementとGVHMR sampleは別の時間格子', current.vertices ? `${current.vertices.length} stored vertices · ${meta.faces ? 'triangle mesh' : 'topologyなし / 点群'}` : 'meshなし · 有効な3D jointのみ', `local track / group: ${current.observation.track_id ?? 'なし / 不明'} · person ${current.person}`]) { const p = document.createElement('div'); p.textContent = text; $('parameters').append(p); }
}
function drawPose(ctx, points, transform, color) {
  if (!points) return;
  const threshold = Number($('threshold').value);
  ctx.lineWidth = 1.7;
  for (const [a, b] of meta.edges) { if (points[a][2] < threshold || points[b][2] < threshold || points[a][2] <= 0 || points[b][2] <= 0) continue; const p = transform(points[a]), q = transform(points[b]); ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke(); }
  for (const point of points) { if (point[2] <= 0) continue; const p = transform(point), low = point[2] < threshold; ctx.beginPath(); ctx.arc(...p, low ? 3.4 : 2.3, 0, Math.PI * 2); ctx.strokeStyle = low ? colors.low : color; ctx.fillStyle = color; if (low) ctx.stroke(); else ctx.fill(); }
}
function drawImages() {
  if (!image || !current) return;
  const canvas = $('image'), ctx = canvas.getContext('2d'), sx = canvas.width / image.width, sy = canvas.height / image.height;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const box = current.observation.box || (current.camera === current.request_camera ? current.request?.box : null);
  if (box) { ctx.strokeStyle = colors.observed; ctx.lineWidth = 1; ctx.strokeRect((box[0]-box[2]/2)*sx, (box[1]-box[2]/2)*sy, box[2]*sx, box[2]*sy); }
  drawPose(ctx, current.observation.joints, p => [p[0]*sx,p[1]*sy], colors.observed);
  if (current.camera === current.request_camera) drawPose(ctx, current.request?.joints, p => [p[0]*sx,p[1]*sy], colors.input);
  canvas.dataset.frame = current.frame; canvas.dataset.camera = current.camera;
  const crop = $('crop'), c = crop.getContext('2d');
  if (!box) { blank(crop, 'この人物の実観測boxなし'); return; }
  const side = box[2]*1.12, x = box[0]-side/2, y = box[1]-side/2, scale = Math.min(crop.width, crop.height)/side;
  c.fillStyle = '#101821'; c.fillRect(0,0,crop.width,crop.height);
  const ox = (crop.width-side*scale)/2, oy = (crop.height-side*scale)/2;
  c.save(); c.beginPath(); c.rect(ox,oy,side*scale,side*scale); c.clip(); c.drawImage(image, ox-x*scale, oy-y*scale, image.width*scale, image.height*scale);
  const transform = p => [ox+(p[0]-x)*scale, oy+(p[1]-y)*scale];
  drawPose(c, current.observation.joints, transform, colors.observed);
  if (current.camera === current.request_camera) drawPose(c, current.request?.joints, transform, colors.input);
  c.restore();
}
function updateConfidence() {
  $('confidence').replaceChildren();
  const threshold = Number($('threshold').value);
  meta.joint_names.forEach((name, i) => { const obs = current.observation.joints?.[i]?.[2], req = current.request?.joints[i][2]; const element = document.createElement('div'); element.className = `joint ${obs !== undefined && obs < threshold || req !== undefined && req < threshold ? 'low' : ''}`; const text = document.createElement('span'); text.textContent = name.replace('left_', 'L ').replace('right_', 'R '); const value = document.createElement('b'); value.textContent = `${obs === undefined ? '—' : obs.toFixed(2)} / ${req === undefined ? '—' : req.toFixed(2)}`; element.append(text, value); $('confidence').append(element); });
}
function drawBody() {
  if (!current) return;
  const canvas = $('body'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#101821';ctx.fillRect(0,0,w,h);
  const points = current.vertices || current.joints_3d?.filter(Boolean) || [];
  if (!points.length) { blank(canvas, '有効なmesh / 3D jointなし'); return; }
  const root = current.placement?.position;
  const center = root || [0,1,2].map(a => points.reduce((s,p)=>s+p[a],0)/points.length);
  const project = p => { const x=p[0]-center[0], y=p[1]-center[1], z=p[2]-(root ? root[2]+.12 : center[2]); const u=x*Math.cos(orbit.yaw)-y*Math.sin(orbit.yaw), d=x*Math.sin(orbit.yaw)+y*Math.cos(orbit.yaw); return [w/2+u*150*orbit.zoom,h/2-(z*Math.cos(orbit.pitch)-d*Math.sin(orbit.pitch))*150*orbit.zoom,d*Math.cos(orbit.pitch)+z*Math.sin(orbit.pitch)]; };
  ctx.lineWidth=1; ctx.strokeStyle='#2f4151';
  for(let i=-2;i<=2;i++){ for(const axis of [0,1]) { const a=[center[0]-2,center[1]+i,0],b=[center[0]+2,center[1]+i,0]; if(axis===1){a[0]=center[0]+i;a[1]=center[1]-2;b[0]=center[0]+i;b[1]=center[1]+2;} const p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke(); } }
  if(current.vertices){ const projected=current.vertices.map(project);
    if(meta.faces){ const triangles=meta.faces.map(face=>({face,depth:face.reduce((sum,i)=>sum+projected[i][2],0)/3})).sort((a,b)=>b.depth-a.depth);
      for(const {face} of triangles){const [a,b,c]=face.map(i=>projected[i]); const normal=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]); const shade=.55+Math.min(.4,Math.abs(normal)/22);ctx.fillStyle=`rgb(${Math.round(104*shade)},${Math.round(167*shade)},${Math.round(223*shade)})`;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill();}
    }else{ctx.fillStyle=colors.mesh;for(const p of projected)ctx.fillRect(p[0],p[1],1.4,1.4);}
  }
  if(current.joints_3d){ctx.strokeStyle=colors.joints;ctx.fillStyle=colors.joints;ctx.lineWidth=2;for(const [a,b] of meta.edges){if(!current.joints_3d[a]||!current.joints_3d[b])continue;const p=project(current.joints_3d[a]),q=project(current.joints_3d[b]);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();}for(const joint of current.joints_3d){if(!joint)continue;const p=project(joint);ctx.beginPath();ctx.arc(p[0],p[1],3,0,Math.PI*2);ctx.fill();}}
  ctx.fillStyle='#a1bacf';ctx.font='12px system-ui';ctx.fillText('1 grid = 1 m · court Z-up',15,h-17);
  if(!current.vertices){ctx.fillStyle='#ffadb1';ctx.fillText('meshを描画しません · 有効3D jointのみ',15,24);}
}
function drawCourt() {
  const canvas=$('court'),ctx=canvas.getContext('2d');ctx.fillStyle='#101821';ctx.fillRect(0,0,canvas.width,canvas.height); const map=p=>[100+p[0]*7,75-p[1]*5];ctx.strokeStyle='#708b9e';ctx.lineWidth=1;ctx.strokeRect(100-5.485*7,75-11.885*5,10.97*7,23.77*5);ctx.beginPath();ctx.moveTo(62,75);ctx.lineTo(138,75);ctx.stroke();ctx.fillStyle='#9fb5c7';ctx.font='10px system-ui';ctx.fillText('court root · m',10,14);
  if(current.placement?.position){const p=map(current.placement.position);ctx.fillStyle=colors.mesh;ctx.beginPath();ctx.arc(...p,3.5,0,Math.PI*2);ctx.fill();}else{ctx.fillStyle=colors.low;ctx.fillText('root無効',72,78);}
}
function drawTimeline() {
  if(!timeline)return;const canvas=$('timeline'),ctx=canvas.getContext('2d'),left=125,right=canvas.width-18,width=right-left;ctx.fillStyle='#101821';ctx.fillRect(0,0,canvas.width,canvas.height);const rows=[['2D実観測',timeline.observed,colors.observed],['GVHMR sample',timeline.sampled,colors.input],['court root',timeline.root,colors.observed],['court mesh',timeline.mesh,colors.observed]];
  rows.forEach(([name,values,color],row)=>{const y=9+row*22;ctx.fillStyle='#afc5d6';ctx.font='12px system-ui';ctx.fillText(name,7,y+12);if(!values){ctx.fillStyle='#9db2c1';ctx.fillText('成果物未保存',left,y+12);return;}for(let i=0;i<meta.frame_count;i++){ctx.fillStyle=values[i]?color:row<2?'#354351':'#d77483';ctx.fillRect(left+i*width/meta.frame_count,y,Math.max(1,width/meta.frame_count),15);}});
  const x=left+current.frame*width/meta.frame_count;ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,3);ctx.lineTo(x,canvas.height-9);ctx.stroke();ctx.fillStyle='#fff';ctx.font='11px system-ui';ctx.fillText(`f${current.frame}`,Math.min(x+5,right-45),canvas.height-1);
}
$('person').addEventListener('change',()=>{stop();load(true);});$('camera').addEventListener('change',()=>{stop();load(true);});
for(const id of ['frame','scrub'])$(id).addEventListener('change',()=>{stop();if(id==='scrub')$('frame').value=$('scrub').value;load();});
$('previous').onclick=()=>{stop();$('frame').value=frameValue()-1;load();};$('next').onclick=()=>{stop();$('frame').value=frameValue()+1;load();};
$('threshold').onchange=()=>{if(current){drawImages();updateConfidence();}};
$('inputView').onclick=()=>{if(current?.request_camera){stop();$('camera').value=current.request_camera;load(true);}};
$('rejected').onclick=()=>{if(!timeline?.mesh)return;stop();const start=frameValue();for(let i=1;i<=meta.frame_count;i++){const frame=(start+i)%meta.frame_count;if(!timeline.mesh[frame]){$('frame').value=frame;load();return;}}};
$('play').onclick=()=>{if(timer!==null){stop();return;} $('play').textContent='停止';const tick=async()=>{if(frameValue()===meta.frame_count-1){stop();return;}$('frame').value=frameValue()+1;await load();if($('play').textContent==='停止')timer=setTimeout(tick,150);};timer=setTimeout(tick,0);};
$('timeline').onclick=event=>{if(!meta)return;stop();const rect=event.target.getBoundingClientRect(),x=(event.clientX-rect.left)*event.target.width/rect.width;$('frame').value=Math.round((x-125)/(event.target.width-143)*(meta.frame_count-1));load();};
$('body').onpointerdown=event=>{drag=[event.clientX,event.clientY];event.target.setPointerCapture(event.pointerId);};
$('body').onpointermove=event=>{if(!drag)return;orbit.yaw+=(event.clientX-drag[0])*.008;orbit.pitch=Math.max(-1.4,Math.min(1.4,orbit.pitch+(event.clientY-drag[1])*.005));drag=[event.clientX,event.clientY];drawBody();};
$('body').onpointerup=()=>{drag=null;};$('body').onpointercancel=()=>{drag=null;};
$('body').addEventListener('wheel',event=>{event.preventDefault();orbit.zoom=Math.max(.4,Math.min(3,orbit.zoom*Math.exp(-event.deltaY*.001)));drawBody();},{passive:false});
$('reset').onclick=()=>{orbit={yaw:-.7,pitch:.13,zoom:1.4};drawBody();};
try {
  meta=await api('/api/meta');
  $('clip').textContent=`${meta.clip_id} · ${meta.frame_count} frames · ${meta.fps.toFixed(3)} fps · ${meta.cameras.length} cameras`;
  $('snapshot').textContent=`保存snapshot: ${meta.store.split('/').slice(-2).join('/')} · identity ${meta.identity_policy} · scene.json ${meta.index_sha256.slice(0,12)}`;
  for(const person of meta.people)$('person').add(new Option(`person ${person}`,person));
  for(const camera of meta.cameras)$('camera').add(new Option(`${camera.id} · ${camera.width}×${camera.height}`,camera.id));
  if(!meta.people.length)throw new Error('このstoreには識別された人物が保存されていません。GVHMR / identity成果物を持つ保存snapshotを指定してください。');
  const url=new URLSearchParams(location.search);if(meta.people.includes(Number(url.get('person'))))$('person').value=url.get('person')??meta.people[0];
  const cameras=meta.cameras.map(c=>c.id);const requested=url.get('camera');$('camera').value=cameras.includes(requested)?requested:cameras.includes('cam1')?'cam1':cameras[0];
  if(url.has('frame'))$('frame').value=url.get('frame');$('frame').max=$('scrub').max=meta.frame_count-1;
  const table=document.createElement('table');const header=document.createElement('tr');for(const title of ['component','schema','version','artifact','origin']){const th=document.createElement('th');th.textContent=title;header.append(th);}table.append(header);for(const [node,artifact] of Object.entries(meta.artifacts)){const row=document.createElement('tr');for(const value of [node,artifact.schema,artifact.version,artifact.artifact_id.slice(0,16),artifact.origin]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}table.append(row);}$('provenance').append(table);
  await load(true);
} catch(error){showError(error);}
