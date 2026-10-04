const $ = (id) => document.getElementById(id);
const colors = ["#68d9ed", "#b795f4", "#fa95b3", "#7fb5ff"];
const reasonColors = ["#65e3a8", "#f59191", "#68798d", "#68798d", "#68798d", "#f4bc69", "#f4bc69", "#68798d"];
const reasonNames = ["観測", "画面外", "未レビュー", "instanceなし", "複数instance", "補間位置", "遮蔽推定", "unresolved"];
const skeleton = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const state = {catalog:null, clip:null, timeline:null, frame:0, data:null, image:null, playing:false, token:0, busy:false};

async function get(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}
function endpoint(path, extra={}) {
  return `${path}?${new URLSearchParams({clip:state.clip, ...extra})}`;
}
function text(id, value) { $(id).textContent = value; }
function fail(error) { $("error").hidden = false; text("error", `読込失敗: ${error.message}`); document.body.dataset.busy="false"; state.busy=false; stop(); }
function stop() { state.playing = false; text("play", "再生"); }
function option(value, label) { const item = document.createElement("option"); item.value=value; item.textContent=label; return item; }
function filterClips(preferred=null) {
  const query = $("search").value.toLowerCase(), source = $("source").value;
  const clips = state.catalog.clips.filter(c => (!source || c.source===source) && c.id.toLowerCase().includes(query));
  $("clip").replaceChildren(...clips.map(c => option(c.id, `${c.id} · ${c.split} · ${c.frames} f${c.has_teachers ? " · 教師/GMM" : " · 教師未提供"}`)));
  if (preferred && clips.some(c=>c.id===preferred)) $("clip").value=preferred;
  for(const element of document.querySelectorAll(".review-grid,.patches-panel,.provenance")) element.style.display=clips.length?"":"none";
  if (!clips.length) { stop(); state.clip=null; state.data=null; state.timeline=null; return; }
  if (state.clip!==$("clip").value) selectClip().catch(fail);
}
async function selectClip() {
  stop();
  const clip=$("clip").value;
  state.clip=clip; state.data=null; state.image=null;
  state.busy=true; document.body.dataset.busy="true"; document.body.dataset.ready="false";
  $("error").hidden=true;
  const timeline=await get(endpoint("/api/timeline"));
  if (state.clip!==clip) return;
  state.timeline=timeline;
  $("frame-range").max=timeline.frame_count-1; $("frame-number").max=timeline.frame_count-1;
  $("condition").options[1].disabled=!timeline.gap_mask;
  if (!timeline.gap_mask) $("condition").value="observed";
  const requested=new URLSearchParams(location.search).get("frame");
  await setFrame(requested===null ? 0 : Number(requested));
}
async function setFrame(frame) {
  if (!state.timeline) return;
  state.frame=Math.max(0,Math.min(state.timeline.frame_count-1,Math.round(Number(frame)||0)));
  const token=++state.token;
  state.busy=true; document.body.dataset.busy="true";
  $("frame-range").value=state.frame; $("frame-number").value=state.frame;
  try {
    const data=await get(endpoint("/api/frame",{frame:state.frame,condition:$("condition").value}));
    let img=null;
    if (data.rgb.available) {
      img=new Image(); img.src=endpoint("/api/image",{frame:data.frame});
      await img.decode();
    }
    if (token!==state.token) return;
    state.data=data; state.image=img;
    const query=new URLSearchParams({clip:state.clip,frame:data.frame,condition:data.condition});
    history.replaceState(null,"",`?${query}`);
    $("error").hidden=true; render();
    document.body.dataset.ready="true";
  } catch(error) { if (token===state.token) fail(error); }
  finally { if(token===state.token) { state.busy=false; document.body.dataset.busy="false"; } }
}
function drawPoint(ctx,x,y,color,r=3) {
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill();
  ctx.strokeStyle="#0a111c"; ctx.lineWidth=1; ctx.stroke();
}
function label(ctx,value,x,y,color) {
  ctx.font="12px system-ui"; ctx.lineWidth=3; ctx.strokeStyle="#101623"; ctx.strokeText(value,x,y); ctx.fillStyle=color; ctx.fillText(value,x,y);
}
function renderScene() {
  const d=state.data, canvas=$("scene"), ctx=canvas.getContext("2d");
  const [w,h]=d.stored_size_wh; canvas.width=w; canvas.height=h;
  const sx=(d.source_size_wh[0]-1)*d.stored_scale, sy=(d.source_size_wh[1]-1)*d.stored_scale;
  const xy=(uv)=>[uv[0]*sx,uv[1]*sy];
  const useRGB=state.image && $("show-rgb").checked;
  ctx.fillStyle="#0b1724"; ctx.fillRect(0,0,w,h);
  if(useRGB) ctx.drawImage(state.image,0,0);
  else {
    ctx.strokeStyle="#23364a"; ctx.lineWidth=1;
    for(let i=0;i<=10;i++) {ctx.beginPath();ctx.moveTo(i*sx/10,0);ctx.lineTo(i*sx/10,h);ctx.moveTo(0,i*sy/10);ctx.lineTo(w,i*sy/10);ctx.stroke();}
    label(ctx,d.rgb.available?"対応検証済みRGBを非表示 / source UV座標面":"RGB未提供 / source UV座標面",20,28,"#9bacbf");
  }
  if(d.context && $("show-context").checked) {
    d.context.court_uv.forEach((uv,i)=>{if(d.context.court_valid[i]) {const [x,y]=xy(uv);drawPoint(ctx,x,y,"#91aaff",3);label(ctx,`C${i+1}`,x+5,y-4,"#91aaff");}});
    for(const p of d.context.people) {
      for(const [a,b] of skeleton) if(p.valid[a]&&p.valid[b]) {
        ctx.beginPath();ctx.moveTo(...xy(p.uv[a]));ctx.lineTo(...xy(p.uv[b]));
        ctx.strokeStyle=[7,8,9,10].includes(b)?"#b6c5ff":"#6d80ab";ctx.lineWidth=2;ctx.stroke();
      }
      p.uv.forEach((uv,i)=>{if(p.valid[i])drawPoint(ctx,...xy(uv),[7,8,9,10].includes(i)?"#e1a7ff":"#91aaff",[7,8,9,10].includes(i)?4:2);});
      if(p.valid[5]) {const [x,y]=xy(p.uv[5]);label(ctx,`P${p.id}`,x-12,y-12,"#c5d0ff");}
    }
  }
  if(d.gmm && $("show-gmm").checked) {
    d.gmm.components.forEach((c,i)=>{
      const [x,y]=xy(c.uv);
      ctx.save();ctx.globalAlpha=Math.max(.2,c.weight);ctx.strokeStyle=colors[i%colors.length];ctx.lineWidth=2;
      ctx.beginPath();ctx.ellipse(x,y,c.semiaxes_px[0]*d.stored_scale,c.semiaxes_px[1]*d.stored_scale,c.angle_degrees*Math.PI/180,0,2*Math.PI);ctx.stroke();ctx.restore();
      drawPoint(ctx,x,y,colors[i%colors.length],3);
    });
    const top=d.gmm.components.reduce((a,b)=>a.weight>=b.weight?a:b), [x,y]=xy(top.uv);
    label(ctx,`R${d.gmm.components.indexOf(top)+1}`,x+7,y+15,"#68d9ed");
  }
  if($("show-candidates").checked) for(const c of d.effective_candidates) {
    const [x,y]=xy(c.uv);drawPoint(ctx,x,y,"#f4cf70",3);label(ctx,`D${c.slot+1}`,x+5,y-6,"#f4cf70");
  }
  if(d.target?.uv && $("show-target").checked) {
    const [x,y]=xy(d.target.uv), color=d.target.position_valid?"#65e3a8":"#f4bc69";
    ctx.beginPath();
    if(d.target.position_valid) {ctx.moveTo(x-8,y);ctx.lineTo(x+8,y);ctx.moveTo(x,y-8);ctx.lineTo(x,y+8);}
    else {ctx.moveTo(x,y-8);ctx.lineTo(x+8,y);ctx.lineTo(x,y+8);ctx.lineTo(x-8,y);ctx.closePath();}
    ctx.lineWidth=5;ctx.strokeStyle="#101623";ctx.stroke();ctx.lineWidth=2.5;ctx.strokeStyle=color;ctx.stroke();
    label(ctx,d.target.position_valid?"T observed":"T reference",x+10,y+3,color);
  }
  if(d.gap_active) label(ctx,"人工gap / このframeのdetector入力証拠は無効",20,h-18,"#f4bc69");
  text("rgb-badge", useRGB?"RGB: 対応検証済み":d.rgb.available?"座標面 / RGB検証済み":"RGB未提供 / 座標面");
  const zoom=$("zoom"), zctx=zoom.getContext("2d");
  let focus=null, focusLabel="位置なし";
  if(d.target?.uv) {focus=d.target.uv;focusLabel=d.target.position_valid?"observed教師中心":"推定参考位置中心";}
  else if(d.gmm) {focus=d.gmm.components.reduce((a,b)=>a.weight>=b.weight?a:b).uv;focusLabel="最大weight成分中心";}
  else if(d.effective_candidates.length) {focus=d.effective_candidates[0].uv;focusLabel="候補D1中心";}
  zctx.fillStyle="#0b1724";zctx.fillRect(0,0,zoom.width,zoom.height);
  if(focus) {
    const [x,y]=xy(focus), cw=w*.2, ch=cw*zoom.height/zoom.width;
    const left=Math.max(0,Math.min(w-cw,x-cw/2)), top=Math.max(0,Math.min(h-ch,y-ch/2));
    zctx.drawImage(canvas,left,top,cw,ch,0,0,zoom.width,zoom.height);
  }
  text("zoom-label",focusLabel);
}
function maskCard(name,value,note) {
  const div=document.createElement("div");div.className="mask";
  const title=document.createElement("span");title.textContent=name;
  const number=document.createElement("b");number.textContent=value;
  const small=document.createElement("span");small.textContent=note;
  div.append(title,number,small);return div;
}
function renderTeacher() {
  const t=state.data.target;
  if(!t) {
    text("teacher-state","未提供");text("teacher-description","このclipの保存教師NPZは未提供。unknown教師とは別の状態です。");
    $("masks").replaceChildren(maskCard("位置mask","—","未提供"),maskCard("存在mask","—","未提供"));text("teacher-coordinates","RGB providerの注釈は読み替えません。");return;
  }
  const estimated=[5,6].includes(t.reason_code);
  text("teacher-state",reasonNames[t.reason_code]);
  text("teacher-description",t.position_valid?"単一球の観測位置。位置と存在の教師を使用。":t.reason_code===1?"明示的な画面外。存在=0の教師のみ。位置教師なし。":estimated?"推定位置は参考表示。位置・存在の主教師maskは共に0。":"存在はunknown。未検出・instanceなしをamodal不存在にしません。");
  $("masks").replaceChildren(maskCard("位置教師mask",t.position_valid?"1":"0",t.position_valid?"observedのみ":"位置lossに使わない"),maskCard("存在教師mask",t.presence_valid?"1":"0",t.presence_valid?`存在教師 = ${t.presence?1:0}`:"存在はunknown"));
  text("teacher-coordinates",t.uv?`${t.reason} · source UV (${t.uv.map(x=>x.toFixed(4)).join(", ")})`:`${t.reason} · 位置なし`);
}
function renderGMM() {
  const g=state.data.gmm;
  text("presence-value",g?g.presence_probability.toFixed(3):"—");
  $("presence-meter").style.width=g?`${g.presence_probability*100}%`:"0%";
  $("components").replaceChildren();
  if(!g) {const p=document.createElement("p");p.className="tiny";p.textContent="保存GMM未提供。新規推論は実行しません。";$("components").append(p);return;}
  g.components.forEach((c,i)=>{
    const row=document.createElement("div");row.className="component-row";
    const name=document.createElement("span");name.textContent=`R${i+1}`;name.style.color=colors[i%colors.length];
    const track=document.createElement("div");track.className="component-track";
    const fill=document.createElement("div");fill.style.width=`${c.weight*100}%`;fill.style.background=colors[i%colors.length];track.append(fill);
    const value=document.createElement("span");value.textContent=`${(c.weight*100).toFixed(1)}%`;
    row.title=`source UV ${c.uv.map(x=>x.toFixed(4))}, 2σ semiaxes ${c.semiaxes_px.map(x=>x.toFixed(1))} source px`;
    row.append(name,track,value);$("components").append(row);
  });
}
function renderPatches() {
  const d=state.data; $("patches").replaceChildren();
  text("candidate-count",`${d.effective_candidates.length} 有効 / ${d.candidates.length} 保存候補`);
  text("patch-note",d.gap_active?"人工gap: 保存patchを参考表示していますが、このframeのGMM入力では全候補が無効です。RGBの遮蔽実験ではありません。":"候補slotの並びは保存時のまま。scoreはnative heatmap peakで、amodal存在確率ではありません。斜線cellはpatch_valid=false。");
  for(const c of d.candidates) {
    const div=document.createElement("div");div.className=`patch-card${d.gap_active?" gap-disabled":""}`;
    const title=document.createElement("b");title.textContent=`D${c.slot+1}`;
    const score=document.createElement("span");score.className="score";score.textContent=c.score.toFixed(3);
    const canvas=document.createElement("canvas");canvas.width=100;canvas.height=100;
    const ctx=canvas.getContext("2d"), step=100/c.patch.length;
    c.patch.forEach((row,y)=>row.forEach((v,x)=>{
      ctx.fillStyle=c.patch_valid[y][x]?`rgb(${Math.round(15+240*v)},${Math.round(28+175*v)},${Math.round(45+40*v)})`:"#263044";
      ctx.fillRect(x*step,y*step,step-1,step-1);
      if(!c.patch_valid[y][x]) {ctx.strokeStyle="#6b7b91";ctx.beginPath();ctx.moveTo(x*step,y*step);ctx.lineTo((x+1)*step,(y+1)*step);ctx.stroke();}
    }));
    const center=Math.floor(c.patch.length/2);ctx.strokeStyle="#68d9ed";ctx.lineWidth=2;ctx.strokeRect(center*step+1,center*step+1,step-3,step-3);
    const small=document.createElement("small");small.textContent=`UV ${c.uv.map(x=>x.toFixed(3)).join(" / ")}\ncell ${c.cell.join(" / ")}`;small.style.whiteSpace="pre-line";
    div.append(title,score,canvas,small);$("patches").append(div);
  }
  if(!d.candidates.length) {const p=document.createElement("p");p.className="tiny";p.textContent="検出証拠は生成済み、有効候補0件。未生成cacheとは別です。";$("patches").append(p);}
}
function renderTimeline() {
  const t=state.timeline, canvas=$("timeline"), ctx=canvas.getContext("2d"), left=126;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#121c29";ctx.fillRect(0,0,canvas.width,canvas.height);
  const rows=[
    ["教師の理由", t.reason, v=>reasonColors[v]],
    ["位置mask",t.position_valid,v=>v?"#65e3a8":"#344457"],
    ["存在mask",t.presence_valid,v=>v?"#8dafd7":"#344457"],
    ["detector候補",t.candidate_count,v=>v?"#ad9249":"#344457"],
    ["pose観測",t.pose_count,v=>v?"#a492ce":"#344457"],
    ["人工gap",t.gap_mask,v=>v?"#dc9658":"#344457"]
  ];
  const step=(canvas.width-left-12)/t.frame_count;
  rows.forEach(([name,values,color],r)=>{
    ctx.fillStyle="#9bacbf";ctx.font="12px system-ui";ctx.fillText(name,12,r*18+14);
    if(!values) {ctx.fillText("未提供",left,r*18+14);return;}
    values.forEach((v,i)=>{ctx.fillStyle=color(v);ctx.fillRect(left+i*step,r*18+3,Math.max(1,step),12);});
  });
  const x=left+state.frame*step;ctx.strokeStyle="#f0f4fa";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();
}
function render() {
  const d=state.data;
  text("scene-title",`${d.clip_id} / frame ${d.frame}`);
  text("frame-subtitle",`${d.source} · ${d.split} · camera ${d.camera??"記録なし"} · PTS ${d.pts} · ${d.seconds.toFixed(3)} s`);
  text("frame-total",`/ ${d.frame_count-1}`);
  text("geometry-note",`source ${d.source_size_wh.join("×")} → JPEG ${d.stored_size_wh.join("×")} / x=(W−1)u, y=(H−1)v`);
  text("support-note",`detector RGB参照 [${d.detector_rgb_support.join(", ")}) / native ${d.heatmap_size_hw.slice().reverse().join("×")}`);
  text("context-state",d.context?`生成済み / このframe ${d.context.people.length}観測track・${d.context.detection_count}人物検出\ncourt: frame 0 / ${d.context.court_valid.filter(Boolean).length}有効KP / pose threshold ${d.context.pose_threshold}`:"このclipのcontextは未提供。未生成を人物0人・コート検出失敗には変換しません。");
  $("context-state").style.whiteSpace="pre-line";
  text("provenance-note",`旧store: ${state.catalog.original_store} (${state.catalog.original_store_present?"存在":"不在"})。RGB: ${d.rgb.reason}。保存教師は旧評価NPZのまま。contextは比較用で、保存された文脈なしGMMの入力ではありません。`);
  text("provenance",JSON.stringify({clip:d.clip_id,frame:d.frame,condition:d.condition,rgb:d.rgb,...d.provenance},null,2));
  renderTeacher();renderGMM();renderPatches();renderScene();renderTimeline();
}
function jump(kind) {
  const t=state.timeline;
  const matches=(i)=>kind==="gap"?t.gap_mask?.[i]:kind==="observed"?t.reason?.[i]===0:kind==="estimated"?[5,6].includes(t.reason?.[i]):kind==="absent"?t.reason?.[i]===1:t.reason && [2,3,4,7].includes(t.reason[i]);
  for(let offset=1;offset<=t.frame_count;offset++) {const i=(state.frame+offset)%t.frame_count;if(matches(i)) {setFrame(i).catch(fail);return;}}
  text("error","このclipには選択した状態の保存frameがありません。");$("error").hidden=false;
}
async function playback() {
  if(!state.playing) return;
  if(state.frame>=state.timeline.frame_count-1) {stop();return;}
  await setFrame(state.frame+1);
  if(state.playing) setTimeout(playback,100);
}
async function init() {
  const catalog=await get("/api/catalog");state.catalog=catalog;
  text("total",`${catalog.counts.clips} clip / ${catalog.counts.frames.toLocaleString()} frame`);
  text("evidence-name",catalog.evidence);
  text("context-name",catalog.context?`${catalog.context} / ${catalog.counts.context_clips} clip`:"未提供");
  text("predictions-name",catalog.predictions?`${catalog.predictions} / ${catalog.counts.teacher_gmm_clips} val clip`:"未提供");
  const table=document.createElement("table"), tr=document.createElement("tr");
  for(const name of ["source","split","clip","frame"]) {const th=document.createElement("th");th.textContent=name;tr.append(th);}table.append(tr);
  for(const row of catalog.groups) {const tr=document.createElement("tr");for(const value of [row.source,row.split,row.clips,row.frames]) {const td=document.createElement("td");td.textContent=value;tr.append(td);}table.append(tr);}$("group-table").append(table);
  const params=new URLSearchParams(location.search), cid=params.get("clip")||"meiji/video_000/clip_001/cam0";
  if(params.get("condition")==="evidence_gap") $("condition").value="evidence_gap";
  filterClips(cid);
}
$("source").addEventListener("change",()=>filterClips());$("search").addEventListener("input",()=>filterClips());
$("clip").addEventListener("change",()=>selectClip().catch(fail));
$("condition").addEventListener("change",()=>{stop();setFrame(state.frame).catch(fail);});
$("previous").addEventListener("click",()=>{stop();setFrame(state.frame-1).catch(fail);});
$("next").addEventListener("click",()=>{stop();setFrame(state.frame+1).catch(fail);});
$("frame-range").addEventListener("input",e=>{stop();setFrame(e.target.value).catch(fail);});
$("frame-number").addEventListener("change",e=>{stop();setFrame(e.target.value).catch(fail);});
$("play").addEventListener("click",()=>{if(state.playing)stop();else {state.playing=true;text("play","停止");playback().catch(fail);}});
$("jump").addEventListener("change",e=>{stop();if(e.target.value)jump(e.target.value);e.target.value="";});
$("timeline").addEventListener("click",e=>{const rect=e.target.getBoundingClientRect(),x=(e.clientX-rect.left)*1280/rect.width;stop();setFrame(Math.floor((x-126)/(1280-138)*state.timeline.frame_count)).catch(fail);});
for(const id of ["show-candidates","show-context","show-target","show-gmm","show-rgb"]) $(id).addEventListener("change",()=>{if(state.data)renderScene();});
init().catch(fail);
