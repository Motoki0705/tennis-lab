"use strict";
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = {clip:params.get("clip"), frame:Number(params.get("frame") || 0), camera:params.get("camera"), track:params.has("track") ? Number(params.get("track")) : null, report:params.has("report") ? Number(params.get("report")) : null, filter:"all", serial:0, loadingClip:false};
let catalog, detail, current;
function el(tag, cls, text) { const node=document.createElement(tag); if(cls) node.className=cls; if(text !== undefined) node.textContent=text; return node; }
function replace(id, ...children) { $(id).replaceChildren(...children); }
function color(observation) { return observation.label_state==="player" ? (observation.person_id==="A" ? "#047d8d" : "#ac4f00") : ({non_player:"#777d86", ambiguous:"#a85ca9", unmatched:"#ae7820"}[observation.label_state] || "#999"); }
function stateName(observation) { return {player:"選手",non_player:"除外対象",ambiguous:"曖昧・採点外",unmatched:"ラベル未対応"}[observation.label_state]; }
function imageUrl(camera, track) { const query=new URLSearchParams({clip:state.clip,camera,frame:state.frame}); if(track !== null && track !== undefined) query.set("track",track); return "/api/image?"+query; }
async function api(path, query) { const response=await fetch(path+"?"+new URLSearchParams(query)); const value=await response.json(); if(!response.ok) throw new Error(value.detail || response.statusText); return value; }
function selected() { return current?.observations.find(o=>o.camera===state.camera && o.track_id===state.track); }
function linked(o) { const item=selected(); return item?.label_state==="player" && o.label_state==="player" && item.person_id===o.person_id; }
function choose(camera, track) { state.camera=camera; state.track=track; loadFrame(); }
function updateUrl() { const p=new URLSearchParams({clip:state.clip,frame:state.frame}); if(state.camera && state.track !== null) {p.set("camera",state.camera);p.set("track",state.track);} if(state.report !== null) p.set("report",state.report); history.replaceState(null,"","?"+p); }
function timeline(track) {
  const row=el("div","timeline"+(track.camera===state.camera && track.track_id===state.track ? " selected":""));
  const button=el("button","",`${track.camera} · t${track.track_id}`); button.onclick=()=>choose(track.camera,track.track_id);
  const bar=el("div","timeline-bar");
  for(const span of track.spans) { const segment=el("span","span"); segment.style.left=100*span.start/detail.frames+"%";segment.style.width=100*(span.end-span.start)/detail.frames+"%";segment.style.background=color(span);segment.title=`[${span.start},${span.end}) ${span.person_id || stateName(span)}`; bar.append(segment); }
  const cursor=el("span","cursor");cursor.style.left=100*state.frame/detail.frames+"%";bar.append(cursor);
  bar.onclick=(event)=>{const rect=bar.getBoundingClientRect();state.frame=Math.min(detail.frames-1,Math.max(0,Math.floor((event.clientX-rect.left)/rect.width*detail.frames)));state.camera=track.camera;state.track=track.track_id;loadFrame();};
  row.append(button,bar,el("span","muted",`${track.observed_frames}f / 切替${track.label_transitions.length}`));return row;
}
function renderTimelines() {
  replace("timelines",...detail.timelines.map(timeline));
  $("timeline-count").textContent=`(${detail.timelines.length} tracks)`;
  const track=detail.timelines.find(t=>t.camera===state.camera && t.track_id===state.track);
  replace("track-preview",...(track ? [timeline(track)] : []));
}
async function renderViews() {
  const cards=[], loaded=[];
  for(const camera of detail.camera_ids) {
    const all=current.observations.filter(o=>o.camera===camera);
    const visible=all.filter(o=>state.filter==="all" || (state.filter==="player" ? o.label_state==="player" : state.filter==="unknown" ? ["ambiguous","unmatched"].includes(o.label_state) : o.label_state==="non_player"));
    const card=el("div","camera-card");card.dataset.camera=camera;
    const heading=el("div","camera-heading");heading.append(el("b","",camera),el("span","",`f${state.frame} · ${all.length} boxes`));card.append(heading);
    const wrap=el("div","image-wrap"), img=el("img"), canvas=el("canvas");img.alt=`${state.clip} ${camera} frame ${state.frame} 全景`;canvas.width=detail.width;canvas.height=detail.height;
    loaded.push(new Promise((resolve,reject)=>{img.onload=()=>{const ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);for(const o of all){const [x1,y1,x2,y2]=o.box;ctx.strokeStyle=color(o);ctx.lineWidth=linked(o)||o.camera===state.camera&&o.track_id===state.track ? 6 : 3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.font="27px system-ui";const text=o.track_id===null?"?":`t${o.track_id}`;const y=Math.max(28,y1-6);ctx.fillStyle="rgba(0,0,0,.65)";ctx.fillRect(x1,y-27,ctx.measureText(text).width+8,31);ctx.fillStyle="#fff";ctx.fillText(text,x1+4,y);}resolve();};img.onerror=()=>reject(new Error(`${camera}: source RGB unavailable`));}));
    img.src=imageUrl(camera,null);wrap.append(img,canvas);
    canvas.onclick=(event)=>{const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*detail.width/rect.width,y=(event.clientY-rect.top)*detail.height/rect.height;const hit=all.filter(o=>o.track_id!==null && x>=o.box[0] && x<=o.box[2] && y>=o.box[1] && y<=o.box[3]).sort((a,b)=>(a.box[2]-a.box[0])*(a.box[3]-a.box[1])-(b.box[2]-b.box[0])*(b.box[3]-b.box[1]))[0];if(hit)choose(camera,hit.track_id);};card.append(wrap);
    const crops=el("div","crop-grid");
    for(const o of visible) {
      const crop=el("div","crop"+(o.camera===state.camera&&o.track_id===state.track?" selected":linked(o)?" linked":""));crop.dataset.track=o.track_id;crop.tabIndex=0;crop.setAttribute("role","button");crop.onclick=()=>{if(o.track_id!==null)choose(camera,o.track_id);};crop.onkeydown=(event)=>{if(event.key==="Enter")crop.click();};
      if(o.track_id!==null){const picture=el("img");picture.src=imageUrl(camera,o.track_id);picture.alt=`${camera} raw t${o.track_id} f${state.frame} crop`;crops.append(crop);crop.append(picture);loaded.push(picture.decode());}
      else{crops.append(crop);}
      const text=el("div","crop-text");text.append(el("b","",`t${o.track_id ?? "?"} → ${o.person_id ?? "?"}`),el("span","badge "+(o.label_state==="player"?(o.person_id==="A"?"player-a":"player-b"):o.label_state),stateName(o)));
      if(o.footpoint)text.append(el("div","coordinate",`x ${o.footpoint[0].toFixed(1)} m`),el("div","coordinate",`y ${o.footpoint[1].toFixed(1)} m`));else text.append(el("div","reason",o.footpoint_reason));crop.append(text);
    }
    if(!visible.length)card.append(el("div","empty",all.length?"このfilterに該当するboxなし":"このframeの実観測boxなし（人物不在とは断定しない）"));else card.append(crops);
    cards.push(card);
  }
  replace("views",...cards);await Promise.all(loaded);
}
function renderInspector() {
  const chosen=selected(), raw=detail.timelines.find(t=>t.camera===state.camera&&t.track_id===state.track);
  const container=el("div");
  if(chosen){container.append(el("b","",`${state.camera} / raw t${state.track} → ${chosen.person_id ?? "?"}`),el("p","",`${stateName(chosen)} · ${chosen.description}`),el("div","muted",`元track IDs: ${(chosen.source_track_ids || []).join(", ")} · box [${chosen.box.map(x=>x.toFixed(1)).join(", ")}] px`));}
  if(chosen?.label_state==="player" && chosen.footpoint){const distances=current.observations.filter(o=>linked(o)&&o.camera!==chosen.camera&&o.footpoint).map(o=>`${chosen.camera} ↔ ${o.camera} ${Math.hypot(chosen.footpoint[0]-o.footpoint[0],chosen.footpoint[1]-o.footpoint[1]).toFixed(2)} m`);container.append(el("div","muted",`同frameの足元距離（CPU表示計算）: ${distances.join(" / ") || "比較可能な点なし"}`));}
  if(!chosen) container.append(el("p","",raw?`${state.camera} t${state.track}: このframeで実観測なし。人物不在・画面外とは断定しない。`:"全景またはcropからraw IDを選択してください。"));
  if(raw){const events=el("div","events");let previous=null;for(const span of raw.spans){const name=span.person_id||stateName(span);if(name===previous)continue;previous=name;const button=el("button","",`f${span.start} ${name}`);button.onclick=()=>{state.frame=span.start;loadFrame();};events.append(button);}container.append(events);}
  replace("inspector",container);
  replace("filters",...[ ["all","全box"],["player","選手"],["non_player","除外対象"],["unknown","曖昧・未対応"] ].map(([key,name])=>{const button=el("button",key===state.filter?"active":"",name);button.onclick=async()=>{state.filter=key;renderInspector();try{await renderViews();}catch(error){showError(error);}};return button;}));
}
function renderCourt() {
  const canvas=$("court"),ctx=canvas.getContext("2d"), {half_width:w,half_length:l,singles_half_width:s,service_y:y}=catalog.court;
  const scale=10,x=(v)=>180+v*scale,py=(v)=>170-v*scale;
  ctx.clearRect(0,0,360,340);ctx.fillStyle="#eff5ed";ctx.fillRect(0,0,360,340);ctx.strokeStyle="#bbc9b7";ctx.lineWidth=1;
  ctx.strokeRect(x(-w),py(l),2*w*scale,2*l*scale);ctx.strokeRect(x(-s),py(l),2*s*scale,2*l*scale);
  const line=(a,b,c,d)=>{ctx.beginPath();ctx.moveTo(x(a),py(b));ctx.lineTo(x(c),py(d));ctx.stroke();};line(-w,0,w,0);line(-s,y,s,y);line(-s,-y,s,-y);line(0,-y,0,y);
  ctx.fillStyle="#64736f";ctx.font="10px system-ui";ctx.fillText("x → (m)",265,326);ctx.fillText("y ↑",12,18);
  const missing=[];const offsets={cam0:[-6,-5],cam1:[6,-5],cam2:[0,12]};
  for(const o of current.observations){if(!o.footpoint)continue;const [px,yy]=o.footpoint;if(Math.abs(px)>17 || Math.abs(yy)>16){missing.push(`${o.camera} t${o.track_id}: (${px.toFixed(1)}, ${yy.toFixed(1)}) m / 表示範囲外`);continue;}ctx.fillStyle=color(o);ctx.beginPath();ctx.arc(x(px),py(yy),linked(o)?5:3,0,Math.PI*2);ctx.fill();const offset=offsets[o.camera]||[0,0];ctx.fillText(`${o.camera.replace("cam", "")} : t${o.track_id}`,x(px)+offset[0],py(yy)+offset[1]);}
  replace("ground-missing",...missing.map(text=>el("div","muted",text)));
}
function number(value, digits=2) { return value===undefined || value===null ? "—" : Number(value).toFixed(digits); }
function renderScores() {
  const score=current.scores;
  if(!score.available){replace("score-binding",el("div","score-message",score.reason));replace("score-table");return;}
  const binding=el("div","binding");binding.textContent=`${score.method} · ${score.report.name} · person_tracks ${Object.values(detail.provenance.track_versions||{}).map(v=>"v"+v).join(" / ")} · 入力binding: ${detail.provenance.store}\n${score.scope} 保存status: ${score.status} · ${score.binding}`;replace("score-binding",binding);
  if(!score.pairs.length){replace("score-table",el("p","score-message","選択raw IDの現在frameに保存候補pairなし。足元不足などで候補から外れたtrackはscoreを補完しません。"));return;}
  const table=el("table"),head=el("tr");for(const name of ["保存候補（[start,end)）","比較先camera / raw ID","距離中央値 m","geometry LLR","appearance LLR","cosine","共通frame"] )head.append(el("th","",name));table.append(head);
  const pairs=[...score.pairs].sort((a,b)=>(b.geometry||0)+(b.appearance||0)-(a.geometry||0)-(a.appearance||0));
  for(const pair of pairs){const row=el("tr",pair.active_now?"":"out-of-range");for(const [i,value] of [`${pair.from.camera} t${pair.from.track_id} [${pair.from.start},${pair.from.end})`,`${pair.to.camera} t${pair.to.track_id} [${pair.to.start},${pair.to.end})${pair.active_now?"":" / 現在区間外"}`,number(pair.median_m),number(pair.geometry),number(pair.appearance),number(pair.cosine,3),pair.shared_frames ?? "—"].entries())row.append(el("td",i<2?"pair-endpoint":"",String(value)));table.append(row);}replace("score-table",table);
}
function showError(error) { $("error").hidden=false;$("error").textContent=String(error.message || error);document.body.dataset.ready="error"; }
async function loadFrame() {
  if(state.loadingClip)return;
  const serial=++state.serial;document.body.dataset.ready="loading";$("error").hidden=true;
  state.frame=Math.max(0,Math.min(detail.frames-1,Math.floor(state.frame)));
  const query={clip:state.clip,frame:state.frame};if(state.report !== null)query.report=state.report;if(state.camera && state.track !== null){query.camera=state.camera;query.track=state.track;}
  try{const data=await api("/api/frame",query);if(serial!==state.serial)return;current=data;$("seek").value=state.frame;$("frame-number").value=state.frame;$("time").textContent=`${data.seconds.toFixed(3)} s / ${detail.frames}f`;renderTimelines();renderInspector();renderCourt();renderScores();updateUrl();await renderViews();if(serial===state.serial)document.body.dataset.ready="true";}catch(error){if(serial===state.serial)showError(error);}
}
async function loadClip() {
  const clip=state.clip, serial=++state.serial;state.loadingClip=true;document.body.dataset.ready="loading";
  try{const loaded=await api("/api/clip",{clip});if(serial!==state.serial)return;detail=loaded;state.loadingClip=false;$("clip-title").textContent=state.clip;$("reference").textContent=catalog.clips.find(c=>c.id===state.clip).reference;$("seek").max=detail.frames-1;$("frame-number").max=detail.frames-1;
    document.querySelectorAll(".clip-button").forEach(button=>button.classList.toggle("active",button.dataset.clip===state.clip));
    replace("coverage",...detail.coverage.map(c=>{const item=el("span");item.append(el("b","",c.camera+" "),document.createTextNode(`参照box照合 ${c.label_coverage===null?"不明":(c.label_coverage*100).toFixed(1)+"%"} · ${c.matched_boxes}/${c.label_boxes}`));return item;}));
    replace("coverage-detail",...detail.coverage.map(c=>el("p","muted",`${c.camera}: ラベルbox ${c.label_boxes} / raw実観測 ${c.raw_observed_boxes} / 照合 ${c.matched_boxes} (うち曖昧 ${c.ambiguous_matches}) / raw未対応 ${c.unmatched_raw_boxes??"不明"} / 参照未照合 ${c.unmatched_label_boxes??"不明"}。人物全体のrecallではない。`)));
    $("provenance").textContent=JSON.stringify(detail.provenance,null,2);await loadFrame();
  }catch(error){if(serial===state.serial){state.loadingClip=false;showError(error);}}
}
async function init() {
  try{catalog=await api("/api/catalog",{});$("scope").textContent=catalog.label_scope;$("counts").append(document.createTextNode(`${catalog.labelled_clips} / ${catalog.source_clips} clips`),el("small","",`${catalog.label_boxes.toLocaleString()} box labels · ${catalog.unlabelled_clips} clipsはラベルなし`));$("method").textContent=catalog.current_method;$("research").textContent=catalog.research;
    replace("clips",...catalog.clips.map(clip=>{const button=el("button","clip-button");button.dataset.clip=clip.id;button.append(el("strong","",clip.id),el("small","",`${clip.frames}f · ${clip.boxes.toLocaleString()} boxes`),el("small","",clip.reference));button.onclick=()=>{state.clip=clip.id;state.frame=0;state.camera=null;state.track=null;loadClip();};return button;}));
    const select=$("report");select.append(el("option","","未指定 / 新規推論なし"));select.options[0].value="";for(const report of catalog.score_reports){const option=el("option","",report.name);option.value=report.id;select.append(option);}if(state.report===null&&catalog.score_reports.length)state.report=0;select.value=state.report===null?"":String(state.report);select.onchange=()=>{state.report=select.value===""?null:Number(select.value);loadFrame();};
    state.clip=state.clip||catalog.clips[0].id;$("previous").onclick=()=>{state.frame--;loadFrame();};$("next").onclick=()=>{state.frame++;loadFrame();};$("seek").onchange=()=>{state.frame=Number($("seek").value);loadFrame();};$("frame-number").onchange=()=>{state.frame=Number($("frame-number").value);loadFrame();};$("clear-selection").onclick=()=>{state.camera=null;state.track=null;loadFrame();};await loadClip();
  }catch(error){showError(error);}
}
init();
