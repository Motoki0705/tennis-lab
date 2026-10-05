"use strict";

const $ = id => document.getElementById(id);
const state = {catalog:null, group:null, sequence:null, info:null, frame:0, track:null, image:null, playing:false, token:0, loadToken:0};
const palette = ["#008895", "#c36533", "#8261b2", "#478bc0", "#9e9333", "#a06584", "#5d9570", "#a27e5b"];
const roleNames = {player:"選手", non_player:"非選手", duplicate:"重複", unknown:"不明"};
const count = value => value === null ? "未保存" : value.toLocaleString("ja-JP");
function element(tag, text, className) {const item=document.createElement(tag); if(text!==undefined)item.textContent=text; if(className)item.className=className; return item;}
function option(value, text) {const item=element("option", text);item.value=value;return item;}
async function json(url) {const response=await fetch(url,{cache:"no-store"});if(!response.ok){let message=await response.text();try{message=JSON.parse(message).detail;}catch{}throw new Error(message);}return response.json();}
function fail(error) {$('error').textContent=String(error.message || error);$('error').hidden=false;}
function updateUrl() {if(!state.sequence)return;const query=new URLSearchParams({sequence:state.sequence.key, frame:state.frame, track:state.track});history.replaceState(null,"",`/?${query}`);}
function endpoint(suffix="") {return `/api/sequences/${encodeURIComponent(state.sequence.key)}${suffix}`;}
function trackMeta() {return state.sequence.tracks.find(track=>track.track_id===state.track);}
function color(id) {return palette[state.sequence.tracks.findIndex(track=>track.track_id===id)%palette.length];}

function renderCatalog() {
  const group=state.group;
  $('inventory').replaceChildren();
  const unavailable=group.records.filter(record=>!record.available).length;
  for(const text of [`${group.clips} clip`, `${group.available} raw保存 / 対応形式`, ...(unavailable?[`${unavailable} 未生成 / 非対応`]:[]), ...Object.entries(group.status_counts).map(([key,value])=>`${key} ${value}`)]) $('inventory').append(element("span",text));
  $('familyNote').textContent=group.description;
  for(const [id,field] of [["sourceFilter","source"],["splitFilter","split"]]) {
    const previous=$(id).value;$(id).replaceChildren(option("","すべて"));
    for(const value of [...new Set(group.records.map(record=>record[field]))])$(id).append(option(value,value));
    $(id).value=previous;
  }
  filterRecords();
}
function filterRecords(preferred) {
  const needle=$('search').value.trim().toLowerCase(), previous=preferred || $('sequence').value;
  const records=state.group.records.filter(record=>(!$('sourceFilter').value || record.source===$('sourceFilter').value) && (!$('splitFilter').value || record.split===$('splitFilter').value) && `${record.clip_id} ${record.camera_id}`.toLowerCase().includes(needle));
  $('sequence').replaceChildren();
  for(const record of records) {
    const item=option(record.key,`${record.clip_id} · ${record.camera_id} · ${record.status}${record.available?"":` · ${record.reason}`}`);
    item.disabled=!record.available;$('sequence').append(item);
  }
  const chosen=records.find(record=>record.key===previous && record.available) || records.find(record=>record.available);
  if(!chosen){$('sequence').append(option("","閲覧可能なclipなし"));$('sequence').value="";return;}
  $('sequence').value=chosen.key;
}

async function loadSequence(key, frame=0, track=null) {
  if(!key)return;
  stopPlayback();const token=++state.loadToken;++state.token;
  $('error').hidden=true;$('loading').hidden=false;$('review').hidden=true;
  try {
    const data=await json(`/api/sequences/${encodeURIComponent(key)}`);
    if(token!==state.loadToken)return;
    state.sequence=data;state.track=track===null?data.tracks.reduce((best,item)=>item.observed_count>(best?.observed_count ?? -1)?item:best,null)?.track_id:track;
    if(!data.tracks.some(item=>item.track_id===state.track))state.track=data.tracks[0]?.track_id ?? null;
    $('clipTitle').textContent=`${data.clip_id} / ${data.camera_id}`;
    const method=data.metadata.tracking_profile?.method || "方式未記録";
    $('sampleInfo').textContent=`${data.metadata.source} · ${data.metadata.split} · ${data.width}×${data.height} · ${data.frame_count.toLocaleString()} frames · ${method} · ${data.metadata.schema}`;
    $('reviewStatus').textContent=data.metadata.status;
    $('idCount').textContent=count(data.tracks.length);$('observedCount').textContent=count(data.observations);
    $('syntheticCount').textContent=count(data.saved_synthetic);
    if(data.metadata.reference) {$('labelCountTitle').textContent="部分参照box";$('labelCount').textContent=count(data.metadata.reference.boxes);}
    else {$('labelCountTitle').textContent="ID区間の被覆";$('labelCount').textContent=data.labelled_observations?`${count(data.labelled_observations)} / ${count(data.observations)}`:"未採用 / 未指定";}
    $('selectedCount').textContent=count(data.selected_observations);
    $('drawMode').options[1].disabled=data.selected_observations===null;$('drawMode').value="raw";
    $('synthetic').disabled=data.saved_synthetic===null;$('synthetic').checked=false;
    $('referenceToggle').hidden=!data.metadata.reference;$('reference').checked=false;
    $('syntheticNote').textContent=data.metadata.synthetic_note;$('selectionNote').textContent=data.metadata.selection_note;
    $('frameSlider').max=data.frame_count-1;$('frameNumber').max=data.frame_count-1;
    $('axisMiddle').textContent=`f${Math.floor((data.frame_count-1)/2)}`;$('axisEnd').textContent=`f${data.frame_count-1}`;
    $('labelNote').textContent=data.metadata.label_note;$('referenceBinding').textContent=data.metadata.reference?.binding || "";
    $('provenance').textContent=JSON.stringify({provenance:data.metadata.provenance,tracking_profile:data.metadata.tracking_profile,reference:data.metadata.reference,saved_link_candidates:data.metadata.saved_link_candidates},null,2);
    renderTimeline();$('loading').hidden=true;$('review').hidden=false;
    await showFrame(Math.max(0,Math.min(data.frame_count-1,Number(frame))));
  } catch(error) {if(token===state.loadToken){$('loading').hidden=true;fail(error);}}
}

function renderTimeline() {
  const data=state.sequence;$('timeline').replaceChildren();
  for(const track of [...data.tracks].sort((a,b)=>b.observed_count-a.observed_count || a.track_id-b.track_id)) {
    const row=element("div",undefined,"track-row");row.dataset.track=track.track_id;
    const name=element("button",undefined,"track-name");name.append(element("b",`ID ${track.track_id}`));
    const identities=[...new Set(track.intervals.map(item=>item.player_id || roleNames[item.role]))].join(" / ");
    name.append(element("span",`${track.observed_count} 観測`));name.append(element("small",identities || `AFLink元ID [${track.source_ids.join(", ")}]`));name.title=`raw ID ${track.track_id}; source IDs ${track.source_ids.join(", ")}; ${identities}`;
    name.onclick=()=>selectTrack(track.track_id);row.append(name);
    const strip=element("div",undefined,"track-strip");strip.style.setProperty("--color",color(track.track_id));
    function bar(start,stop,className,title){const segment=element("span",undefined,`bar ${className}`);segment.style.left=`${start/data.frame_count*100}%`;segment.style.width=`${(stop-start)/data.frame_count*100}%`;segment.title=title;strip.append(segment);return segment;}
    for(const [start,stop] of track.observed_runs)bar(start,stop,"observed",`実観測 [${start}, ${stop})`);
    for(const [start,stop] of track.synthetic_runs)bar(start,stop,"synthetic",`保存GSI / 未観測 [${start}, ${stop})`);
    for(const item of track.intervals){const segment=bar(item.start_frame,item.stop_frame,`interval ${item.role}`,`${roleNames[item.role]} ${item.player_id || ""} [${item.start_frame}, ${item.stop_frame}) / 実観測にだけ適用`);if(item.player_id && item.role==="player")segment.style.background=palette[(Number(item.player_id.split('_')[1])-1)%palette.length];}
    const cursor=element("span",undefined,"cursor");strip.append(cursor);
    strip.onclick=event=>{const bounds=strip.getBoundingClientRect();const frame=Math.min(data.frame_count-1,Math.max(0,Math.floor((event.clientX-bounds.left)/bounds.width*data.frame_count)));state.track=track.track_id;showFrame(frame).catch(fail);};
    row.append(strip);$('timeline').append(row);
  }
  highlightTimeline();
}
function highlightTimeline(){for(const row of $('timeline').children){row.classList.toggle("selected",Number(row.dataset.track)===state.track);row.querySelector('.cursor').style.left=`${(state.frame+.5)/state.sequence.frame_count*100}%`;}}
function selectTrack(track){state.track=track;if(state.info){renderScene();renderInspection();}highlightTimeline();updateUrl();}

async function showFrame(frame) {
  if(!state.sequence || !Number.isInteger(frame) || frame<0 || frame>=state.sequence.frame_count)return;
  const token=++state.token;$('error').hidden=true;$('frameLoading').hidden=false;
  try {
    const [info,response]=await Promise.all([json(endpoint(`/frames/${frame}`)),fetch(endpoint(`/frames/${frame}/image`),{cache:"no-store"})]);
    if(!response.ok){const error=await response.json();throw new Error(error.detail);}
    const blob=await response.blob(), url=URL.createObjectURL(blob), image=new Image();
    try{image.src=url;await image.decode();}finally{URL.revokeObjectURL(url);}
    if(token!==state.token)return;
    state.info=info;state.image=image;state.frame=frame;
    $('frameSlider').value=frame;$('frameNumber').value=frame;
    const [numerator,denominator=1]=info.time_base.split('/').map(Number);
    $('frameTime').textContent=`${(info.pts*numerator/denominator).toFixed(3)} s · ${state.sequence.frame_count} f`;
    renderScene();renderInspection();highlightTimeline();updateUrl();
    $('frameLoading').hidden=true;
  }catch(error){if(token===state.token){$('frameLoading').hidden=true;stopPlayback();fail(error);}throw error;}
}
function renderScene() {
  const canvas=$('scene'), data=state.sequence, context=canvas.getContext('2d');canvas.width=data.width;canvas.height=data.height;
  context.drawImage(state.image,0,0);const stroke=Math.max(2,Math.round(data.width/700)), font=Math.max(12,Math.round(data.width/110));
  context.lineWidth=stroke;context.font=`600 ${font}px system-ui`;
  function drawBox(box,label,paint,dashed,emphasis=false){context.strokeStyle=paint;context.lineWidth=emphasis?stroke*1.7:stroke;context.setLineDash(dashed?[stroke*4,stroke*3]:[]);context.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);context.setLineDash([]);const width=context.measureText(label).width+8,x=Math.max(0,Math.min(box[0],data.width-width)),y=Math.max(font+6,box[1]);context.fillStyle=paint;context.fillRect(x,y-font-6,width,font+6);context.fillStyle="#fff";context.fillText(label,x+4,y-4);}
  for(const track of state.info.tracks) {
    if(!track.box || (track.state==="interpolated" && (!$('synthetic').checked || $('drawMode').value==="selected")))continue;
    if($('drawMode').value==="selected" && !track.selected)continue;
    const label=`R${track.track_id}${track.state==="interpolated"?" · G":""}`;
    drawBox(track.box,label,track.state==="interpolated"?"#c16c13":color(track.track_id),track.state==="interpolated",track.track_id===state.track);
  }
  if($('reference').checked)for(const item of state.info.reference_boxes)drawBox(item.box,`参照 ${item.person || "不明"}`,"#435971",true);
}
$('scene').onclick=event=>{if(!state.info)return;const bounds=$('scene').getBoundingClientRect(),x=(event.clientX-bounds.left)/bounds.width*state.sequence.width,y=(event.clientY-bounds.top)/bounds.height*state.sequence.height;const hits=state.info.tracks.filter(track=>track.box && (track.state==="observed" || ($('synthetic').checked && $('drawMode').value==="raw")) && ($('drawMode').value!=="selected" || track.selected) && x>=track.box[0] && x<=track.box[2] && y>=track.box[1] && y<=track.box[3]);hits.sort((a,b)=>(a.box[2]-a.box[0])*(a.box[3]-a.box[1])-(b.box[2]-b.box[0])*(b.box[3]-b.box[1]) || a.track_id-b.track_id);if(hits.length)selectTrack(hits[0].track_id);};
function renderInspection() {
  const track=state.info.tracks.find(item=>item.track_id===state.track), meta=trackMeta();
  if(!track){$('trackTitle').textContent="raw IDなし";return;}
  $('trackTitle').textContent=`raw ID ${track.track_id} · f${state.frame}`;
  $('frameState').textContent={observed:"実観測",interpolated:"保存GSI / 未観測",missing:"box欠損"}[track.state];$('frameState').className=`status ${track.state}`;
  const canvas=$('crop'),context=canvas.getContext('2d');canvas.width=600;canvas.height=430;context.clearRect(0,0,canvas.width,canvas.height);$('noCrop').hidden=!!track.box;
  if(track.box){const [x1,y1,x2,y2]=track.box;const x=Math.max(0,Math.floor(x1)),y=Math.max(0,Math.floor(y1)),w=Math.min(state.sequence.width,Math.ceil(x2))-x,h=Math.min(state.sequence.height,Math.ceil(y2))-y;
    if(w>0 && h>0){const scale=Math.min(570/w,400/h),cw=w*scale,ch=h*scale;context.drawImage(state.image,x,y,w,h,(600-cw)/2,(430-ch)/2,cw,ch);}
  }
  $('cropCaption').textContent=track.state==="interpolated"?"保存GSI boxの画像範囲。検出・pose観測ではありません。":track.state==="missing"?"未保存boxのcropは作りません。前後の実観測を確認できます。":"元画像の観測boxを縦横比を保って拡大。";
  const assignment=track.assignment;
  const selection=track.state!=="observed"?"未観測のため対象外":track.selected===null?"未判定 / 未採用":track.selected?"保持された実観測":"保持対象外";
  const details=[["匿名選手ID",assignment?.player_id || "未割当"],["役割",assignment?roleNames[assignment.role]:"未判定"],["選別",selection],["元検出row",track.detection_row===null?"未保存 / 未観測":String(track.detection_row)],["AFLink元ID",`[${meta.source_ids.join(", ")}]`],["コートgroup",track.group_id===null?"未割当":`group ${track.group_id} (raw IDと別)`]];
  $('trackDetails').replaceChildren();for(const [label,value] of details){$('trackDetails').append(element("dt",label),element("dd",value));}
  for(const [id,target] of [["previousObserved",track.previous_observed],["nextObserved",track.next_observed]]){$(id).disabled=target===null;$(id).onclick=()=>showFrame(target).catch(fail);}
  $('assignmentDetail').replaceChildren();
  if(meta.intervals.length){const intervals=element("div");for(const item of meta.intervals){const button=element("button",`${item.player_id || roleNames[item.role]} f${item.start_frame}–${item.stop_frame-1}`);button.title="保存区間の根拠frameへ移動";button.onclick=()=>showFrame(item.evidence_frames[0]).catch(fail);intervals.append(button);}$('assignmentDetail').append(intervals);}
  if(assignment){$('assignmentDetail').append(element("strong",`${roleNames[assignment.role]} ${assignment.player_id || ""} · 保存区間 [${assignment.start_frame}, ${assignment.stop_frame})`));$('assignmentDetail').append(element("p",assignment.reason));const evidence=element("div");for(const frame of assignment.evidence_frames){const button=element("button",`f${frame}`);button.onclick=()=>showFrame(frame).catch(fail);evidence.append(button);}$('assignmentDetail').append(evidence);}
  else $('assignmentDetail').append(element("p",state.sequence.metadata.selection_kind==="gpt_interval"?"このframeには採用ID区間の実観測がありません。":"匿名選手ID区間は保存されていません。"));
  renderGaps();
}
function renderGaps() {
  const events=state.sequence.events.filter(event=>event.track_id===state.track);
  $('gapTitle').textContent=`ID ${state.track} の実観測の途切れ · ${events.length} 区間`;$('gaps').replaceChildren();
  if(!events.length){$('gaps').append(element("p","最初と最後の実観測の間に欠測区間はありません。","muted"));return;}
  for(const event of events){const item=element("div",undefined,`gap${state.frame>=event.start && state.frame<event.stop?" active":""}`);
    item.append(element("span",`[${event.start}, ${event.stop}) · ${event.stop-event.start} f / 保存GSI ${event.synthetic_frames} f`));
    for(const [label,frame] of [[`前 f${event.previous}`,event.previous],["欠測へ",event.start],[`後 f${event.next}`,event.next]]){const button=element("button",label);button.onclick=()=>{if(frame===event.start && event.synthetic_frames && !$('synthetic').disabled)$('synthetic').checked=true;showFrame(frame).catch(fail);};item.append(button);}$('gaps').append(item);
  }
}
function stopPlayback(){state.playing=false;$('play').textContent="再生 · 4fps";}
async function playback(){state.playing=!state.playing;$('play').textContent=state.playing?"停止":"再生 · 4fps";while(state.playing){if(state.frame+1>=state.sequence.frame_count){stopPlayback();break;}try{await showFrame(state.frame+1);}catch{break;}await new Promise(resolve=>setTimeout(resolve,250));}}
$('play').onclick=playback;$('previous').onclick=()=>{stopPlayback();showFrame(state.frame-1).catch(fail);};$('next').onclick=()=>{stopPlayback();showFrame(state.frame+1).catch(fail);};
$('frameSlider').oninput=()=>{stopPlayback();showFrame(Number($('frameSlider').value)).catch(fail);};$('frameNumber').onchange=()=>{stopPlayback();showFrame(Number($('frameNumber').value)).catch(fail);};
$('drawMode').onchange=()=>{if(state.info){$('synthetic').disabled=state.sequence.saved_synthetic===null || $('drawMode').value==="selected";renderScene();}};for(const id of ["synthetic","reference"])$(id).onchange=()=>{if(state.info)renderScene();};
$('sequence').onchange=()=>loadSequence($('sequence').value);
for(const id of ["sourceFilter","splitFilter","search"])$(id).oninput=()=>{const previous=$('sequence').value;filterRecords();if($('sequence').value && $('sequence').value!==previous)loadSequence($('sequence').value);if(!$('sequence').value){stopPlayback();++state.loadToken;++state.token;$('review').hidden=true;$('loading').hidden=true;fail(new Error("この検索条件には閲覧可能な保存trackingがありません。"));}};
$('family').onchange=()=>{state.group=state.catalog.groups[Number($('family').value)];$('sourceFilter').value="";$('splitFilter').value="";$('search').value="";renderCatalog();loadSequence($('sequence').value);};
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","BUTTON"].includes(document.activeElement?.tagName) || !state.sequence)return;if(event.key==="ArrowRight"){event.preventDefault();stopPlayback();showFrame(state.frame+1).catch(fail);}if(event.key==="ArrowLeft"){event.preventDefault();stopPlayback();showFrame(state.frame-1).catch(fail);}});
async function init(){try{state.catalog=await json('/api/catalog');const query=new URLSearchParams(location.search),key=query.get('sequence');let chosen=state.catalog.groups.findIndex(group=>group.records.some(record=>record.key===key));if(chosen<0)chosen=0;state.catalog.groups.forEach((group,index)=>$('family').append(option(index,group.name)));$('family').value=chosen;state.group=state.catalog.groups[chosen];renderCatalog();filterRecords(key);await loadSequence($('sequence').value,Number(query.get('frame') || 0),query.has('track')?Number(query.get('track')):null);}catch(error){fail(error);}}
init();
