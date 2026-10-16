import {eventsAt, finitePoint, imageTransform, observationAt} from "./blcs-observation.mjs";

const panel = document.createElement("section");
panel.className = "blcs-inspection";
panel.setAttribute("aria-label", "保存2D観測と3D教師の検品");
panel.innerHTML = `
  <header><h2>保存2D観測と3D教師</h2>
    <p class="blcs-caption">物理シミュレーションの合成truth · RGBは保存されていません</p>
    <p id="blcs-inventory" class="blcs-source">single_object / physical_v1</p>
  </header>
  <p id="blcs-status" class="blcs-status" role="status">シーンを選択してください</p>
  <div id="blcs-content" class="blcs-content" hidden>
    <p id="blcs-frame" class="blcs-frame"></p>
    <dl class="blcs-metrics">
      <dt>位置 [m]</dt><dd id="blcs-position"></dd>
      <dt>速度 [m/s]</dt><dd id="blcs-velocity"></dd>
      <dt>保存正規化位置</dt><dd id="blcs-position-norm"></dd>
    </dl>
    <p id="blcs-normalization" class="blcs-normalization"></p>
    <p class="blcs-caption">UVは画像の左上が(0,0)、右下が(1,1)。保存CourtKP20を全点表示（学習入力の点数選択とは別）。</p>
    <div class="blcs-legend">
      <span><b style="color:#dc6c1b">●</b> 保存ball</span>
      <span><b style="color:#236ca5">＋</b> 3D再投影</span>
      <span><b style="color:#157a6e">●</b> 保存court</span>
      <span>○ 非可視 · 破線枠＝画像領域</span>
    </div>
    <div class="blcs-options">
      <label>表示 <select id="blcs-camera-select" aria-label="検品カメラ"><option value="all">全カメラ</option></select></label>
      <label><input id="blcs-kp-labels" type="checkbox" checked>court番号</label>
      <label><input id="blcs-trail" type="checkbox" checked>直近30frameの保存軌跡</label>
    </div>
    <div id="blcs-cameras" class="blcs-cameras"></div>
    <section class="blcs-events" aria-label="保存イベント">
      <h3 id="blcs-event-now"></h3>
      <div id="blcs-event-buttons" class="blcs-event-buttons"></div>
      <p class="blcs-caption">hit＝保存shot開始。timeはframe / fps_out。returnは保存時刻（次hitの実行は保証しない）。区間外bounce候補は再生イベントに含めません。</p>
      <details><summary id="blcs-event-summary"></summary>
        <table class="blcs-event-table"><thead><tr><th>shot</th><th>event</th><th>frame / 秒</th><th>保存状態</th><th>shot / return type</th></tr></thead><tbody id="blcs-event-table"></tbody></table>
      </details>
    </section>
  </div>`;
document.querySelector(".workspace").append(panel);

const byId = id => document.getElementById(id);
const dom = Object.fromEntries(["status", "content", "inventory", "frame", "position", "velocity", "position-norm", "normalization", "cameras", "event-now", "event-buttons", "event-summary", "event-table", "kp-labels", "trail", "camera-select"].map(key => [key, byId(`blcs-${key}`)]));
const state = {scene:null, evidence:null, frame:0, cards:[], controller:null, token:0};
const vec = value => Array.isArray(value) ? value.map(v => Number.isFinite(v) ? v.toFixed(3) : "不明").join(", ") : "保存ファイルなし";
const metric = value => Number.isFinite(value) ? value.toFixed(4) : "不明";
const precise = value => Number.isFinite(value) ? value === 0 ? "0" : value.toExponential(2) : "不明";
const statusLabels = {on_trajectory:"軌道区間内", after_shot:"次shot以降の候補", outside_scene:"scene範囲外", before_shot:"shot開始前", not_recorded:"未記録 (-1)", missing:"fieldなし"};

function textElement(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function buildCameras(evidence) {
  dom.cameras.replaceChildren();
  dom["camera-select"].replaceChildren(new Option("全カメラ", "all"));
  for (const camera of evidence.cameras) dom["camera-select"].append(new Option(camera.id, camera.id));
  state.cards = evidence.cameras.map(camera => {
    const card = document.createElement("article");
    card.className = "blcs-camera";
    card.dataset.camera = camera.id;
    const heading = textElement("div", "", "blcs-camera-heading");
    const badge = textElement("span", "", "blcs-badge");
    heading.append(textElement("b", camera.id), badge);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", `${camera.id} 保存ball・courtと3D再投影`);
    const readout = textElement("div", "", "blcs-camera-readout");
    const ball = textElement("span", ""), court = textElement("span", "");
    readout.append(ball, court);
    const note = textElement("div", "", "blcs-camera-note");
    const details = document.createElement("details");
    details.append(textElement("summary", `camera・全scene検査 · ${camera.image_size.join("×")}`));
    details.append(textElement("pre", `C [m]: ${vec(camera.center_m)}\nR (world→camera): ${JSON.stringify(camera.rotation_world_to_camera)}\nK [px]: ${JSON.stringify(camera.intrinsics_px)}\n保存ball可視: ${camera.ball.summary.saved_visible ?? "不明"} / ${camera.ball.summary.total}\nball再投影 max [px]: ${metric(camera.ball.summary.max_error_px)}\nball vis不一致: ${camera.ball.summary.visibility_mismatches ?? "不明"}\ncourt vis不一致: ${camera.court.summary.visibility_mismatches ?? "不明"}\n非有限ball UV: ${camera.ball.summary.invalid_saved_uv ?? "不明"}`));
    card.append(heading, canvas, readout, note, details);
    dom.cameras.append(card);
    return {camera, card, badge, canvas, ball, court, note};
  });
}

function buildEventTable(events) {
  dom["event-table"].replaceChildren();
  for (const event of events ?? []) {
    const row = document.createElement("tr");
    row.dataset.status = event.status;
    row.append(textElement("td", String(event.shot_index ?? "不明")), textElement("td", event.kind),
      textElement("td", `${event.frame ?? "不明"} / ${Number.isFinite(event.time_seconds) ? event.time_seconds.toFixed(3) : "–"}`), textElement("td", statusLabels[event.status]), textElement("td", `${event.shot_type ?? "不明"} / ${event.return_type ?? "不明"}`));
    dom["event-table"].append(row);
  }
  const candidates = (events ?? []).filter(e => ["after_shot", "outside_scene", "before_shot"].includes(e.status)).length;
  dom["event-summary"].textContent = events === null ? "shots metadataなし（イベント不明）" : `保存イベント ${events.length}件（区間外候補 ${candidates}件を含む）`;
}

function seek(frame) {
  const play = byId("play");
  if (play.title === "一時停止") play.click();
  const scrub = byId("scrub");
  scrub.value = String(frame);
  scrub.dispatchEvent(new Event("input", {bubbles:true}));
}

function drawPoint(context, point, transform, {color, radius=3, filled=false, cross=false, label=null}) {
  if (!finitePoint(point)) return;
  const [x,y] = transform.map(point);
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = cross ? 1.3 : 1.2;
  context.beginPath();
  if (cross) { context.moveTo(x-4,y); context.lineTo(x+4,y); context.moveTo(x,y-4); context.lineTo(x,y+4); }
  else context.arc(x,y,radius,0,Math.PI*2);
  if (filled) context.fill();
  context.stroke();
  if (label !== null) { context.font = "9px monospace"; context.fillText(String(label),x+4,y-3); }
}

function drawCamera(card, frame, observation) {
  const {camera, canvas} = card;
  const context = canvas.getContext("2d");
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  if (!width || !height) return;
  if (canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)) {
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  }
  context.setTransform(ratio,0,0,ratio,0,0);
  context.clearRect(0,0,width,height);
  const transform = imageTransform(camera, observation, width, height);
  const [x0,y0] = transform.map([0,0]), [x1,y1] = transform.map([1,1]);
  context.fillStyle = "#ffffff";
  context.fillRect(x0,y0,x1-x0,y1-y0);
  context.strokeStyle = "#a1adb6";
  context.setLineDash([4,3]);
  context.strokeRect(x0,y0,x1-x0,y1-y0);
  context.setLineDash([]);
  const court = camera.court;
  for (const [a,b] of state.scene.court.edges) {
    const pa = court.saved_uv?.[a], pb = court.saved_uv?.[b];
    if (!finitePoint(pa) || !finitePoint(pb)) continue;
    const [ax,ay] = transform.map(pa), [bx,by] = transform.map(pb);
    context.strokeStyle = "#b0c8bd"; context.lineWidth = 1;
    context.beginPath(); context.moveTo(ax,ay); context.lineTo(bx,by); context.stroke();
  }
  for (let index=0; index<20; index++) {
    drawPoint(context,court.projected_uv[index],transform,{color:"#84a7c3",cross:true});
    drawPoint(context,court.saved_uv?.[index],transform,{color:"#157a6e",radius:2,filled:court.saved_visibility?.[index] === true,label:dom["kp-labels"].checked ? index : null});
  }
  if (dom.trail.checked && camera.ball.saved_uv !== null) {
    context.strokeStyle = "#e9b98d"; context.lineWidth = 1.2;
    context.beginPath();
    let connected = false;
    for (let index=Math.max(0,frame-30); index<=frame; index++) {
      const uv = camera.ball.saved_uv[index];
      if (!finitePoint(uv) || camera.ball.saved_visibility?.[index] !== true) {connected=false;continue;}
      const [x,y] = transform.map(uv);
      if (connected) context.lineTo(x,y); else context.moveTo(x,y);
      connected = true;
    }
    context.stroke();
  }
  drawPoint(context,observation.uv,transform,{color:"#dc6c1b",radius:4,filled:observation.visible === true});
  drawPoint(context,observation.projected,transform,{color:"#236ca5",cross:true});
  if (transform.expanded) {
    context.fillStyle = "#8b4a0e"; context.font = "10px sans-serif";
    context.fillText("画像外の座標も表示",8,height-4);
  }
}

function updateCamera(card, frame) {
  const {camera} = card;
  const o = observationAt(camera,frame);
  const labels = {visible:"可視",not_visible:"非可視",unknown_visibility:"可視性不明",missing_uv:"UV fileなし",invalid_uv:"UV非有限"};
  const locations = {in_frame:"画像内",out_of_frame:"画像外",behind_camera:"camera後方"};
  card.badge.textContent = `${labels[o.state]} · ${locations[o.location]}`;
  card.badge.dataset.tone = o.state === "visible" && !o.mismatch ? "normal" : o.mismatch || o.state === "invalid_uv" ? "error" : "warning";
  card.card.dataset.warning = String(card.badge.dataset.tone !== "normal");
  card.ball.textContent = `UV ${vec(o.uv)} · Δ ${metric(o.errorPx)} px`;
  const vis = camera.court.summary.saved_visible;
  card.court.textContent = `court ${vis ?? "不明"}/20 可視 · Δmax ${metric(camera.court.summary.max_error_px)} px`;
  const notes = [];
  if (o.mismatch) notes.push("保存visと再投影の可視性が不一致");
  if (camera.ball.saved_uv === null) notes.push("ball UV保存ファイルなし");
  if (camera.ball.saved_visibility === null) notes.push("ball vis保存ファイルなし");
  if (camera.court.saved_uv === null) notes.push("court UV保存ファイルなし");
  if (camera.court.saved_visibility === null) notes.push("court vis保存ファイルなし");
  card.note.textContent = notes.join(" / ");
  card.note.hidden = notes.length === 0;
  drawCamera(card,frame,o);
}

function render() {
  const evidence = state.evidence;
  if (!evidence) return;
  const frame = state.frame;
  panel.dataset.frame = String(frame);
  panel.dataset.scene = evidence.scene_id;
  dom.frame.textContent = `frame ${frame} / ${evidence.frame_count-1} · ${(frame/evidence.fps).toFixed(3)} s · ${evidence.fps} fps`;
  dom.position.textContent = vec(evidence.ball.position_m[frame]);
  dom.velocity.textContent = vec(evidence.ball.velocity_mps?.[frame]);
  dom["position-norm"].textContent = vec(evidence.ball.position_normalized?.[frame]);
  const selected = dom["camera-select"].value;
  dom.cameras.dataset.single = String(selected !== "all");
  state.cards.forEach(card => {
    card.card.hidden = selected !== "all" && selected !== card.camera.id;
    if (!card.card.hidden) updateCamera(card,frame);
  });
  const current = eventsAt(evidence.events,frame);
  dom["event-now"].textContent = current.length ? `現在: ${current.map(e=>`shot ${e.shot_index ?? "不明"} ${e.kind}`).join(" / ")}` : evidence.events === null ? "現在: イベント不明" : "現在: hit/bounce等の保存イベントなし";
  const active = (evidence.events ?? []).filter(e=>e.status === "on_trajectory");
  const hits = active.filter(e=>e.kind === "hit");
  const shot = hits.filter(e=>e.frame<=frame).at(-1)?.shot_record_index;
  const nearby = active.filter(e=>e.shot_record_index === shot);
  dom["event-buttons"].replaceChildren();
  const previous = hits.filter(e=>e.frame<frame).at(-1), next = hits.find(e=>e.frame>frame);
  for (const e of [previous ? {...previous,label:"← hit"} : null, ...nearby, next ? {...next,label:"hit →"} : null].filter(Boolean)) {
    const button = textElement("button", `${e.label ?? e.kind} f${e.frame}`);
    button.type = "button";
    button.dataset.frame = String(e.frame);
    button.setAttribute("aria-pressed",String(e.frame === frame));
    button.addEventListener("click",()=>seek(e.frame));
    dom["event-buttons"].append(button);
  }
}

async function load(scene) {
  state.scene = scene;
  const token = state.token;
  state.controller = new AbortController();
  const query = new URLSearchParams({form:scene.form,scene:scene.scene_id,revision:scene.revision});
  try {
    const response = await fetch(`/api/inspection?${query}`,{signal:state.controller.signal});
    const evidence = await response.json();
    if (!response.ok) throw Error(evidence.detail ?? `HTTP ${response.status}`);
    if (token !== state.token) return;
    if (evidence.scene_id !== scene.scene_id || evidence.revision !== scene.revision) throw Error("3D表示と検品データのrevisionが一致しません");
    state.evidence = evidence;
    const counts = evidence.dataset.split_counts;
    dom.inventory.textContent = `single_object · ${evidence.dataset.scene_count} scenes · train ${counts.train ?? "不明"} / val ${counts.val ?? "不明"} / test ${counts.test ?? "不明"} · このscene: ${evidence.dataset.scene_splits.join(", ") || "split未登録"}`;
    const {position,velocity} = evidence.ball.normalization;
    const labels = {ok:"一致",mismatch:"不一致",nonfinite:"非有限値",missing:"保存ペアなし"};
    dom.normalization.textContent = `全XYZ × ${evidence.contract.normalization.scale_xyz_m[0]} → m / m/s · 保存ペア検査: 位置 ${labels[position.status]} (max ${precise(position.max_abs_error)} m) / 速度 ${labels[velocity.status]} (max ${precise(velocity.max_abs_error)} m/s)`;
    dom.normalization.dataset.warning = String(position.status !== "ok" || velocity.status !== "ok");
    buildCameras(evidence);
    buildEventTable(evidence.events);
    dom.status.hidden = true;
    dom.content.hidden = false;
    render();
  } catch (error) {
    if (error.name === "AbortError" || token !== state.token) return;
    state.evidence = null;
    dom.content.hidden = true;
    dom.status.hidden = false;
    dom.status.textContent = `検品データの読取エラー: ${error.message}`;
  }
}

window.addEventListener("dataset-review:scene",({detail})=>{
  if (detail.phase === "loaded") {load(detail.scene);return;}
  state.token += 1;
  state.controller?.abort();
  state.scene = null; state.evidence = null; state.frame = 0; state.cards = [];
  delete panel.dataset.frame; delete panel.dataset.scene;
  dom.content.hidden = true;
  dom.status.hidden = false;
  dom.status.textContent = detail.phase === "error" ? "シーンを読み込めません。前の検品データを消去しました。" : "保存観測を読み込み中…";
  dom.inventory.textContent = "single_object / physical_v1";
});
window.addEventListener("dataset-review:frame",({detail})=>{
  if (detail.sceneId !== state.scene?.scene_id || detail.form !== state.scene?.form) return;
  state.frame = detail.frame;
  render();
});
dom["kp-labels"].addEventListener("change",render);
dom.trail.addEventListener("change",render);
dom["camera-select"].addEventListener("change",render);
new ResizeObserver(()=>render()).observe(panel);
