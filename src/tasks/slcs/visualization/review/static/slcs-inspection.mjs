import { drawOverlay, poseCrop, ballResidualPixels } from "./slcs-overlay.mjs";
import { fitOrbit, framingPoints } from "./slcs-framing.mjs";
import { PRESETS } from "/static/scene.mjs";

const panel = document.createElement("section");
panel.className = "slcs-inspector";
panel.setAttribute("aria-label", "SLCS元画像・観測・教師の検査");
// This markup is constant. Dataset strings are only assigned through textContent.
panel.innerHTML = `
  <h2>元画像・2D観測・疑似教師</h2>
  <p class="slcs-notice">教師はモデル・幾何処理による推定値です。実測GTは含まれません。</p>
  <div class="slcs-toolbar"><label>入力camera <select id="slcs-camera" aria-label="入力カメラ"></select></label>
    <label><input id="slcs-pose" type="checkbox" checked> pose</label>
    <label><input id="slcs-ball" type="checkbox" checked> ball 2D</label>
    <label><input id="slcs-court" type="checkbox" checked> court</label>
    <label><input id="slcs-projection" type="checkbox" checked> 近似3D投影</label></div>
  <p id="slcs-sync">clipを選択してください</p>
  <canvas id="slcs-image" class="slcs-picture" aria-label="保存clipのRGBと観測overlay"></canvas>
  <p id="slcs-image-status" class="slcs-source-image-status" hidden></p>
  <div class="slcs-legend"><b style="color:#3a6fb0">● P0</b><b style="color:#d1623f">● P1</b><b style="color:#9c8514">● ball観測</b><b style="color:#168c74">● court</b><b style="color:#8843bf">○ 疑似教師投影</b></div>
  <p id="slcs-calibration" class="slcs-muted"></p>
  <h3>このフレームの教師quality</h3>
  <table class="slcs-quality"><thead><tr><th>教師</th><th>有効 / weight</th><th>観測と棄却理由</th><th>XYZ(m) / yaw</th></tr></thead><tbody id="slcs-quality"></tbody></table>
  <div class="slcs-crop-row"><div>
    <div class="slcs-toolbar"><label>pose拡大 <select id="slcs-crop-slot" aria-label="拡大する選手"><option value="1">P1（奥）</option><option value="0">P0（手前）</option></select></label></div>
    <canvas id="slcs-crop" width="440" height="276" aria-label="選択選手のpose crop"></canvas></div>
    <div><p id="slcs-crop-note" class="slcs-muted"></p><p id="slcs-ball-residual" class="slcs-muted"></p><p id="slcs-policy" class="slcs-muted"></p><p class="slcs-muted">不可視 = 観測なし（物体の不在は未確認）。枠外観測も区別します。</p></div></div>
  <h3>教師mask 全clip <span class="slcs-muted">P0 / P1 / ball</span></h3>
  <canvas id="slcs-timeline" class="slcs-timeline" width="1200" height="90" tabindex="0" role="slider" aria-label="quality timeline" aria-valuemin="0"></canvas>
  <div class="slcs-gap-controls"><button id="slcs-prev-gap">前の無効frame</button><button id="slcs-next-gap">次の無効frame</button><span>橙=無効 / 緑=有効</span></div>
  <p id="slcs-thresholds" class="slcs-muted"></p>
  <details id="slcs-source"><summary>source・教師provenance・学習準備</summary><dl id="slcs-provenance"></dl></details>
  <p id="slcs-error" class="slcs-error" role="status" hidden></p>`;
document.querySelector(".workspace").append(panel);
const byId = (id) => document.getElementById(`slcs-${id}`);
const canvas = byId("image"), ctx = canvas.getContext("2d"), crop = byId("crop"), cropCtx = crop.getContext("2d");
let scene = null, camera = "", targetFrame = 0, generation = 0, pending = null, busy = false, last = null, timer = null;
let sceneView = null;
const fitButton = document.createElement("button"); fitButton.className = "icon-button"; fitButton.id = "slcs-fit"; fitButton.textContent = "全体fit"; fitButton.title = "コートと有効教師の全軌跡を表示"; fitButton.disabled = true;
document.querySelector(".controls").append(fitButton);
function fitView() {
  if (!sceneView?.model) return;
  const preset = document.querySelector("[data-preset][aria-pressed='true']")?.dataset.preset ?? "corner";
  const {yaw, pitch} = PRESETS[preset];
  const orbit = fitOrbit(framingPoints(sceneView.model), {yaw, pitch, fov: sceneView.fov, aspect: sceneView.canvas.clientWidth / sceneView.canvas.clientHeight});
  const follow = document.getElementById("follow"); if (follow.getAttribute("aria-pressed") === "true") follow.click();
  sceneView.setOrbit(orbit);
  fitButton.dataset.orbit = JSON.stringify(orbit);
}
fitButton.addEventListener("click", fitView);
for (const button of document.querySelectorAll("[data-preset], #reset")) button.addEventListener("click", () => requestAnimationFrame(fitView));
const number = (n, digits = 2) => Number.isFinite(n) ? n.toFixed(digits) : "不明";
const vector = (xyz) => xyz ? xyz.map((n) => number(n)).join(", ") : "—";

async function responseJson(response) {
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message += ` ${((await response.json()).detail)}`; } catch {}
    throw new Error(message);
  }
  return response.json();
}
function clearImage() { ctx.clearRect(0, 0, canvas.width, canvas.height); cropCtx.clearRect(0, 0, crop.width, crop.height); }
function setError(message) { byId("error").textContent = message; byId("error").hidden = !message; }
function syncText() {
  const same = last && last.generation === generation && last.sample.camera_id === camera && last.sample.frame === targetFrame;
  byId("sync").dataset.pending = String(!same);
  byId("sync").textContent = same ? `${camera} · frame ${targetFrame} · ${number(last.sample.time_seconds)}s · RGB/観測/教師一致` : `3D frame ${targetFrame} → 読込中${last ? `（画像: ${last.sample.camera_id} / frame ${last.sample.frame}）` : ""}`;
}
function requestFrame() {
  if (!scene) return;
  pending = { scene, camera, frame: targetFrame, generation };
  syncText();
  if (!busy && !timer) timer = setTimeout(() => { timer = null; pump(); }, 80);
}
async function pump() {
  if (busy || !pending) return;
  const request = pending; pending = null; busy = true;
  const query = new URLSearchParams({ form: request.scene.form, scene: request.scene.scene_id, camera: request.camera, frame: String(request.frame), revision: request.scene.revision });
  let bitmap;
  try {
    const [sample, imageResponse] = await Promise.all([
      fetch(`/api/inspection/frame?${query}`).then(responseJson),
      fetch(`/api/inspection/image?${query}`),
    ]);
    if (!imageResponse.ok && imageResponse.status !== 404) {
      await responseJson(imageResponse);
    }
    if (imageResponse.ok) bitmap = await createImageBitmap(await imageResponse.blob());
    if (request.generation !== generation || request.camera !== camera) { if (bitmap) bitmap.close(); return; }
    // Install RGB and observations together. During playback this sampled
    // frame is explicitly labelled even if the shared 3D clock runs ahead.
    if (last?.bitmap) last.bitmap.close();
    last = { sample, bitmap, generation };
    byId("image-status").hidden = Boolean(bitmap);
    byId("image-status").textContent = bitmap ? "" : "保存clipのmediaなし。2D座標を表示しています（RGB未確認）。";
    setError(""); render(); syncText();
  } catch (error) {
    if (bitmap) bitmap.close();
    if (request.generation === generation && request.camera === camera) {
      setError(`観測の取得に失敗: ${error.message}`);
      // A stale publication must not leave an apparently valid old overlay.
      if (last?.bitmap) last.bitmap.close(); last = null; clearImage();
      byId("quality").textContent = "";
      byId("sync").textContent = `frame ${targetFrame}: 取得失敗（clipを再選択してください）`;
    }
  } finally {
    busy = false;
    if (pending) timer = setTimeout(() => { timer = null; pump(); }, 80);
  }
}
function options() { return Object.fromEntries(["pose", "ball", "court", "projection"].map((name) => [name, byId(name).checked])); }
function render() {
  if (!last || !scene) return;
  const sample = last.sample;
  [canvas.width, canvas.height] = sample.image_size;
  ctx.fillStyle = "#14222b"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (last.bitmap) ctx.drawImage(last.bitmap, 0, 0);
  drawOverlay(ctx, sample, scene.inspection.skeleton, options());
  const p = sample.players[Number(byId("crop-slot").value)];
  const region = p ? poseCrop(p.pose_uv, ...sample.image_size) : null;
  cropCtx.clearRect(0, 0, crop.width, crop.height);
  if (region) cropCtx.drawImage(canvas, ...region, 0, 0, crop.width, crop.height);
  byId("crop-note").textContent = region ? `P${p.slot}: ${p.observed_joints}/17 joints。root/yaw教師の有効性とは別に2D poseを確認。` : "選択選手の枠内2D観測なし。cropは利用不可。";
  renderQuality(sample);
  const residual = ballResidualPixels(sample.ball, ...sample.image_size);
  byId("ball-residual").textContent = residual === null ? "ball観測↔教師投影差: 利用不可（未観測・無効教師・校正なし等）" : `ball観測↔教師投影差: ${number(residual, 1)} px。同じ再構成に使った観測との整合であり、独立GT誤差ではありません。`;
  const fit = sample.calibration;
  byId("calibration").textContent = fit ? `近似pinhole K/R/t · fit ${number(fit.rmse_px)} px · calibration frame ${fit.calibration_frame_index ?? "不明"} · 歪み補正なし / 独立校正GTなし` : "保存K/R/tなし。3D投影は利用不可。";
  byId("projection").disabled = !fit;
  renderProvenance(sample); renderTimeline();
}
function renderQuality(sample) {
  const rows = byId("quality"); rows.textContent = "";
  const entries = sample.players.map((p) => ({ name: `P${p.slot}`, valid: p.label_valid, weight: p.weight, reasons: p.reasons, evidence: `${p.observed_joints}/17 joints · 全cam coverage ${number(p.coverage)}`, xyz: p.position_m, yaw: p.yaw_rad }));
  const ball = sample.ball;
  const observation = { in_frame: "枠内観測", out_of_frame: "枠外観測", unobserved: "未観測（不在未確認）" }[ball.observation_state];
  entries.push({ name: "ball", valid: ball.label_valid, weight: ball.weight, reasons: ball.reasons, evidence: `${observation} · ${ball.observed_cameras}/${scene.source_camera_ids.length} cam`, xyz: ball.position_m });
  for (const e of entries) {
    const row = document.createElement("tr"); row.className = e.valid ? "good" : "bad";
    for (const text of [e.name, `${e.valid ? "有効" : "無効"} / ${number(e.weight, 3)}`, `${e.evidence}${e.reasons.length ? ` / ${e.reasons.join("・")}` : ""}`, `${vector(e.xyz)}${e.yaw != null ? ` / ${number(e.yaw * 180 / Math.PI, 1)}°` : ""}`]) {
      const td = document.createElement("td"); td.textContent = text; row.append(td);
    }
    rows.append(row);
  }
}
function renderTimeline() {
  if (!scene) return;
  const el = byId("timeline"), c = el.getContext("2d"), timeline = scene.inspection.timeline;
  c.clearRect(0, 0, el.width, el.height);
  for (const [row, values] of [...timeline.players, timeline.ball].entries()) {
    for (let f = 0; f < values.length; f++) {
      c.fillStyle = values[f] ? "#cde5d8" : "#d9731f";
      c.fillRect(f * el.width / values.length, row * 30, Math.ceil(el.width / values.length), 27);
    }
  }
  c.strokeStyle = "#25384b"; c.lineWidth = 3; c.beginPath(); c.moveTo(targetFrame * el.width / scene.frame_count, 0); c.lineTo(targetFrame * el.width / scene.frame_count, el.height); c.stroke();
  el.setAttribute("aria-valuemax", String(scene.frame_count - 1)); el.setAttribute("aria-valuenow", String(targetFrame));
}
function seek(frame) {
  const play = document.getElementById("play");
  if (play.title === "一時停止") play.click();
  const scrub = document.getElementById("scrub"); scrub.value = String(Math.max(0, Math.min(scene.frame_count - 1, frame))); scrub.dispatchEvent(new Event("input", { bubbles: true }));
}
function jumpGap(direction) {
  if (!scene) return;
  const t = scene.inspection.timeline;
  for (let f = targetFrame + direction; f >= 0 && f < scene.frame_count; f += direction) if (!t.ball[f] || t.players.some((p) => !p[f])) { seek(f); return; }
  setError(direction > 0 ? "この先に無効frameはありません。" : "これより前に無効frameはありません。");
}
function renderProvenance(sample) {
  const rows = byId("provenance"); rows.textContent = "";
  const source = scene.inspection.source_cameras.find((s) => s.camera_id === sample.camera_id), metadata = source.source;
  const entries = [
    ["dataset / clip", `${scene.dataset} / ${scene.clip_id}`], ["保存RGB", source.media],
    ["元source", metadata?.source_path ?? "未記録"], ["同期offset", metadata?.offset_sec == null ? "未記録" : `${metadata.offset_sec} s`],
    ["元source frames", metadata?.source_frame_start == null ? "未記録" : `${metadata.source_frame_start}–${metadata.source_frame_end} (source fps ${metadata.source_fps})`],
    ["letterbox", metadata?.letterbox ? JSON.stringify(metadata.letterbox) : "なし / 未記録"],
    ["教師", `疑似ラベル / ${scene.inspection.representation}`], ["3D座標系", `${scene.coordinate_frame} / metres, Z up`], ["player slot", `表示near→far / 元slot ${scene.inspection.player_source_slots.join(", ")}`],
    ["ID scope", scene.inspection.identity_scope], ["source正本", "clip.json → annotation.json → scene.jsonの現行export"],
    ["DINO / split", scene.inspection.preparation_note ?? "レビューに不要。学習準備の完了を意味しません。"],
  ];
  for (const [name, value] of entries) { const dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = name; dd.textContent = value; rows.append(dt, dd); }
}
byId("camera").addEventListener("change", () => { camera = byId("camera").value; requestFrame(); });
for (const name of ["pose", "ball", "court", "projection", "crop-slot"]) byId(name).addEventListener("change", render);
byId("prev-gap").addEventListener("click", () => jumpGap(-1)); byId("next-gap").addEventListener("click", () => jumpGap(1));
byId("timeline").addEventListener("click", (event) => { if (scene) seek(Math.floor((event.clientX - event.currentTarget.getBoundingClientRect().left) / event.currentTarget.clientWidth * scene.frame_count)); });
byId("timeline").addEventListener("keydown", (event) => { if (scene && ["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); seek(targetFrame + (event.key === "ArrowLeft" ? -1 : 1)); } });
window.addEventListener("dataset-review:scene", (event) => {
  generation++; pending = null; if (last?.bitmap) last.bitmap.close(); last = null; scene = null; sceneView = null; clearImage();
  fitButton.disabled = true;
  byId("quality").textContent = ""; byId("sync").textContent = event.detail.phase === "error" ? "clip読込失敗" : "読み込み中…"; setError("");
  for (const id of ["calibration", "crop-note", "ball-residual", "policy", "thresholds", "provenance"]) byId(id).textContent = "";
  byId("timeline").getContext("2d").clearRect(0, 0, byId("timeline").width, byId("timeline").height);
  byId("image-status").hidden = true;
  if (event.detail.phase !== "loaded") return;
  scene = event.detail.scene;
  sceneView = event.detail.view;
  fitButton.disabled = !sceneView;
  requestAnimationFrame(fitView);
  if (!scene.inspection) { setError("このsceneにはSLCS inspection情報がありません。"); return; }
  const selector = byId("camera"); selector.textContent = "";
  for (const id of scene.source_camera_ids) { const option = document.createElement("option"); option.value = id; option.textContent = id; selector.append(option); }
  camera = selector.value; targetFrame = 0;
  const q = scene.inspection.quality;
  byId("thresholds").textContent = `標準quality: player coverage ≥ ${q.min_player_confidence} / ball ≥ ${q.min_ball_cameras} cam / weight power ${q.label_weight_power} / 3D reconstruction・heading maskも必須`;
  const origin = scene.inspection.court_point_origin === "homography_fit" ? "採用homographyで整形した点" : "点の生成方法は未記録";
  byId("policy").textContent = `court入力: ${scene.inspection.court_temporal_policy}（観測frame: ${scene.inspection.court_observed_frames.join(", ") || "未記録"}）。${origin}。`;
  requestFrame();
});
window.addEventListener("dataset-review:frame", (event) => {
  if (!scene || event.detail.form !== scene.form || event.detail.sceneId !== scene.scene_id) return;
  targetFrame = event.detail.frame; renderTimeline(); requestFrame();
});
