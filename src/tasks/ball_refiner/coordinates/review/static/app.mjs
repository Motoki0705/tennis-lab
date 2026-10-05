import { canvases, ScenePanels } from "./scene.mjs";
import { draw2D, drawGraph, drawTimeline, frameFromPointer } from "./plots.mjs";

const $ = (id) => document.getElementById(id);
const SCHEMA = "ball_refiner.coordinate_review.v1";
const state = { catalog: null, result: null, frame: 0, playing: false, lastTick: 0, sequence: 0, controller: null, canvases: [], debounce: null };
const world = new ScenePanels($("view-3d"));
const visibility = () => Object.fromEntries(["gt", "input", "prediction"].map((key) => [key, $(`show-${key}`).checked]));
const checkpoint = (dimension) => state.catalog?.checkpoints.find((item) => item.id === $(`model-${dimension}d`).value);
const percent = (value) => `${(value * 100).toFixed(1)}%`;

function message(text = "", error = false) {
  $("message").textContent = text; $("message").hidden = !text; $("message").classList.toggle("error", error);
}
function busy(text = null) {
  $("infer").disabled = !!text; $("load-saved").disabled = !!text;
  $("source").classList.toggle("busy", !!text);
  if (text) $("source").textContent = text;
}
function option(select, value, label) {
  const element = document.createElement("option"); element.value = value; element.textContent = label; select.append(element);
}
function applyProfile(profile) {
  const a = profile.augmentation;
  $("event-probability").value = a.event_probability * 100;
  $("isolated-probability").value = a.isolated_probability * 100;
  $("gap-min").value = a.gap_min; $("gap-max").value = a.gap_max; $("noise-p95").value = a.noise_p95_px;
  $("augmentation-seed").value = profile.augmentation_seed; $("flow-seed").value = profile.flow_seed;
  $("augmentation-mode").value = profile.missing_enabled ? (profile.noise_enabled ? "both" : "missing") : (profile.noise_enabled ? "noise" : "none");
  state.augmentationBase = { ...a };
}
function requestFromForm() {
  for (const input of document.querySelectorAll("aside input[type=number]")) {
    if (!input.checkValidity() || input.value === "") throw new Error("拡張パラメータを入力範囲内で指定してください。");
  }
  const mode = $("augmentation-mode").value;
  const augmentation = { ...state.augmentationBase, event_probability: Number($("event-probability").value) / 100,
    isolated_probability: Number($("isolated-probability").value) / 100, gap_min: Number($("gap-min").value),
    gap_max: Number($("gap-max").value), noise_p95_px: Number($("noise-p95").value) };
  if (augmentation.gap_min >= augmentation.gap_max) throw new Error("左右幅の最大値は最小値より大きくしてください。");
  return { rally: $("rally").value, manifest_sha256: state.catalog.manifest_sha256,
    checkpoint_2d: $("model-2d").value || null, checkpoint_3d: $("model-3d").value || null,
    augmentation, augmentation_seed: Number($("augmentation-seed").value), flow_seed: Number($("flow-seed").value),
    missing_enabled: ["both", "missing"].includes(mode), noise_enabled: ["both", "noise"].includes(mode), device: $("device").value,
    checkpoint_hashes: Object.fromEntries([2, 3].filter((d) => checkpoint(d)).map((d) => [String(d), checkpoint(d).sha256])) };
}
function canLoadSaved(request) {
  const selected = [2, 3].map(checkpoint).filter(Boolean);
  const rally = state.catalog.rallies.find((item) => item.id === request.rally);
  return rally?.split === "test" && selected.length > 0 && selected.every((item) => {
    const p = item.evaluation_profile;
    return item.saved_available && p && p.augmentation_seed === request.augmentation_seed && p.flow_seed === request.flow_seed
      && p.missing_enabled === request.missing_enabled && p.noise_enabled === request.noise_enabled
      && Object.keys(p.augmentation).every((key) => p.augmentation[key] === request.augmentation[key]);
  });
}
function fillModels(preferred = {}) {
  for (const dimension of [2, 3]) {
    const select = $(`model-${dimension}d`), previous = preferred[dimension] ?? select.value;
    select.replaceChildren(); option(select, "", "モデルなし（GT・入力のみ）");
    state.catalog.checkpoints.filter((item) => item.compatible && item.dimensions === dimension
      && ($("show-last").checked || item.filename !== "last.ckpt")).sort((a, b) => Number(b.recommended) - Number(a.recommended)
        || a.validation_rmse - b.validation_rmse || a.id.localeCompare(b.id)).forEach((item) => option(select, item.id, `${item.recommended ? "★ " : ""}${item.label}`));
    if ([...select.options].some((item) => item.value === previous)) select.value = previous;
    else select.value = state.catalog.default_checkpoints[dimension] || "";
  }
  modelLabels();
}
function modelLabels() {
  for (const dimension of [2, 3]) {
    const item = checkpoint(dimension);
    $(`model-${dimension}d-info`).textContent = item ? `${item.method.toUpperCase()} · step ${item.step.toLocaleString()} · validation ${item.validation_rmse.toFixed(3)} ${item.unit}` : "モデル未選択";
  }
}
function fillRallies(preferred = null) {
  const select = $("rally"), previous = preferred || select.value, query = $("search").value.toLowerCase();
  select.replaceChildren();
  const rows = state.catalog.rallies.filter((r) => r.split === $("split").value && r.id.toLowerCase().includes(query));
  rows.forEach((row) => option(select, row.id, `${row.id} · ${row.frames} frames`));
  if (rows.some((row) => row.id === previous)) select.value = previous;
  $("rally-info").textContent = `${rows.length} ラリー · 3D真値を2D・3Dで共有`;
  return rows.length;
}
async function loadCatalog(refresh = false) {
  const response = await fetch(`/api/catalog?refresh=${refresh}`);
  if (!response.ok) throw new Error(`モデル一覧を読み込めません (${response.status})`);
  state.catalog = await response.json();
  $("dataset-summary").textContent = `single_object · ${state.catalog.rallies.length.toLocaleString()} rallies · ${state.catalog.views} cameras · ${state.catalog.fps} fps`;
  const camera = $("camera"), old = camera.value; camera.replaceChildren();
  for (let i = 0; i < state.catalog.views; i++) option(camera, String(i), `Camera ${i + 1}`);
  if (old) camera.value = old;
  const excluded = state.catalog.checkpoints.filter((entry) => !entry.compatible);
  $("excluded-summary").textContent = `非対応のcheckpoint (${excluded.length})`;
  $("excluded-list").replaceChildren();
  excluded.forEach((entry) => { const li = document.createElement("li"); li.textContent = `${entry.id}: ${entry.reason}`; $("excluded-list").append(li); });
}

function clearPredictions() {
  if (!state.result) return;
  state.result = { ...state.result, source: "preview", scene: { ...state.result.scene,
    prediction_2d: null, prediction_3d: null, metrics_2d: state.result.scene.metrics_2d.map(() => null), metrics_3d: null } };
  mountViews(); renderMetrics();
}
function invalidate() {
  state.sequence++; state.controller?.abort(); clearTimeout(state.debounce);
  clearPredictions();
  $("save-config").disabled = true; $("save-image").disabled = true;
}
async function loadReview(action = "preview", restoredView = null) {
  invalidate();
  const sequence = state.sequence;
  state.controller = new AbortController();
  let body;
  try {
    body = requestFromForm();
    if (!body.rally) throw new Error("該当するラリーがありません。");
    if (action === "auto") action = canLoadSaved(body) ? "saved" : "preview";
    if (action === "saved" && !canLoadSaved(body)) throw new Error("この条件には保存済み評価がありません。testのラリーとbest.ckptを選び、評価条件に戻すか再推論してください。");
    message();
    busy(action === "infer" ? (body.device === "cuda" ? "GPUキューで推論中…" : "CPUで推論中…") : "読み込み中…");
    const response = await fetch(`/api/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: state.controller.signal });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
    if (sequence !== state.sequence) return;
    state.result = result;
    state.frame = Math.min(restoredView?.frame ?? state.frame, result.scene.frames - 1);
    state.lastTick = 0;
    $("scene-title").textContent = `${result.scene.rally} / ${result.scene.split}`;
    $("scrub").max = result.scene.frames - 1;
    $("source").textContent = result.source === "saved" ? "保存済み評価" : result.source === "live" ? `${body.device.toUpperCase()} 推論 · ${result.seconds.toFixed(2)} s` : "GT・拡張後入力";
    if (restoredView) {
      $("camera").value = String(Math.min(restoredView.camera, state.catalog.views - 1));
      $("layout").value = restoredView.layout; $("plot-mode").value = restoredView.plot;
      for (const [kind, value] of Object.entries(restoredView.visibility)) if ($(`show-${kind}`)) $(`show-${kind}`).checked = value;
    }
    mountViews(); renderMetrics(); audit(); applyFrame(state.frame);
    $("save-config").disabled = false; $("save-image").disabled = false;
    $("provenance").textContent = `dataset ${body.manifest_sha256.slice(0, 12)} · input ${result.input_sha256.slice(0, 12)} · augmentation seed ${body.augmentation_seed} · Flow seed ${body.flow_seed} · 1本の全フレーム予測`;
  } catch (error) {
    if (sequence === state.sequence && error.name !== "AbortError") { message(error.message, true); $("source").textContent = "未適用 / 推論結果なし"; }
  } finally { if (sequence === state.sequence) busy(); }
}
function changedInput() {
  invalidate(); busy("拡張を適用中…");
  state.debounce = setTimeout(() => loadReview("preview"), 250);
}
function mountViews() {
  if (!state.result) return;
  const separate = $("layout").value === "separate";
  $("views").classList.toggle("separate", separate);
  state.canvases = canvases($("view-2d"), separate);
  world.setup(state.result.scene, separate, visibility(), $("show-cameras").checked);
  draw();
}
function draw() {
  if (!state.result) return;
  const scene = state.result.scene, camera = Number($("camera").value), flags = visibility(), mode = $("plot-mode").value;
  state.canvases.forEach(({ canvas, kind }) => draw2D(canvas, scene, camera, state.frame, flags, kind));
  world.setFrame(state.frame);
  drawTimeline($("timeline"), scene, camera, state.frame);
  drawGraph($("graph-2d"), scene, camera, state.frame, 2, mode, flags);
  drawGraph($("graph-3d"), scene, camera, state.frame, 3, mode, flags);
  $("graph-2d-title").textContent = `2D · Camera ${camera + 1} · ${mode === "speed" ? "px/s" : "px"}`;
  $("graph-3d-title").textContent = `3D · ${mode === "speed" ? "m/s" : "m"}`;
  $("camera-label").textContent = `Camera ${camera + 1}`;
}
function applyFrame(frame) {
  if (!state.result) return;
  const scene = state.result.scene;
  state.frame = Math.max(0, Math.min(scene.frames - 1, frame)); $("scrub").value = state.frame;
  $("clock").textContent = `${state.frame} / ${scene.frames - 1} · ${(state.frame / scene.fps).toFixed(2)} s`;
  const event = scene.events[state.frame];
  $("event-info").textContent = `${event & 1 ? "ショット " : ""}${event & 2 ? "バウンド " : ""}${scene.missing_3d[state.frame] ? "3D入力欠損" : ""}`;
  draw();
}
function renderMetrics() {
  if (!state.result) return;
  const camera = Number($("camera").value), scene = state.result.scene;
  $("metrics").replaceChildren();
  for (const [label, metrics, unit] of [[`2D / Cam ${camera + 1}`, scene.metrics_2d[camera], "px"], ["3D", scene.metrics_3d, "m"]]) {
    const row = document.createElement("tr");
    const cells = [label, ...["all", "missing", "observed", "event"].map((key) => metrics?.[key]?.rmse == null ? "—" : `${metrics[key].rmse.toFixed(unit === "m" ? 3 : 2)} ${unit}`), metrics ? `${metrics.velocity_rmse.toFixed(2)} ${unit}/s` : "—"];
    cells.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
    $("metrics").append(row);
  }
}
function audit() {
  const a = state.result.scene.audit;
  $("audit-events").textContent = `選択イベント ${a.selected_events} / ${a.events}`;
  $("audit-missing").textContent = `実欠損率 2D ${percent(a.frame_missing_rate_2d)} / 3D ${percent(a.frame_missing_rate_3d)}`;
  $("audit-missing").title = "2Dは全カメラ平均。イベント選択率とは異なります。";
  $("audit-noise").textContent = `実測P95 ${a.noise_p95_px === null ? "—" : a.noise_p95_px.toFixed(1) + " px"}（このラリー）`;
}
function seek(frame) { state.lastTick = 0; applyFrame(frame); }
function eventJump(direction) {
  if (!state.result) return;
  const events = state.result.scene.events.flatMap((value, index) => value ? [index] : []);
  const next = direction > 0 ? events.find((index) => index > state.frame) : events.findLast((index) => index < state.frame);
  if (next !== undefined) seek(next);
}
function download(blob, filename) {
  const url = URL.createObjectURL(blob), link = document.createElement("a"); link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportConfig() {
  if (!state.result) return;
  const data = { schema: SCHEMA, request: state.result.request, source: state.result.source, input_sha256: state.result.input_sha256,
    view: { frame: state.frame, camera: Number($("camera").value), layout: $("layout").value, plot: $("plot-mode").value, visibility: visibility() } };
  download(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), `${data.request.rally}-review.json`);
}
async function importConfig(file) {
  try {
    const data = JSON.parse(await file.text());
    if (data.schema !== SCHEMA || data.request.manifest_sha256 !== state.catalog.manifest_sha256) throw new Error("設定の形式またはdatasetのmanifestが一致しません。");
    const record = state.catalog.rallies.find((r) => r.id === data.request.rally);
    if (!record) throw new Error("保存されたラリーがありません。");
    for (const dim of [2, 3]) {
      const id = data.request[`checkpoint_${dim}d`];
      if (!id) continue;
      const item = state.catalog.checkpoints.find((entry) => entry.id === id && entry.compatible);
      if (!item || item.sha256 !== data.request.checkpoint_hashes[String(dim)]) throw new Error("保存時のcheckpointが見つからないか、内容が変更されています。");
      if (item.filename === "last.ckpt") $("show-last").checked = true;
    }
    $("split").value = record.split; $("search").value = ""; fillRallies(record.id);
    fillModels({ 2: data.request.checkpoint_2d || "", 3: data.request.checkpoint_3d || "" });
    applyProfile(data.request); $("device").value = data.request.device;
    await loadReview(data.source === "saved" ? "saved" : data.source === "live" ? "infer" : "preview", data.view);
    if (state.result && state.result.input_sha256 !== data.input_sha256) throw new Error("再現した入力hashが保存時と一致しません。");
  } catch (error) { message(error.message, true); }
}
function exportImage() {
  if (!state.result) return;
  draw(); world.render();
  const views = $("views").getBoundingClientRect(), canvas = document.createElement("canvas");
  canvas.width = Math.ceil(views.width * 2); canvas.height = Math.ceil((views.height + 425) * 2);
  const ctx = canvas.getContext("2d"); ctx.scale(2, 2); ctx.fillStyle = "#0b111b"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e2eaf5"; ctx.font = "bold 16px system-ui"; ctx.fillText(`Ball Refiner · ${state.result.scene.rally} · frame ${state.frame}`, 12, 24);
  ctx.font = "10px system-ui"; ctx.fillStyle = "#a3b7cb";
  ctx.fillText(`GT: green / Input: gray / Prediction: orange · ${$("source").textContent} · augmentation seed ${state.result.request.augmentation_seed}`, 12, 44);
  const models = Object.entries(state.result.models).map(([dim, item]) => `${dim}D ${item.method.toUpperCase()} · train events ${item.train_event_probability == null ? "unknown" : percent(item.train_event_probability)} · step ${item.step}`);
  ctx.fillText(models.join(" / "), 12, 62);
  const a = state.result.request.augmentation;
  const missing = state.result.request.missing_enabled ? `events ${percent(a.event_probability)} / isolated ${percent(a.isolated_probability)} / gaps ${a.gap_min}–${a.gap_max} frames per side` : "missing off";
  const noise = state.result.request.noise_enabled ? `noise P95 ${a.noise_p95_px} px` : "noise off";
  ctx.fillText(`Camera ${Number($("camera").value) + 1} · ${missing} · ${noise} · Flow seed ${state.result.request.flow_seed}`, 12, 80);
  for (const source of $("views").querySelectorAll("canvas")) {
    const rect = source.getBoundingClientRect(); ctx.drawImage(source, rect.left - views.left, rect.top - views.top + 95, rect.width, rect.height);
  }
  let y = views.height + 106;
  ctx.fillStyle = "#a3b7cb"; ctx.fillText(`${$("audit-events").textContent} / ${$("audit-missing").textContent} / ${$("audit-noise").textContent}`, 12, y); y += 12;
  ctx.drawImage($("timeline"), 0, y, views.width, 60); y += 76;
  ctx.fillText(`Time series: ${$("plot-mode").selectedOptions[0].textContent} · left 2D / right 3D`, 12, y); y += 7;
  ctx.drawImage($("graph-2d"), 0, y, views.width / 2, 165); ctx.drawImage($("graph-3d"), views.width / 2, y, views.width / 2, 165); y += 184;
  ctx.fillText(`dataset ${state.result.request.manifest_sha256.slice(0, 12)} / input ${state.result.input_sha256.slice(0, 12)}`, 12, y);
  canvas.toBlob((blob) => { if (blob) download(blob, `${state.result.scene.rally}-comparison.png`); });
}

$("split").addEventListener("change", () => { state.frame = 0; fillRallies(); loadReview("auto"); });
$("search").addEventListener("input", () => { fillRallies(); state.frame = 0; loadReview("auto"); });
$("rally").addEventListener("change", () => { state.frame = 0; loadReview("auto"); });
for (const [id, delta] of [["previous-rally", -1], ["next-rally", 1]]) $(id).addEventListener("click", () => {
  const select = $("rally"), next = select.selectedIndex + delta;
  if (next >= 0 && next < select.options.length) { select.selectedIndex = next; state.frame = 0; loadReview("auto"); }
});
for (const dim of [2, 3]) $(`model-${dim}d`).addEventListener("change", () => { modelLabels(); loadReview("auto"); });
$("show-last").addEventListener("change", () => { fillModels(); loadReview("auto"); });
$("refresh").addEventListener("click", async () => { try { invalidate(); await loadCatalog(true); fillModels(); await loadReview("auto"); } catch (error) { message(error.message, true); } });
for (const id of ["event-probability", "isolated-probability", "gap-min", "gap-max", "noise-p95", "augmentation-seed", "flow-seed"]) $(id).addEventListener("input", changedInput);
$("augmentation-mode").addEventListener("change", changedInput);
$("resample").addEventListener("click", () => { $("augmentation-seed").value = (Number($("augmentation-seed").value) + 1) % 4294967296; changedInput(); });
$("evaluation-preset").addEventListener("click", () => { applyProfile(checkpoint(2)?.evaluation_profile || checkpoint(3)?.evaluation_profile || state.catalog.default_profile); loadReview("auto"); });
$("infer").addEventListener("click", () => loadReview("infer")); $("load-saved").addEventListener("click", () => loadReview("saved"));
$("camera").addEventListener("change", () => { draw(); renderMetrics(); });
$("layout").addEventListener("change", mountViews); $("plot-mode").addEventListener("change", draw);
for (const kind of ["gt", "input", "prediction"]) $(`show-${kind}`).addEventListener("change", () => { world.visibility(visibility()); draw(); });
$("show-cameras").addEventListener("change", () => world.cameras($("show-cameras").checked)); $("reset-view").addEventListener("click", () => world.reset());
$("scrub").addEventListener("input", () => seek(Number($("scrub").value)));
$("previous-frame").addEventListener("click", () => seek(state.frame - 1)); $("next-frame").addEventListener("click", () => seek(state.frame + 1));
$("previous-event").addEventListener("click", () => eventJump(-1)); $("next-event").addEventListener("click", () => eventJump(1));
$("play").addEventListener("click", () => { state.playing = !state.playing; state.lastTick = 0; $("play").textContent = state.playing ? "Ⅱ" : "▶"; $("play").setAttribute("aria-label", state.playing ? "一時停止" : "再生"); });
for (const id of ["timeline", "graph-2d", "graph-3d"]) $(id).addEventListener("click", (event) => { if (state.result) seek(frameFromPointer(event, $(id), state.result.scene.frames)); });
$("save-config").addEventListener("click", exportConfig); $("save-image").addEventListener("click", exportImage);
$("load-config").addEventListener("click", () => $("config-file").click());
$("config-file").addEventListener("change", async () => { if ($("config-file").files[0]) await importConfig($("config-file").files[0]); $("config-file").value = ""; });
new ResizeObserver(() => draw()).observe($("views"));
function tick(now) {
  if (state.playing && state.result) {
    if (!state.lastTick) state.lastTick = now;
    const period = 1000 / (state.result.scene.fps * Number($("speed").value));
    const frames = Math.floor((now - state.lastTick) / period);
    if (frames > 0) { applyFrame((state.frame + frames) % state.result.scene.frames); state.lastTick += frames * period; }
  }
  world.render(); requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
try { await loadCatalog(); applyProfile(state.catalog.default_profile); fillModels(state.catalog.default_checkpoints); fillRallies(); await loadReview("auto"); }
catch (error) { message(error.message, true); busy(); }
