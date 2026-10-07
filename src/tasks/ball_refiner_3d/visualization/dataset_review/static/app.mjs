import { ScenePanels, SERIES, LABELS, COLORS, present } from "./scene.mjs";
import { drawGraph, drawTimeline, drawEvents, frameFromPointer } from "./plots.mjs";

const $ = (id) => document.getElementById(id);
const SCHEMA = "ball_refiner_3d.event_review.v2";
const state = { catalog: null, result: null, frame: 0, playing: false, lastTick: 0, sequence: 0, controller: null, debounce: null };
const world = new ScenePanels($("view-3d"));
const visibility = () => Object.fromEntries(SERIES.map((key) => [key, $(`show-${key}`).checked]));
const checkpoint = (dimension) => state.catalog?.checkpoints.find((item) => item.id === $(`model-${dimension}d`).value);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const fixed = (value, digits, unit = "") => value == null ? "—" : `${value.toFixed(digits)}${unit ? " " + unit : ""}`;
const cell = (row, value, title = "") => { const td = document.createElement("td"); td.textContent = value; if (title) td.title = title; row.append(td); return td; };
// Model-dependent scene fields; cleared whenever the shown prediction is invalidated.
const MODEL_FIELDS = ["prediction_3d", "integrated_3d", "integrated_truth_3d", "event_probability"];

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
  $("noise-jitter").value = a.jitter_sigma_px; $("noise-outlier").value = a.outlier_probability * 100;
  state.augmentationBase = { ...a };
}
function requestFromForm() {
  for (const input of document.querySelectorAll("aside input[type=number]")) {
    if (!input.checkValidity() || input.value === "") throw new Error("拡張パラメータを入力範囲内で指定してください。");
  }
  const mode = $("augmentation-mode").value;
  const augmentation = { ...state.augmentationBase, event_probability: Number($("event-probability").value) / 100,
    isolated_probability: Number($("isolated-probability").value) / 100, gap_min: Number($("gap-min").value),
    gap_max: Number($("gap-max").value), noise_p95_px: Number($("noise-p95").value),
    jitter_sigma_px: Number($("noise-jitter").value), outlier_probability: Number($("noise-outlier").value) / 100 };
  if (augmentation.gap_min >= augmentation.gap_max) throw new Error("左右幅の最大値は最小値より大きくしてください。");
  return { rally: $("rally").value, manifest_sha256: state.catalog.manifest_sha256,
    checkpoint_3d: $("model-3d").value || null,
    augmentation, augmentation_seed: Number($("augmentation-seed").value), flow_seed: Number($("flow-seed").value),
    missing_enabled: ["both", "missing"].includes(mode), noise_enabled: ["both", "noise"].includes(mode), device: $("device").value,
    checkpoint_hashes: Object.fromEntries([3].filter((d) => checkpoint(d)).map((d) => [String(d), checkpoint(d).sha256])) };
}
function canLoadSaved(request) {
  const selected = [3].map(checkpoint).filter(Boolean);
  const rally = state.catalog.rallies.find((item) => item.id === request.rally);
  return rally?.split === "test" && selected.length > 0 && selected.every((item) => {
    const p = item.evaluation_profile;
    return item.saved_available && p && p.augmentation_seed === request.augmentation_seed && p.flow_seed === request.flow_seed
      && p.missing_enabled === request.missing_enabled && p.noise_enabled === request.noise_enabled
      && Object.keys(p.augmentation).every((key) => p.augmentation[key] === request.augmentation[key]);
  });
}
function fillModels(preferred = {}) {
  for (const dimension of [3]) {
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
  for (const dimension of [3]) {
    const item = checkpoint(dimension);
    $(`model-${dimension}d-info`).textContent = item ? `${item.method.toUpperCase()}${item.physics_heads ? " + 物理head" : ""} · step ${item.step.toLocaleString()} · validation ${item.validation_rmse.toFixed(3)} ${item.unit}${item.saved_available ? "" : " · 保存済み評価なし"}` : "モデル未選択";
    $(`model-${dimension}d-info`).title = item ? (item.saved_unavailable_reason || "重みと対応を検証した保存済み評価があります") : "";
  }
  renderSummary();
}
function renderSummary() {
  if (!state.catalog) return;
  const selected = $("model-3d").value, body = $("summary");
  const rows = state.catalog.checkpoints.filter((item) => item.compatible && item.test_summary
    && ($("show-last").checked || item.filename !== "last.ckpt")).sort((a, b) => a.label.localeCompare(b.label));
  body.replaceChildren();
  for (const item of rows) {
    const s = item.test_summary, row = document.createElement("tr");
    row.classList.toggle("selected", item.id === selected);
    row.title = item.id;
    cell(row, `${item.recommended ? "★ " : ""}${item.run_name ?? item.id} · ${item.filename}`);
    cell(row, `${item.method.toUpperCase()}${item.physics_heads ? " + 物理head" : ""}`);
    cell(row, `${fixed(s.test_rmse_m, 3)} / ${fixed(s.test_missing_rmse_m, 3)} m`);
    cell(row, fixed(s.test_acceleration_rmse_mps2, 0, "m/s²"));
    cell(row, s.test_implausible_acceleration_rate == null ? "—" : percent(s.test_implausible_acceleration_rate));
    cell(row, fixed(s.test_fit_residual_rmse_m, 3, "m"));
    cell(row, fixed(s.test_event_f1, 3));
    cell(row, s.test_segmentation_failure_rate == null ? "—" : percent(s.test_segmentation_failure_rate));
    cell(row, s.test_integrated_rmse_m == null ? "—" : `${fixed(s.test_integrated_rmse_m, 3)} / ${fixed(s.test_integrated_truth_segments_rmse_m, 3)} m`);
    cell(row, s.test_surface_accuracy == null ? "—" : percent(s.test_surface_accuracy));
    cell(row, fixed(s.test_wind_error_median_mps, 2, "m/s"), "中央値");
    cell(row, fixed(s.test_k_drag_relative_error_median, 2), "中央値");
    row.addEventListener("click", () => {
      if (item.filename === "last.ckpt") $("show-last").checked = true;
      fillModels({ 3: item.id }); loadReview("auto");
    });
    body.append(row);
  }
  if (!rows.length) { const row = document.createElement("tr"); cell(row, "checkpointと対応を確認できる学習時のtest評価がありません。").colSpan = 12; body.append(row); }
  const hidden = state.catalog.checkpoints.filter((item) => item.compatible && !item.test_summary && item.filename === "best.ckpt").length;
  $("summary-note").textContent = `学習時のevaluate_checkpointが保存した値で、checkpointとdatasetのhashを照合済み。物理指標は physics_eval.v1（当てはめ残差はGT区間への力モデル当てはめ）。積分RMSEのGT区間は、区間分割が正しい場合の上限。${hidden ? `test評価を照合できないbest.ckpt: ${hidden}件。` : ""}`;
}
function fillRallies(preferred = null) {
  const select = $("rally"), previous = preferred || select.value, query = $("search").value.toLowerCase();
  select.replaceChildren();
  const rows = state.catalog.rallies.filter((r) => r.split === $("split").value && r.id.toLowerCase().includes(query));
  rows.forEach((row) => option(select, row.id, `${row.id} · ${row.frames} frames`));
  if (rows.some((row) => row.id === previous)) select.value = previous;
  $("rally-info").textContent = `${rows.length} ラリー · 共通3D真値と三角測量入力`;
  return rows.length;
}
async function loadCatalog(refresh = false) {
  const response = await fetch(`/api/catalog?refresh=${refresh}`);
  if (!response.ok) throw new Error(`モデル一覧を読み込めません (${response.status})`);
  state.catalog = await response.json();
  $("dataset-summary").textContent = `single_object · ${state.catalog.rallies.length.toLocaleString()} rallies · ${state.catalog.views} cameras · ${state.catalog.fps} fps`;
  const excluded = state.catalog.checkpoints.filter((entry) => !entry.compatible);
  $("excluded-summary").textContent = `非対応のcheckpoint (${excluded.length})`;
  $("excluded-list").replaceChildren();
  excluded.forEach((entry) => { const li = document.createElement("li"); li.textContent = `${entry.id}: ${entry.reason}`; $("excluded-list").append(li); });
  renderSummary();
}

function clearPredictions() {
  if (!state.result) return;
  const scene = { ...state.result.scene, ...Object.fromEntries(MODEL_FIELDS.map((key) => [key, null])),
    metrics: { linear: state.result.scene.metrics.linear }, segments: { ...state.result.scene.segments, predicted: null },
    physics: { ...state.result.scene.physics, predicted: null, surface_probability: null } };
  state.result = { ...state.result, source: "preview", scene };
  syncSeries(); mountViews(); renderMetrics(); renderPhysics();
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
      $("layout").value = restoredView.layout; $("plot-mode").value = restoredView.plot;
      for (const [kind, value] of Object.entries(restoredView.visibility)) if ($(`show-${kind}`)) $(`show-${kind}`).checked = value;
    }
    syncSeries(); mountViews(); renderMetrics(); renderPhysics(); audit(); applyFrame(state.frame);
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
function syncSeries() {
  const available = new Set(present(state.result.scene));
  for (const kind of SERIES) {
    const box = $(`show-${kind}`);
    box.disabled = !available.has(kind);
    box.parentElement.title = box.disabled ? `${LABELS[kind]}: このモデル・結果にはありません` : box.parentElement.dataset.title ?? box.parentElement.title;
  }
}
function mountViews() {
  if (!state.result) return;
  const separate = $("layout").value === "separate";
  $("views").classList.toggle("separate", separate);
  world.setup(state.result.scene, separate, visibility(), $("show-cameras").checked);
  draw();
}
function draw() {
  if (!state.result) return;
  const scene = state.result.scene, flags = visibility(), mode = $("plot-mode").value;
  world.setFrame(state.frame);
  drawTimeline($("timeline"), scene, state.frame);
  drawEvents($("event-graph"), scene, state.frame);
  $("event-target-info").textContent = `GT: Gaussian σ=${scene.event_sigma_frames} frame · 推論: softmax`;
  const { clipped } = drawGraph($("graph-3d"), scene, state.frame, mode, flags);
  const unit = { speed: "m/s", acceleration: "m/s²" }[mode] ?? "m";
  $("graph-3d-title").textContent = `3D · ${unit}${["speed", "acceleration"].includes(mode) ? " · GT飛行区間内の差分" : ""}${clipped ? " · 縦軸はP99で打ち切り" : ""}`;
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
  const scene = state.result.scene;
  $("metrics").replaceChildren();
  for (const kind of SERIES.filter((key) => scene.metrics[key])) {
    const metrics = scene.metrics[kind], kinematics = metrics.kinematics.all, row = document.createElement("tr");
    const label = cell(row, LABELS[kind]); label.style.color = COLORS[kind];
    for (const key of ["all", "missing", "observed", "event"]) cell(row, fixed(metrics[key]?.rmse, 3, "m"));
    cell(row, fixed(kinematics.velocity_rmse, 2, "m/s"));
    cell(row, fixed(kinematics.acceleration_rmse, 1, "m/s²"), `欠損frame: ${fixed(metrics.kinematics.missing.acceleration_rmse, 1, "m/s²")}`);
    cell(row, kinematics.implausible_acceleration_rate == null ? "—" : percent(kinematics.implausible_acceleration_rate));
    cell(row, fixed(kinematics.jerk_ratio, 2));
    $("metrics").append(row);
  }
}
function renderPhysics() {
  if (!state.result) return;
  const physics = state.result.scene.physics, body = $("physics");
  body.replaceChildren();
  const names = { wind_x_mps: ["風 x", 2, "m/s"], wind_y_mps: ["風 y", 2, "m/s"], k_drag: ["k_drag", 4, ""], k_magnus: ["k_magnus", 5, ""] };
  physics.columns.forEach((column, i) => {
    const [label, digits, unit] = names[column], truth = physics.truth[i], guess = physics.predicted?.[i];
    const row = document.createElement("tr");
    cell(row, label); cell(row, fixed(truth, digits, unit)); cell(row, fixed(guess, digits, unit));
    cell(row, guess == null ? "—" : unit ? fixed(guess - truth, digits, unit) : `${((guess / truth - 1) * 100).toFixed(1)}%`);
    body.append(row);
  });
  const [wx, wy] = physics.truth, [px, py] = physics.predicted ?? [];
  const wind = document.createElement("tr");
  cell(wind, "風のベクトル誤差"); cell(wind, fixed(Math.hypot(wx, wy), 2, "m/s"), "GTの風速"); cell(wind, px == null ? "—" : fixed(Math.hypot(px, py), 2, "m/s"), "予測の風速");
  cell(wind, px == null ? "—" : fixed(Math.hypot(px - wx, py - wy), 2, "m/s"));
  body.append(wind);
  const surface = document.createElement("tr");
  cell(surface, "surface"); cell(surface, physics.surface);
  const probability = physics.surface_probability;
  const best = probability ? physics.surfaces[probability.indexOf(Math.max(...probability))] : null;
  cell(surface, probability ? physics.surfaces.map((name, i) => `${name} ${percent(probability[i])}`).join(" · ") : "—");
  cell(surface, best == null ? "—" : best === physics.surface ? "正解" : `誤り（${best}）`);
  body.append(surface);
  const segments = state.result.scene.segments, count = (labels) => labels ? labels[labels.length - 1] + 1 : null;
  const row = document.createElement("tr");
  cell(row, "飛行区間の数"); cell(row, String(count(segments.truth))); cell(row, segments.predicted ? String(count(segments.predicted)) : "—");
  cell(row, segments.predicted ? `${count(segments.predicted) - count(segments.truth) >= 0 ? "+" : ""}${count(segments.predicted) - count(segments.truth)}` : "—");
  body.append(row);
  const model = Object.values(state.result.models)[0];
  $("physics-note").textContent = physics.predicted ? "予測は場head（ラリーで1回）の出力。k_drag・k_magnusの誤差は相対値" : model ? "このモデルには物理headがありません（GTのみ表示）" : "GTのみ表示";
}
function audit() {
  const a = state.result.scene.audit;
  $("audit-events").textContent = `選択イベント ${a.selected_events} / ${a.events}`;
  $("audit-missing").textContent = `3D入力の欠損率 ${percent(a.frame_missing_rate_3d)}`;
  $("audit-missing").title = "三角測量で有効な3D座標が得られないフレームの割合です。";
  $("audit-noise").textContent = `三角測量前のノイズP95 ${a.noise_p95_px === null ? "—" : a.noise_p95_px.toFixed(1) + " px"}（このラリー）`;
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
    view: { frame: state.frame, layout: $("layout").value, plot: $("plot-mode").value, visibility: visibility() } };
  download(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), `${data.request.rally}-review.json`);
}
async function importConfig(file) {
  try {
    const data = JSON.parse(await file.text());
    if (data.schema !== SCHEMA || data.request.manifest_sha256 !== state.catalog.manifest_sha256) throw new Error("設定の形式またはdatasetのmanifestが一致しません。");
    const record = state.catalog.rallies.find((r) => r.id === data.request.rally);
    if (!record) throw new Error("保存されたラリーがありません。");
    for (const dim of [3]) {
      const id = data.request[`checkpoint_${dim}d`];
      if (!id) continue;
      const item = state.catalog.checkpoints.find((entry) => entry.id === id && entry.compatible);
      if (!item || item.sha256 !== data.request.checkpoint_hashes[String(dim)]) throw new Error("保存時のcheckpointが見つからないか、内容が変更されています。");
      if (item.filename === "last.ckpt") $("show-last").checked = true;
    }
    $("split").value = record.split; $("search").value = ""; fillRallies(record.id);
    fillModels({ 3: data.request.checkpoint_3d || "" });
    applyProfile(data.request); $("device").value = data.request.device;
    await loadReview(data.source === "saved" ? "saved" : data.source === "live" ? "infer" : "preview", data.view);
    if (state.result && state.result.input_sha256 !== data.input_sha256) throw new Error("再現した入力hashが保存時と一致しません。");
  } catch (error) { message(error.message, true); }
}
function exportImage() {
  if (!state.result) return;
  draw(); world.render();
  const views = $("views").getBoundingClientRect(), canvas = document.createElement("canvas");
  canvas.width = Math.ceil(views.width * 2); canvas.height = Math.ceil((views.height + 625) * 2);
  const ctx = canvas.getContext("2d"); ctx.scale(2, 2); ctx.fillStyle = "#0b111b"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e2eaf5"; ctx.font = "bold 16px system-ui"; ctx.fillText(`Ball Refiner · ${state.result.scene.rally} · frame ${state.frame}`, 12, 24);
  ctx.font = "10px system-ui"; ctx.fillStyle = "#a3b7cb";
  ctx.fillText(`GT: green / Input: gray / Prediction: orange / Integrated (predicted segments): blue / Integrated (GT segments): pink / Linear: brown · ${$("source").textContent} · augmentation seed ${state.result.request.augmentation_seed}`, 12, 44);
  const models = Object.entries(state.result.models).map(([dim, item]) => `${item.run_name ?? dim + "D"} ${item.method.toUpperCase()}${item.physics_heads ? " + physics heads" : ""} · train events ${item.train_event_probability == null ? "unknown" : percent(item.train_event_probability)} · step ${item.step}`);
  ctx.fillText(models.join(" / "), 12, 62);
  const a = state.result.request.augmentation;
  const missing = state.result.request.missing_enabled ? `events ${percent(a.event_probability)} / isolated ${percent(a.isolated_probability)} / gaps ${a.gap_min}–${a.gap_max} frames per side` : "missing off";
  const noise = state.result.request.noise_enabled ? `noise P95 ${a.noise_p95_px} px` : "noise off";
  ctx.fillText(`${missing} · ${noise} · Flow seed ${state.result.request.flow_seed}`, 12, 80);
  for (const source of $("views").querySelectorAll("canvas")) {
    const rect = source.getBoundingClientRect(); ctx.drawImage(source, rect.left - views.left, rect.top - views.top + 95, rect.width, rect.height);
  }
  let y = views.height + 106;
  ctx.fillStyle = "#a3b7cb"; ctx.fillText(`${$("audit-events").textContent} / ${$("audit-missing").textContent} / ${$("audit-noise").textContent}`, 12, y); y += 12;
  ctx.drawImage($("timeline"), 0, y, views.width, 78); y += 94;
  ctx.fillText(`Time series: ${$("plot-mode").selectedOptions[0].textContent} · 3D`, 12, y); y += 7;
  ctx.drawImage($("graph-3d"), 0, y, views.width, 165); y += 184;
  ctx.fillText(`Event probability: GT Gaussian sigma=${state.result.scene.event_sigma_frames} / softmax prediction`, 12, y); y += 8;
  ctx.drawImage($("event-graph"), 0, y, views.width, 170); y += 190;
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
for (const dim of [3]) $(`model-${dim}d`).addEventListener("change", () => { modelLabels(); loadReview("auto"); });
$("show-last").addEventListener("change", () => { fillModels(); loadReview("auto"); });
for (const kind of SERIES) $(`show-${kind}`).parentElement.dataset.title = $(`show-${kind}`).parentElement.title;
$("refresh").addEventListener("click", async () => { try { invalidate(); await loadCatalog(true); fillModels(); await loadReview("auto"); } catch (error) { message(error.message, true); } });
for (const id of ["event-probability", "isolated-probability", "gap-min", "gap-max", "noise-p95", "noise-jitter", "noise-outlier", "augmentation-seed", "flow-seed"]) $(id).addEventListener("input", changedInput);
$("augmentation-mode").addEventListener("change", changedInput);
$("resample").addEventListener("click", () => { $("augmentation-seed").value = (Number($("augmentation-seed").value) + 1) % 4294967296; changedInput(); });
$("evaluation-preset").addEventListener("click", () => { applyProfile(checkpoint(3)?.evaluation_profile || state.catalog.default_profile); loadReview("auto"); });
$("infer").addEventListener("click", () => loadReview("infer")); $("load-saved").addEventListener("click", () => loadReview("saved"));
$("layout").addEventListener("change", mountViews); $("plot-mode").addEventListener("change", draw);
for (const kind of SERIES) $(`show-${kind}`).addEventListener("change", () => {
  if ($("layout").value === "separate") mountViews(); else { world.visibility(visibility()); draw(); }
});
$("show-cameras").addEventListener("change", () => world.cameras($("show-cameras").checked)); $("reset-view").addEventListener("click", () => world.reset());
$("scrub").addEventListener("input", () => seek(Number($("scrub").value)));
$("previous-frame").addEventListener("click", () => seek(state.frame - 1)); $("next-frame").addEventListener("click", () => seek(state.frame + 1));
$("previous-event").addEventListener("click", () => eventJump(-1)); $("next-event").addEventListener("click", () => eventJump(1));
$("play").addEventListener("click", () => { state.playing = !state.playing; state.lastTick = 0; $("play").textContent = state.playing ? "Ⅱ" : "▶"; $("play").setAttribute("aria-label", state.playing ? "一時停止" : "再生"); });
for (const id of ["timeline", "graph-3d", "event-graph"]) $(id).addEventListener("click", (event) => { if (state.result) seek(frameFromPointer(event, $(id), state.result.scene.frames)); });
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
