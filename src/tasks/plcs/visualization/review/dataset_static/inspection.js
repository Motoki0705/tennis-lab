// PLCS task-owned inspection panel; the shared scene player owns the timeline.
import { decodeObservations, frameObservation, frameStatistics, inImage, playerCrop } from "./observation.mjs";

const panel = document.createElement("aside");
panel.className = "inspection-panel";
panel.setAttribute("aria-label", "2D入力と3D教師の検品");
panel.innerHTML = `
  <h2>2D入力 ↔ 3D教師</h2>
  <p class="synthetic-note">ACCAD由来の合成投影。RGB映像・実測の3D教師は含みません。</p>
  <p id="inspection-status" role="status">シーンの読み込み待ち…</p>
  <div id="inspection-body" class="inspection-body" hidden>
    <section class="teacher">
      <h3>3D教師 <span id="inspection-split" class="split-badge"></span></h3>
      <p id="inspection-frame" class="inspection-frame"></p>
      <dl><div><dt>root位置 [m]</dt><dd id="teacher-root"></dd></div>
      <div><dt>yaw [°]</dt><dd id="teacher-yaw"></dd></div></dl>
      <p class="muted">世界座標のCOCO17を中央に表示。rootは保存教師をメートルに復元。</p>
    </section>
    <section>
      <h3>カメラごとの保存観測</h3>
      <div id="inspection-cameras" class="camera-grid" role="group" aria-label="観測カメラ"></div>
      <p id="observation-size" class="muted"></p>
      <canvas id="observation-view" aria-label="保存2D観測と3D教師の再投影"></canvas>
      <div class="observation-legend">
        <span><i class="marker"></i>選手の保存2D</span><span><i class="marker court"></i>コートの保存2D</span>
        <span><i class="marker projection"></i>3D教師の再投影</span><span><i class="marker hidden-point"></i>vis=0</span>
      </div>
      <div class="zoom-row"><canvas id="observation-zoom" aria-label="選手観測の拡大"></canvas>
        <dl><div><dt>選手 vis=1</dt><dd id="human-visible"></dd></div>
        <div><dt>コート vis=1</dt><dd id="court-visible"></dd></div>
        <div><dt>再投影 最大差</dt><dd id="projection-error"></dd></div>
        <div><dt>可視性の不一致</dt><dd id="visibility-mismatch"></dd></div></dl>
      </div>
      <p id="frame-check" class="check-status" role="status"></p>
      <label><input id="all-court-points" type="checkbox" />保存コート20点を表示（既定学習は先頭14点）</label>
      <p class="muted">拡大はvis=1の選手関節で切り出し。欠落点を補間せず、保存時の観測を表示します。</p>
    </section>
    <section>
      <h3>データの由来・保存内容</h3>
      <p id="motion-source" class="source-path"></p>
      <p id="motion-category" class="summary-line"></p>
      <p id="canonical-description" class="summary-line"></p>
      <p id="dataset-splits" class="summary-line"></p>
      <p id="source-sharing" class="warning" hidden></p>
      <p id="split-warning" class="warning" hidden></p>
      <p id="camera-summary" class="muted"></p>
      <p class="muted">再投影差は保存2Dと世界COCO17/コートの対応を検査する値です。visは前方かつ画像内の判定で、遮蔽の教師ではありません。学習のカメラ選択・crop・augmentation前。</p>
    </section>
  </div>`;
document.querySelector(".workspace").append(panel);
document.getElementById("brand-sub").textContent = "ACCAD合成投影 / physical court";
document.querySelector("#hud-pos").parentElement.querySelector("dt").textContent = "hip centre";

const element = id => document.getElementById(id);
const state = { token: 0, scene: null, inspection: null, fields: null, frame: 0, camera: 0 };
const canvases = [element("observation-view"), element("observation-zoom")];

async function response(url, type) {
  const result = await fetch(url);
  if (!result.ok) {
    const body = await result.json().catch(() => ({}));
    throw new Error(`${result.status} ${body.detail || result.statusText}`);
  }
  return type === "binary" ? result.arrayBuffer() : result.json();
}

function showStatus(message, error = false) {
  element("inspection-status").hidden = false;
  element("inspection-status").className = error ? "error" : "muted";
  element("inspection-status").textContent = message;
  element("inspection-body").hidden = true;
}

window.addEventListener("dataset-review:scene", async ({ detail }) => {
  const token = ++state.token;
  state.inspection = null;
  state.fields = null;
  if (detail.phase !== "loaded") {
    showStatus(detail.phase === "error" ? `3Dシーンを読み込めません: ${detail.error}` : "2D入力と教師を読み込み中…", detail.phase === "error");
    return;
  }
  state.scene = detail.scene;
  state.frame = 0;
  state.camera = 0;
  showStatus("保存2D観測を読み込み中…");
  const query = new URLSearchParams({ form: detail.scene.form, scene: detail.scene.scene_id, revision: detail.scene.revision });
  try {
    const [inspection, buffer] = await Promise.all([
      response(`/api/scene/inspection?${query}`, "json"),
      response(`/api/scene/observations?${query}`, "binary"),
    ]);
    if (token !== state.token) return;
    state.inspection = inspection;
    state.fields = decodeObservations(buffer, inspection);
    renderMetadata();
    element("inspection-status").hidden = true;
    element("inspection-body").hidden = false;
    renderFrame();
  } catch (error) {
    if (token !== state.token) return;
    showStatus(`2D入力の検品は利用できません: ${error.message}`, true);
  }
});
window.addEventListener("dataset-review:frame", ({ detail }) => {
  if (!state.scene || detail.sceneId !== state.scene.scene_id || detail.form !== state.scene.form) return;
  state.frame = detail.frame;
  renderFrame();
});

function renderMetadata() {
  const inspection = state.inspection;
  element("inspection-split").textContent = inspection.splits.join(" / ") || "split未割当";
  element("motion-source").textContent = inspection.source.motion || "元モーション未記録";
  element("motion-category").textContent = `動作: ${inspection.source.category || "未記録"} / gender: ${inspection.source.gender || "未記録"}`;
  element("canonical-description").textContent = inspection.canonical_shape
    ? `保存canonical pose: ${inspection.canonical_shape[1]}関節。学習のpose教師はworld COCO17から構成。`
    : "canonical_pose_3d.npyは未保存。world COCO17・root/yawを表示。";
  const summary = inspection.split_summary;
  element("dataset-splits").textContent = `保存split: ${Object.entries(summary.counts).map(([key, value]) => `${key} ${value ?? "未保存"}`).join(" / ")}`;
  element("source-sharing").hidden = inspection.source_splits.length < 2;
  element("source-sharing").textContent = `同じ元モーションが ${inspection.source_splits.join(" / ")} にあります。元モーションはsplit間で独立していません。`;
  const splitIssues = [
    summary.missing_files.length && `splitファイル不足: ${summary.missing_files.join(", ")}`,
    summary.unknown_scenes.length && `未存在scene ${summary.unknown_scenes.length}件`,
    summary.duplicate_entries.length && `split内重複 ${summary.duplicate_entries.length}件`,
    summary.unassigned_scenes.length && `未割当 ${summary.unassigned_scenes.length}件`,
    summary.overlapping_scenes.length && `複数splitのscene ${summary.overlapping_scenes.length}件`,
  ].filter(Boolean);
  element("split-warning").hidden = !splitIssues.length;
  element("split-warning").textContent = splitIssues.join(" / ");
  const buttons = element("inspection-cameras");
  buttons.textContent = "";
  for (const camera of inspection.cameras) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.cameraIndex = camera.index;
    const label = document.createElement("span");
    label.textContent = camera.id;
    const count = document.createElement("small");
    button.append(label, count);
    button.addEventListener("click", () => { state.camera = camera.index; renderFrame(); });
    buttons.append(button);
  }
}

function pxError(value) { return value === null ? "比較点なし" : `${value.toFixed(4)} px`; }

function renderFrame() {
  if (!state.inspection || !state.fields) return;
  const inspection = state.inspection, frame = state.frame;
  const camera = inspection.cameras.find(camera => camera.index === state.camera);
  const human = frameObservation(state.fields, camera.index, frame, "human");
  const court = frameObservation(state.fields, camera.index, frame, "court");
  const humanStats = frameStatistics(human, camera.image_size);
  const courtStats = frameStatistics(court, camera.image_size, element("all-court-points").checked ? 20 : 14);
  element("inspection-frame").textContent = `frame ${frame} / ${inspection.frame_count - 1} · ${(frame / state.scene.fps).toFixed(2)}s`;
  const root = state.fields.root_m.subarray(frame * 3, frame * 3 + 3);
  element("teacher-root").textContent = `${Array.from(root).map(value => value.toFixed(2)).join(", ")}`;
  const heading = state.fields.heading.subarray(frame * 2, frame * 2 + 2);
  element("teacher-yaw").textContent = `${(Math.atan2(heading[1], heading[0]) * 180 / Math.PI).toFixed(1)}`;
  for (const button of element("inspection-cameras").children) {
    const index = Number(button.dataset.cameraIndex);
    const counts = frameStatistics(frameObservation(state.fields, index, frame, "human"), camera.image_size);
    button.setAttribute("aria-pressed", String(index === camera.index));
    button.dataset.empty = String(counts.visible === 0);
    button.querySelector("small").textContent = `選手 ${counts.visible}/17`;
  }
  element("observation-size").textContent = `${camera.id} · ${camera.image_size.join(" × ")} px · UVの左上(0,0) / 右下(1,1)`;
  element("human-visible").textContent = `${humanStats.visible} / 17`;
  element("court-visible").textContent = `${courtStats.visible} / ${element("all-court-points").checked ? 20 : 14}`;
  const errors = [humanStats.maxError, courtStats.maxError].filter(value => value !== null);
  const maxError = errors.length ? Math.max(...errors) : null;
  element("projection-error").textContent = pxError(maxError);
  const mismatch = humanStats.mismatch + courtStats.mismatch;
  element("visibility-mismatch").textContent = `${mismatch}点`;
  const outside = humanStats.outside + courtStats.outside;
  const warning = mismatch > 0 || outside > 0 || (maxError !== null && maxError > 0.5);
  element("frame-check").dataset.warning = String(warning || humanStats.visible === 0);
  const issues = [mismatch && `可視性不一致 ${mismatch}点`, outside && `vis=1の画面外 ${outside}点`,
    maxError !== null && maxError > 0.5 && `再投影差 ${pxError(maxError)}（0.5px超）`].filter(Boolean);
  element("frame-check").textContent = warning ? `要確認: ${issues.join(" · ")}`
    : humanStats.visible === 0 ? "このカメラは選手の有効観測なし。他カメラ・フレームを確認できます。"
    : "このフレームの保存観測と3D教師の再投影は整合しています。";
  element("camera-summary").textContent = `シーン全体（${camera.id}）: 選手の観測なし ${camera.human.empty_frames}フレーム / 可視性不一致 ${camera.human.visibility_mismatch_count + camera.court.visibility_mismatch_count}点 / 再投影 最大差 ${pxError(camera.human.max_error_px)}`;
  draw(canvases[0], human, court, camera.image_size, null);
  draw(canvases[1], human, court, camera.image_size, playerCrop(human, camera.image_size));
}

function draw(canvas, human, court, imageSize, crop) {
  const width = canvas.clientWidth;
  if (!width) return;
  const height = canvas.clientHeight, ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.fillStyle = "#eef2f6";
  context.fillRect(0, 0, width, height);
  const zoom = canvas === canvases[1];
  if (zoom && crop === null) {
    context.fillStyle = "#7b8998"; context.font = "11px sans-serif"; context.textAlign = "center";
    context.fillText("有効観測なし", width / 2, height / 2);
    return;
  }
  const bounds = crop || [0, 0, ...imageSize];
  const toPixel = (uv, index) => [
    (uv[index * 2] * imageSize[0] - bounds[0]) / bounds[2] * width,
    (uv[index * 2 + 1] * imageSize[1] - bounds[1]) / bounds[3] * height,
  ];
  context.save();
  context.beginPath(); context.rect(0, 0, width, height); context.clip();
  function skeleton(observation, edges, color) {
    context.strokeStyle = color; context.lineWidth = zoom ? 1.8 : 1;
    for (const [a, b] of edges) {
      if (!observation.visible[a] || !observation.visible[b]) continue;
      const pa = toPixel(observation.uv, a), pb = toPixel(observation.uv, b);
      context.beginPath(); context.moveTo(...pa); context.lineTo(...pb); context.stroke();
    }
  }
  if (!zoom) skeleton(court, state.inspection.court_edges, "#a3b5c4");
  skeleton(human, state.inspection.skeleton, "#167e70");
  function points(observation, color, limit) {
    for (let index = 0; index < limit; index += 1) {
      const visible = observation.visible[index] === 1;
      const saved = toPixel(observation.uv, index), projected = toPixel(observation.projected, index);
      context.setLineDash(visible ? [] : [2, 2]);
      context.fillStyle = color; context.strokeStyle = visible ? color : "#7b8998";
      // Some writers mask unavailable UV to zero. Do not plot that as a real observation.
      const masked = !visible && observation.uv[index * 2] === 0 && observation.uv[index * 2 + 1] === 0;
      if (!masked) {
        context.beginPath(); context.arc(...saved, zoom ? 3.5 : 2.5, 0, Math.PI * 2);
        if (visible) context.fill(); else context.stroke();
      }
      context.setLineDash([]);
      if (observation.front[index] && inImage(observation.projected[index * 2], observation.projected[index * 2 + 1])) {
        context.strokeStyle = visible ? "#c8661b" : "#98a6b4";
        context.lineWidth = visible ? 1.2 : 1;
        context.beginPath(); context.arc(...projected, zoom ? 5.5 : 4, 0, Math.PI * 2); context.stroke();
      }
      if (zoom && visible) {
        context.font = "9px ui-monospace, monospace"; context.fillStyle = "#475668";
        context.fillText(String(index), saved[0] + 5, saved[1] - 4);
      }
    }
  }
  if (!zoom) points(court, "#377bc2", element("all-court-points").checked ? 20 : 14);
  points(human, "#167e70", 17);
  context.restore();
}

element("all-court-points").addEventListener("change", renderFrame);
new ResizeObserver(renderFrame).observe(panel);
