const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const targetNames = { ball: "ボール", player: "選手" };
const labels = {
  missing: "JSONなし",
  unreviewed: "未確認",
  in_progress: "確認途中",
  pending: "未確認あり",
  reviewed_partial: "確認済・未解決",
  completed: "completed",
  invalid: "検証エラー",
  selection_required: "版の選択が必要",
};
const state = {
  catalog: null,
  filter: "work",
  page: 0,
  pageSize: 40,
  selected: new Set(),
  clip: null,
  detail: null,
  annotations: {},
  versions: {},
  epoch: 0,
  detailAbort: null,
  previewAbort: null,
  blobUrl: null,
  handoff: null,
};
let toastTimer;
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 3500);
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = `notice${error ? " error" : ""}`;
  $("notice").hidden = !message;
}
async function api(path, options = {}) {
  const r = await fetch(path, options);
  if (!r.ok) {
    let d;
    try {
      d = await r.json();
    } catch {
      d = { detail: `HTTP ${r.status}` };
    }
    throw new Error(
      typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail),
    );
  }
  return r;
}
const jsonPost = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const badge = (kind, text = labels[kind]) =>
  `<span class="badge ${esc(kind)}">${esc(text)}</span>`;
const pending = (row) =>
  row.targets.ball.unreviewed + row.targets.player.unreviewed;
const uncertain = (row) =>
  row.targets.ball.uncertain_frames + row.targets.player.uncertain_frames;
function matches(row, filter) {
  return (
    filter === "all" ||
    (filter === "work" &&
      ["missing", "pending", "invalid", "selection_required"].includes(
        row.state,
      )) ||
    (filter === "reviewed" &&
      ["completed", "reviewed_partial"].includes(row.state)) ||
    row.state === filter
  );
}
function filtered() {
  let rows = state.catalog?.clips || [];
  const q = $("search").value.toLowerCase().trim(),
    src = $("source").value;
  rows = rows.filter(
    (r) =>
      matches(r, state.filter) &&
      (!src || r.source_id === src) &&
      (!q || `${r.id} ${r.title} ${r.source_id}`.toLowerCase().includes(q)),
  );
  const sort = $("sort").value;
  return rows.toSorted((a, b) =>
    sort === "pending"
      ? pending(b) - pending(a) || a.id.localeCompare(b.id)
      : sort === "uncertain"
        ? uncertain(b) - uncertain(a) || a.id.localeCompare(b.id)
        : a.id.localeCompare(b.id),
  );
}
function summary() {
  const c = state.catalog.summary,
    counts = c.states;
  const reviewed = (counts.reviewed_partial || 0) + (counts.completed || 0);
  $("summary").innerHTML = [
    ["missing", "未着手", counts.missing || 0, "両対象とも表示対象のJSONなし"],
    [
      "pending",
      "未確認あり",
      counts.pending || 0,
      "片方未着・未確認フレームを含む",
    ],
    ["reviewed", "全フレーム確認済み", reviewed, "不確実性を残すpartialも含む"],
    [
      "completed",
      "両対象 completed",
      counts.completed || 0,
      `全 ${c.clips} クリップ / JSON ${c.accepted_annotations} 件採用済み`,
    ],
  ]
    .map(
      ([kind, label, value, note]) =>
        `<div class="metric ${kind}"><div class="metric-label">${label}</div><div class="metric-value">${Number(value).toLocaleString()} <small style="font-size:12px">clips</small></div><div class="metric-note">${esc(note)}</div></div>`,
    )
    .join("");
  const status = state.catalog.campaign.status;
  $("campaign-state").textContent =
    status === "stopped_by_user"
      ? "キャンペーン停止済み"
      : status === "running"
        ? "キャンペーン稼働中"
        : status === "draining_current_generation"
          ? "最終世代を終了処理中"
          : "読み取り専用";
  const errors = state.catalog.diagnostics;
  $("diagnostics-box").hidden = !errors.length;
  $("diagnostics-title").textContent = `読み込み時の問題 ${errors.length} 件`;
  $("diagnostics").textContent = errors.join("\n");
  const sources = [
    ...new Map(
      state.catalog.clips.map((r) => [r.source_id, r.title]),
    ).entries(),
  ].toSorted((a, b) => a[0].localeCompare(b[0]));
  const prior = $("source").value;
  $("source").innerHTML =
    '<option value="">すべてのソース</option>' +
    sources
      .map(
        ([id, title]) =>
          `<option value="${esc(id)}">${esc(id)} — ${esc(title).slice(0, 55)}</option>`,
      )
      .join("");
  $("source").value = prior;
}
function renderList() {
  if (!state.catalog) return;
  const defs = [
    ["work", "委託・確認が必要"],
    ["missing", "未着手"],
    ["pending", "未確認あり"],
    ["reviewed", "全確認済み"],
    ["all", "すべて"],
  ];
  $("filters").innerHTML = defs
    .map(
      ([key, label]) =>
        `<button data-filter="${key}" class="${state.filter === key ? "active" : ""}">${label} <span class="filter-count">${state.catalog.clips.filter((r) => matches(r, key)).length}</span></button>`,
    )
    .join("");
  const rows = filtered();
  state.page = Math.max(
    0,
    Math.min(state.page, Math.ceil(rows.length / state.pageSize) - 1),
  );
  const pageRows = rows.slice(
    state.page * state.pageSize,
    (state.page + 1) * state.pageSize,
  );
  const cell = (t) =>
    `<div class="target-cell">${badge(t.state)}${t.origin === "draft" ? badge("draft", "下書き") : ""}<div class="progress"><span style="width:${t.reviewed_percent}%"></span></div><span class="progress-text">${t.reviewed.toLocaleString()} / ${t.frames.toLocaleString()} 確認済み</span>${t.uncertain_frames ? `<div class="progress-text">未解決 ${t.uncertain_frames}</div>` : ""}</div>`;
  $("clips").innerHTML = pageRows
    .map(
      (r) =>
        `<tr class="${state.clip?.id === r.id ? "selected-row" : ""}"><td><input type="checkbox" data-select="${esc(r.id)}" aria-label="${esc(r.id)}を委託候補に選択" ${state.selected.has(r.id) ? "checked" : ""} ${r.active || r.state === "invalid" ? "disabled" : ""}></td><td><button class="clip-button" data-open="${esc(r.id)}">${esc(r.title || r.source_id)}</button><span class="clip-subtitle">${esc(r.id)}</span><div class="clip-flags">${badge(r.state)}${r.active ? badge("active", "担当稼働中") : ""}${r.warnings.length ? badge("warning", `提出物の注意 ${r.warnings.length}`) : ""}</div><span class="progress-text">${r.frames} frames · ${r.duration_seconds.toFixed(2)}s · ${r.width}×${r.height}</span></td><td>${cell(r.targets.ball)}</td><td>${cell(r.targets.player)}</td><td><div class="pending-number">${pending(r).toLocaleString()}</div><span class="progress-text">対象フレーム</span></td></tr>`,
    )
    .join("");
  $("empty").hidden = rows.length > 0;
  $("list-count").textContent =
    `${rows.length.toLocaleString()} 件 / 全 ${state.catalog.clips.length.toLocaleString()} 件`;
  $("page-label").textContent =
    `${state.page + 1} / ${Math.max(1, Math.ceil(rows.length / state.pageSize))}`;
  $("prev-page").disabled = state.page === 0;
  $("next-page").disabled = (state.page + 1) * state.pageSize >= rows.length;
  selectionCount();
}
function selectionCount() {
  $("selection-count").textContent = `${state.selected.size}件選択`;
  $("handoff-batch").disabled = !state.selected.size;
}
function releasePreview() {
  state.previewAbort?.abort();
  state.previewAbort = null;
  const video = $("video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  if (state.blobUrl) {
    URL.revokeObjectURL(state.blobUrl);
    state.blobUrl = null;
  }
  $("cancel-preview").hidden = true;
  $("render-preview").disabled = false;
}
function closeDetail() {
  state.epoch++;
  state.detailAbort?.abort();
  releasePreview();
  state.clip = null;
  state.detail = null;
  state.annotations = {};
  $("detail").hidden = true;
  $("workspace").classList.remove("has-detail");
  renderList();
}
async function loadCatalog(refresh = false) {
  $("refresh").disabled = true;
  notice("");
  try {
    const path = refresh ? "/api/refresh" : "/api/catalog";
    const data = await (
      await api(
        `${path}?view=${$("view").value}`,
        refresh ? { method: "POST" } : {},
      )
    ).json();
    closeDetail();
    state.catalog = data;
    state.selected = new Set(
      [...state.selected].filter((id) => data.clips.some((r) => r.id === id)),
    );
    state.page = 0;
    summary();
    renderList();
  } catch (e) {
    notice(`一覧を読み込めませんでした: ${e.message}`, true);
  } finally {
    $("refresh").disabled = false;
  }
}
async function openClip(id) {
  closeDetail();
  const row = state.catalog.clips.find((r) => r.id === id);
  if (!row) return;
  state.clip = row;
  const epoch = state.epoch;
  state.detailAbort = new AbortController();
  $("detail").hidden = false;
  $("workspace").classList.add("has-detail");
  $("detail-title").textContent = row.title || row.source_id;
  $("detail-source").textContent = row.source_id;
  $("detail-id").textContent = row.id;
  $("detail-meta").textContent =
    `${row.frames} frames · ${row.duration_seconds.toFixed(2)} 秒 · ${row.width}×${row.height} · ${row.fps} fps`;
  $("versions").textContent = "版と統計を読み込み中…";
  $("quality").textContent = "";
  $("ranges").textContent = "";
  $("timelines").textContent = "";
  $("frame-data").textContent = "";
  renderList();
  try {
    const d = await (
      await api(
        `/api/clips/${encodeURIComponent(id)}?revision=${state.catalog.revision}&view=${$("view").value}`,
        { signal: state.detailAbort.signal },
      )
    ).json();
    if (epoch !== state.epoch) return;
    state.detail = d;
    state.versions = Object.fromEntries(
      Object.entries(d.targets).map(([t, v]) => [t, v.default_version]),
    );
    state.annotations = {};
    $("detail-warnings").textContent = [...d.errors, ...d.warnings].join("\n");
    $("detail-warnings").hidden = !d.errors.length && !d.warnings.length;
    $("frame-index").max = row.frames - 1;
    $("frame-index").value = 0;
    $("frame-total").textContent = `/ ${row.frames - 1}`;
    $("download-video").href = videoUrl(true);
    $("download-video").hidden = !row.video_available;
    renderVersions();
    useOriginal(0);
    await Promise.all(["ball", "player"].map((t) => loadVersion(t, epoch)));
    if (epoch !== state.epoch) return;
    renderQuality();
    if (innerWidth < 1100)
      $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    if (e.name !== "AbortError")
      notice(`詳細を読み込めませんでした: ${e.message}`, true);
  }
}
function videoUrl(download = false) {
  return `/api/clips/${encodeURIComponent(state.clip.id)}/video?revision=${state.detail.revision}${download ? "&download=true" : ""}`;
}
function versionInfo(target) {
  return state.detail?.targets[target].versions.find(
    (v) => v.id === state.versions[target],
  );
}
function renderVersions() {
  $("versions").innerHTML = ["ball", "player"]
    .map((t) => {
      const data = state.detail.targets[t];
      const current = versionInfo(t);
      return `<label>${targetNames[t]}のJSON<select data-version="${t}" aria-label="${targetNames[t]}のJSONの版">${!data.default_version ? '<option value="">JSON未選択</option>' : ""}${data.versions.map((v) => `<option value="${v.id}" ${state.versions[t] === v.id ? "selected" : ""}>${esc(v.label)} · ${esc(labels[v.statistics.state])} · ${v.statistics.reviewed}/${v.statistics.frames}</option>`).join("")}</select><span class="version-info">${current ? esc(current.path) + (current.member ? ` / ${esc(current.member)}` : "") : "採用済み注釈なし"}</span></label>`;
    })
    .join("");
}
async function loadVersion(target, epoch) {
  const id = state.versions[target];
  if (!id) {
    state.annotations[target] = null;
    return;
  }
  try {
    const data = await (
      await api(
        `/api/clips/${encodeURIComponent(state.clip.id)}/annotations/${target}/${id}?revision=${state.detail.revision}`,
        { signal: state.detailAbort.signal },
      )
    ).json();
    if (epoch === state.epoch && state.versions[target] === id)
      state.annotations[target] = data;
  } catch (e) {
    if (
      e.name !== "AbortError" &&
      epoch === state.epoch &&
      state.versions[target] === id
    ) {
      state.annotations[target] = null;
      toast(`${targetNames[target]}: ${e.message}`);
    }
  }
}
function stats(target) {
  return (
    state.annotations[target]?.statistics ||
    versionInfo(target)?.statistics ||
    state.detail.targets[target].missing
  );
}
function renderQuality() {
  if (!state.detail) return;
  const b = stats("ball"),
    p = stats("player");
  $("quality").innerHTML =
    `<div class="quality-status">ボール ${badge(b.state)} 選手 ${badge(p.state)}</div><table class="quality-table"><thead><tr><th>指標</th><th>ボール</th><th>選手</th></tr></thead><tbody>${[
      ["確認済み", "reviewed"],
      ["未確認", "unreviewed"],
      ["未解決・notesのあるフレーム", "uncertain_frames"],
      ["座標のあるフレーム", "localized_frames"],
      ["確認済み・対象なし", "absent_frames"],
      ["座標nullの対象数", "null_objects"],
    ]
      .map(
        ([label, key]) =>
          `<tr><td>${label}</td><td>${b[key]}</td><td>${p[key]}</td></tr>`,
      )
      .join(
        "",
      )}<tr><td>補間されたボール / 推定bbox</td><td>${b.interpolated_objects}</td><td>${p.inferred_boxes}</td></tr><tr><td>画面外のボール / 画面切れbbox</td><td>${b.out_of_frame_objects}</td><td>${p.truncated_boxes}</td></tr><tr><td>イベント境界 / 遮蔽bbox</td><td>${b.event_boundaries}</td><td>${p.occluded_boxes}</td></tr></tbody></table><p class="caption">未解決率: ボール ${b.reviewed ? ((100 * b.uncertain_frames) / b.reviewed).toFixed(1) : "—"}%、選手 ${p.reviewed ? ((100 * p.uncertain_frames) / p.reviewed).toFixed(1) : "—"}%（確認済みフレームを分母とした指標。精度スコアではありません）</p>`;
  $("timelines").innerHTML = ["ball", "player"]
    .map(
      (t) =>
        `<div class="timeline-row"><span>${targetNames[t]}</span><canvas data-timeline="${t}" width="1000" height="22" aria-label="${targetNames[t]}の確認状態。クリックでフレーム移動"></canvas></div>`,
    )
    .join("");
  $("ranges").innerHTML = ["ball", "player"]
    .map((t) => {
      const s = stats(t);
      const rangeButtons = (ranges, kind) =>
        ranges.length
          ? ranges
              .map(
                ([a, z]) =>
                  `<button class="range-button" data-jump="${a}" title="${kind} [${a},${z})">[${a}, ${z})</button>`,
              )
              .join("")
          : "なし";
      return `<div class="ranges-group"><h4>${targetNames[t]}</h4><div class="range-label">未確認（半開区間）</div>${rangeButtons(s.unreviewed_ranges, "未確認")}<div class="range-label">未解決・notesあり</div>${rangeButtons(s.uncertain_ranges, "未解決")}<ul class="error-list">${s.errors
        .slice(0, 50)
        .map((e) => `<li>${esc(e)}</li>`)
        .join(
          "",
        )}</ul><div class="caption">${s.issues.map(esc).join("<br>")}</div></div>`;
    })
    .join("");
  $("json-downloads").innerHTML = ["ball", "player"]
    .filter((t) => state.versions[t] && state.annotations[t])
    .map(
      (t) =>
        `<a class="button" download href="/api/clips/${encodeURIComponent(state.clip.id)}/annotations/${t}/${state.versions[t]}?revision=${state.detail.revision}&download=true">${targetNames[t]}JSON</a>`,
    )
    .join("");
  updatePreviewAvailability();
  drawTimelines();
  updateFrame();
}
function updatePreviewAvailability() {
  if (!state.detail) return;
  const target = $("overlay-target").value;
  const targets = target === "both" ? ["ball", "player"] : [target];
  $("render-preview").disabled =
    !!state.previewAbort ||
    !state.clip.video_available ||
    !targets.some((t) => state.annotations[t]) ||
    targets.some((t) => stats(t).state === "invalid");
  $("handoff-one").disabled = state.clip.active;
  $("handoff-refine").disabled = state.clip.active;
}
function currentFrame() {
  const times = state.detail?.times || [];
  const time = $("video").currentTime || 0;
  let lo = 0,
    hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= time + 0.00001) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}
function drawTimelines() {
  if (!state.detail) return;
  for (const canvas of document.querySelectorAll("[data-timeline]")) {
    const ctx = canvas.getContext("2d"),
      target = canvas.dataset.timeline,
      n = state.clip.frames;
    ctx.fillStyle = "#cad1d8";
    ctx.fillRect(0, 0, 1000, 22);
    for (const f of state.annotations[target]?.timeline || []) {
      ctx.fillStyle = !f.reviewed
        ? "#cad1d8"
        : f.uncertain
          ? "#d6a044"
          : "#2d9d89";
      const start = Math.floor((f.frame * 1000) / n);
      const end = Math.ceil(((f.frame + 1) * 1000) / n);
      ctx.fillRect(start, 0, end - start, 22);
    }
    ctx.fillStyle = "#17373f";
    ctx.fillRect((currentFrame() * 1000) / n, 0, 2, 22);
  }
}
function updateFrame() {
  if (!state.detail) return;
  const index = currentFrame();
  if (document.activeElement !== $("frame-index"))
    $("frame-index").value = index;
  const lines = [
    `frame ${index} · source frame ${state.detail.source_frames[index]} · ${state.detail.times[index].toFixed(4)} 秒${state.detail.is_target[index] ? "" : " · 参考区間（注釈対象に含む）"}`,
  ];
  for (const target of ["ball", "player"]) {
    const row = state.annotations[target]?.annotation.frames.find(
      (f) => f.frame_index === index,
    );
    lines.push(
      `\n${targetNames[target]} / ${row ? (row.reviewed ? "確認済み" : "未確認") : "JSONなし"}`,
    );
    if (row) lines.push(JSON.stringify(row, null, 2));
  }
  $("frame-data").textContent = lines.join("\n");
  drawTimelines();
}
function seekFrame(index) {
  if (!state.detail) return;
  index = Math.max(0, Math.min(state.clip.frames - 1, Number(index) || 0));
  $("video").pause();
  $("frame-index").value = index;
  $("video").currentTime = state.detail.times[index] + 0.000001;
  updateFrame();
}
function useOriginal(time = $("video").currentTime || 0) {
  releasePreview();
  if (!state.detail || !state.clip.video_available) return;
  const video = $("video");
  video.src = videoUrl();
  video.addEventListener(
    "loadedmetadata",
    () => {
      video.currentTime = Math.min(time, video.duration || time);
      video.playbackRate = Number($("speed").value);
    },
    { once: true },
  );
  $("video-mode").textContent = "元動画";
  $("preview-status").textContent =
    "overlayはメモリ上で生成され、画面を離れると破棄されます。";
  updatePreviewAvailability();
}
async function renderPreview() {
  if (!state.detail) return;
  const epoch = state.epoch,
    time = $("video").currentTime || 0;
  state.previewAbort?.abort();
  const controller = new AbortController();
  state.previewAbort = controller;
  $("cancel-preview").hidden = false;
  $("preview-status").textContent = "元動画と選択JSONからoverlayを生成中…";
  updatePreviewAvailability();
  try {
    const response = await api(
      `/api/clips/${encodeURIComponent(state.clip.id)}/preview`,
      {
        ...jsonPost({
          revision: state.detail.revision,
          versions: state.versions,
          target: $("overlay-target").value,
        }),
        signal: controller.signal,
      },
    );
    const blob = await response.blob();
    if (epoch !== state.epoch || controller.signal.aborted) return;
    if (state.blobUrl) URL.revokeObjectURL(state.blobUrl);
    state.blobUrl = URL.createObjectURL(blob);
    const video = $("video");
    video.pause();
    video.src = state.blobUrl;
    video.addEventListener(
      "loadedmetadata",
      () => {
        video.currentTime = Math.min(time, video.duration || time);
        video.playbackRate = Number($("speed").value);
      },
      { once: true },
    );
    $("video-mode").textContent = "一時overlay";
    $("preview-status").textContent =
      "生成済み · メモリ上の動画です。元動画への切替・詳細を閉じる操作で破棄します。";
  } catch (e) {
    if (e.name !== "AbortError") {
      toast(e.message);
      $("preview-status").textContent = e.message;
    } else if (epoch === state.epoch)
      $("preview-status").textContent = "生成を中止しました。";
  } finally {
    if (state.previewAbort === controller) {
      state.previewAbort = null;
      $("cancel-preview").hidden = true;
      updatePreviewAvailability();
    }
  }
}
function saveBlob(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}
async function showHandoff(selections) {
  try {
    const data = await (
      await api(
        "/api/handoff",
        jsonPost({
          revision: state.detail?.revision || state.catalog.revision,
          view: $("view").value,
          selections,
        }),
      )
    ).json();
    state.handoff = data;
    $("handoff-target").innerHTML = Object.keys(data.requests)
      .map(
        (target) => `<option value="${target}">${targetNames[target]}</option>`,
      )
      .join("");
    selectHandoffTarget();
    $("handoff-dialog").showModal();
  } catch (e) {
    toast(e.message);
  }
}
function selectHandoffTarget() {
  const request = state.handoff.requests[$("handoff-target").value];
  state.handoff.text = request.text;
  state.handoff.selectedManifest = request.manifest;
  $("handoff-text").value = request.text;
}
$("handoff-target").onchange = selectHandoffTarget;
$("refresh").onclick = () => loadCatalog(true);
$("view").onchange = () => loadCatalog();
["search", "source", "sort"].forEach((id) =>
  $(id).addEventListener(id === "search" ? "input" : "change", () => {
    state.page = 0;
    renderList();
  }),
);
$("filters").onclick = (e) => {
  const b = e.target.closest("[data-filter]");
  if (b) {
    state.filter = b.dataset.filter;
    state.page = 0;
    renderList();
  }
};
$("clips").onclick = (e) => {
  const b = e.target.closest("[data-open]");
  if (b) openClip(b.dataset.open);
};
$("clips").onchange = (e) => {
  const id = e.target.dataset.select;
  if (id) {
    if (e.target.checked && state.selected.size >= 20) {
      e.target.checked = false;
      toast("一度に選択できる動画は20件までです");
      return;
    }
    if (e.target.checked) state.selected.add(id);
    else state.selected.delete(id);
    selectionCount();
  }
};
$("select-page").onclick = () => {
  filtered()
    .slice(state.page * state.pageSize, (state.page + 1) * state.pageSize)
    .filter((r) => r.delegatable)
    .slice(0, Math.max(0, 20 - state.selected.size))
    .forEach((r) => state.selected.add(r.id));
  renderList();
};
$("clear-selection").onclick = () => {
  state.selected.clear();
  renderList();
};
$("prev-page").onclick = () => {
  state.page--;
  renderList();
};
$("next-page").onclick = () => {
  state.page++;
  renderList();
};
$("close-detail").onclick = closeDetail;
$("versions").onchange = async (e) => {
  const target = e.target.dataset.version;
  if (!target) return;
  useOriginal();
  state.versions[target] = e.target.value || null;
  state.annotations[target] = null;
  const epoch = state.epoch;
  renderVersions();
  await loadVersion(target, epoch);
  if (epoch === state.epoch) renderQuality();
};
$("overlay-target").onchange = () => useOriginal();
$("render-preview").onclick = renderPreview;
$("original").onclick = () => useOriginal();
$("cancel-preview").onclick = () => state.previewAbort?.abort();
$("frame-prev").onclick = () => seekFrame(currentFrame() - 1);
$("frame-next").onclick = () => seekFrame(currentFrame() + 1);
$("frame-index").onchange = (e) => seekFrame(e.target.value);
$("speed").onchange = () => {
  $("video").playbackRate = Number($("speed").value);
};
$("video").addEventListener("timeupdate", updateFrame);
$("video").addEventListener("seeked", updateFrame);
$("video").addEventListener("error", () => {
  if ($("video").getAttribute("src"))
    $("preview-status").textContent =
      "動画を読み込めません。再集計してファイル状態を確認してください。";
});
$("timelines").onclick = (e) => {
  const c = e.target.closest("canvas");
  if (c && state.detail) {
    const rect = c.getBoundingClientRect();
    seekFrame(
      Math.floor(((e.clientX - rect.left) / rect.width) * state.clip.frames),
    );
  }
};
$("ranges").onclick = (e) => {
  const b = e.target.closest("[data-jump]");
  if (b) seekFrame(b.dataset.jump);
};
$("handoff-batch").onclick = () =>
  showHandoff([...state.selected].map((id) => ({ clip_id: id })));
$("handoff-one").onclick = () =>
  showHandoff([{ clip_id: state.clip.id, versions: state.versions }]);
$("handoff-refine").onclick = () =>
  showHandoff([
    { clip_id: state.clip.id, versions: state.versions, refine: true },
  ]);
$("close-handoff").onclick = () => $("handoff-dialog").close();
$("copy-handoff").onclick = async () => {
  try {
    await navigator.clipboard.writeText(state.handoff.text);
    toast("依頼文をコピーしました");
  } catch {
    $("handoff-text").select();
    toast("コピーできない環境です。選択された依頼文を手動でコピーしてください");
  }
};
$("download-handoff").onclick = () =>
  saveBlob(
    state.handoff.text,
    "text/plain;charset=utf-8",
    `annotation_request_${$("handoff-target").value}.txt`,
  );
$("download-manifest").onclick = () =>
  saveBlob(
    JSON.stringify(state.handoff.selectedManifest, null, 2),
    "application/json",
    `annotation_targets_${$("handoff-target").value}.json`,
  );
window.addEventListener("pagehide", () => {
  state.detailAbort?.abort();
  releasePreview();
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted && state.detail) useOriginal(0);
});
await loadCatalog();
