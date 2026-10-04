import { SceneView, COLORS } from "./scene.mjs";
const $ = (id) => document.getElementById(id),
  gallery = $("gallery"),
  dialog = $("lightbox");
let scenes = [],
  data = null,
  group = null,
  selectedImage = 0,
  request = null,
  observer = null,
  sceneSequence = 0,
  detailSequence = 0;
const view = new SceneView($("view"), (id) => selectGroup(id, true));
function status(message) {
  $("status").textContent = message;
  $("status").hidden = !message;
}
function imageURL(sample, width, mode = "overlay") {
  return `/api/scenes/${encodeURIComponent(data.id)}/images/${encodeURIComponent(sample.id)}?revision=${encodeURIComponent(data.revision)}&width=${width}&mode=${mode}`;
}
function filteredGroups() {
  return data.groups.filter(
    (g) =>
      $("split-filter").value === "all" || g.split === $("split-filter").value,
  );
}
function applySplit() {
  if (!data) return;
  view.setScene({ ...data, groups: filteredGroups() });
  renderList();
  const selected =
    filteredGroups().find((g) => g.id === group?.id) || filteredGroups()[0];
  if (selected) selectGroup(selected.id);
  else {
    group = null;
    gallery.replaceChildren();
    $("sample-info").textContent = "このsplitには採用軌道がありません";
  }
}
function inspectSample(sample) {
  const c = sample.target_counts;
  $("sample-info").replaceChildren();
  for (const value of [
    sample.id,
    `frame ${sample.frame} · ${sample.view}`,
    `target ${sample.target_court ?? "未記録"} · ${sample.target_coverage ?? "未記録"}`,
    `${sample.resolution.join("×")} · 採用 / ${group.split}`,
    c
      ? `target可視 ${c.visible}/${c.total} · 画面内不可視 ${c.in_frame_hidden} · 画面外 ${c.out_of_frame}`
      : "target投影：未記録",
  ]) {
    const row = document.createElement("div");
    row.textContent = value;
    $("sample-info").append(row);
  }
}
function resizeTiles() {
  gallery.style.setProperty(
    "--tile-height",
    `${Math.max(30, (gallery.clientHeight + 8) / 5.5 - 8)}px`,
  );
}
new ResizeObserver(resizeTiles).observe(gallery);
async function loadScene() {
  if (dialog.open) dialog.close();
  if ($("reject-dialog").open) $("reject-dialog").close();
  const scene = scenes.find((s) => s.id === $("scene").value);
  const sequence = ++sceneSequence;
  ++detailSequence;
  request?.abort();
  request = new AbortController();
  observer?.disconnect();
  gallery.replaceChildren();
  $("trajectories").replaceChildren();
  $("stat-cards").replaceChildren();
  for (const id of ["splits", "shapes", "coverage"]) $(id).replaceChildren();
  $("schema").textContent = "";
  $("image-count").textContent = "";
  $("trajectory-count").textContent = "";
  $("sample-info").textContent = "画像にカーソルを合わせて確認";
  $("publication-badge").textContent = "";
  data = null;
  group = null;
  view.groups = [];
  view.courts = [];
  view.selected = null;
  view.schedule();
  if (!scene) return;
  if (scene.error) {
    status(scene.error);
    return;
  }
  status(`${scene.id} を読み込んでいます…`);
  $("scene-title").textContent = scene.id;
  try {
    const response = await fetch(
      `/api/scenes/${encodeURIComponent(scene.id)}?revision=${scene.revision}`,
      { signal: request.signal, cache: "no-store" },
    );
    const result = await response.json();
    if (!response.ok) throw Error(result.detail || "読み込みに失敗しました");
    if (sequence !== sceneSequence) return;
    data = result;
    renderStats();
    status("");
    if (data.groups.length) applySplit();
    else status("表示できる軌道がありません。");
  } catch (error) {
    if (error.name !== "AbortError" && sequence === sceneSequence)
      status(error.message);
  }
}
function renderList() {
  $("trajectories").replaceChildren();
  $("trajectory-count").textContent =
    `${filteredGroups().length} / ${data.groups.length}`;
  for (const g of filteredGroups()) {
    const button = document.createElement("button");
    button.className = "trajectory";
    button.dataset.group = g.id;
    button.style.setProperty("--split", COLORS[g.split] || "#fff");
    button.setAttribute("aria-current", "false");
    const name = document.createElement("span");
    name.className = "name";
    const label = document.createElement("span");
    label.textContent = g.trajectory.trajectory_id;
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.textContent = "●";
    dot.title = g.split;
    name.append(label, dot);
    const description = document.createElement("span");
    description.className = "description";
    description.textContent = `${g.trajectory.shape} · ${g.samples.length} views`;
    button.append(name, description);
    button.onclick = () => selectGroup(g.id);
    $("trajectories").append(button);
  }
}
function selectGroup(id, scroll = false) {
  if (!data) return;
  group = data.groups.find((g) => g.id === id);
  if (!group) return;
  view.select(id);
  for (const button of $("trajectories").children) {
    button.setAttribute("aria-current", String(button.dataset.group === id));
    if (scroll && button.dataset.group === id)
      button.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  $("selected-title").textContent = group.trajectory.trajectory_id;
  $("image-count").textContent =
    `${group.samples.length} images · ${group.split}`;
  inspectSample(group.samples[0]);
  observer?.disconnect();
  gallery.replaceChildren();
  gallery.scrollTop = 0;
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          observer.unobserve(img);
        }
      }
    },
    { root: gallery, rootMargin: "250px" },
  );
  group.samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.className = "thumbnail";
    button.dataset.sample = sample.id;
    button.setAttribute("aria-label", `${sample.id} を全画面表示`);
    const img = document.createElement("img");
    img.alt = `${sample.id}：画像と教師ラベル`;
    img.decoding = "async";
    img.dataset.src = imageURL(
      sample,
      480,
      $("image-mode").value === "raw" ? "raw" : "overlay",
    );
    img.onerror = () => {
      img.remove();
      const message = document.createElement("span");
      message.className = "failure";
      message.textContent = "画像を読み込めません。クリックして詳細を確認";
      button.append(message);
    };
    button.append(img);
    button.onclick = () => openImage(index);
    button.onmouseenter = () => {
      view.select(group.id, index);
      inspectSample(sample);
    };
    button.onfocus = button.onmouseenter;
    button.onmouseleave = () => {
      view.select(group.id, dialog.open ? selectedImage : null);
      inspectSample(group.samples[dialog.open ? selectedImage : 0]);
    };
    gallery.append(button);
    observer.observe(img);
  });
  resizeTiles();
}
function openImage(index) {
  if (!group) return;
  selectedImage = index;
  $("large-image").removeAttribute("src");
  $("image-error").hidden = true;
  const sample = group.samples[index];
  inspectSample(sample);
  const mode = $("image-mode").value;
  $("large-mode").value = mode;
  $("raw-pane").hidden = mode !== "compare";
  $("image-panes").classList.toggle("compare", mode === "compare");
  $("large-caption").textContent =
    mode === "raw" ? "生成RGB · 保存JPEG" : "教師ラベル · 合成truth";
  $("large-image").alt = `${sample.id}：画像と教師ラベル`;
  $("large-image").src = imageURL(
    sample,
    0,
    mode === "raw" ? "raw" : "overlay",
  );
  $("raw-image").removeAttribute("src");
  if (mode === "compare") $("raw-image").src = imageURL(sample, 0, "raw");
  $("position").textContent =
    `${index + 1} / ${group.samples.length} · ${sample.id}`;
  $("previous").disabled = index === 0;
  $("next").disabled = index === group.samples.length - 1;
  view.select(group.id, index);
  for (const [i, button] of [...gallery.children].entries())
    button.classList.toggle("active", i === index);
  if (!dialog.open) dialog.showModal();
  loadSampleDetail(sample);
}
async function loadSampleDetail(sample) {
  const sequence = ++detailSequence;
  $("large-sample-info").textContent = "sampleの保存値を読み込み中…";
  $("point-table").replaceChildren();
  try {
    const response = await fetch(
      `/api/scenes/${encodeURIComponent(data.id)}/samples/${encodeURIComponent(sample.id)}?revision=${data.revision}`,
    );
    const detail = await response.json();
    if (sequence !== detailSequence || !dialog.open) return;
    if (!response.ok) throw Error(detail.detail || "保存値を取得できません");
    $("large-sample-info").replaceChildren();
    const pose = detail.camera_to_scene,
      k = detail.intrinsics;
    const rasters = data.publication.renderer_rasters_retained;
    const rasterNote =
      rasters === false
        ? "alpha/depthは公開後に保持しません。"
        : rasters === true
          ? "alpha/depthは保存されています。"
          : "alpha/depthの保持状態は未記録です。";
    for (const text of [
      detail.id,
      `${data.id} · ${detail.split} · 採用`,
      `target ${detail.target_court ?? "未記録"} / ${detail.target_coverage ?? "未記録"}`,
      `camera ${detail.camera}`,
      `frame ${detail.frame} · ${detail.view}`,
      `scene位置(m): ${[pose[3], pose[7], pose[11]].map((v) => v.toFixed(2)).join(", ")}`,
      `画像 ${detail.resolution.join("×")} / fx,fy ${k[0].toFixed(1)},${k[4].toFixed(1)} px`,
      "● renderer可視 / ○ 画面内不可視",
      `画面外・カメラ後方は描画しません。可視性は生成時ゲートの保存値です。${rasterNote}`,
    ]) {
      const row = document.createElement("p");
      row.textContent = text;
      $("large-sample-info").append(row);
    }
    renderPointTable($("point-table"), detail.visibility, detail.target_court);
  } catch (error) {
    if (sequence === detailSequence)
      $("large-sample-info").textContent = error.message;
  }
}
function renderPointTable(root, visibility, target) {
  root.replaceChildren();
  if (!visibility.projection_recorded) {
    root.textContent = "投影：未記録 · renderer可視性：未記録";
    return;
  }
  const geometry = {
      in_frame: "画面内",
      out_of_frame: "画面外",
      behind_camera: "後方",
      unrecorded: "未記録",
    },
    renderer = { visible: "可視", not_visible: "不可視", unrecorded: "未記録" };
  for (const court of visibility.courts) {
    const heading = document.createElement("h3");
    heading.textContent = `${court.id}${court.id === target ? " · target" : ""} · ${court.counts.visible}/${court.counts.total} 可視 · ${court.counts.unknown_renderer} 未記録`;
    const table = document.createElement("table");
    for (const point of court.points) {
      const row = table.insertRow();
      row.title = `uv(pixel): ${point.uv.map((v) => v.toFixed(2)).join(", ")}`;
      row.insertCell().textContent = `${point.class} [${point.physical_index}]`;
      row.insertCell().textContent = geometry[point.geometry];
      const state = row.insertCell();
      state.textContent = renderer[point.renderer];
      state.className = point.renderer;
    }
    root.append(heading, table);
  }
}
async function openRejection(sample) {
  if (dialog.open) dialog.close();
  $("large-image").removeAttribute("src");
  $("raw-image").removeAttribute("src");
  const sequence = ++detailSequence,
    scene = data.id,
    revision = data.revision;
  $("reject-title").textContent = `${scene} / ${sample.id} · reject`;
  $("reject-info").textContent = "候補の保存値を読み込み中…";
  $("reject-projection").replaceChildren();
  if (!$("reject-dialog").open) $("reject-dialog").showModal();
  try {
    const response = await fetch(
        `/api/scenes/${encodeURIComponent(scene)}/rejections/${encodeURIComponent(sample.id)}?revision=${revision}`,
      ),
      d = await response.json();
    if (sequence !== detailSequence || !$("reject-dialog").open) return;
    if (!response.ok) throw Error(d.detail || "候補詳細を取得できません");
    $("reject-info").replaceChildren();
    const pose = d.camera_to_scene;
    for (const value of [
      `${d.split} / ${d.trajectory_group} / frame ${d.frame} / ${d.view}`,
      `target ${d.target_court ?? "未記録"} · ${d.resolution.join("×")}`,
      `camera ${d.camera} / scene位置(m) ${[pose[3], pose[7], pose[11]].map((v) => v.toFixed(3)).join(", ")}`,
      `理由: ${d.reasons.join(", ")}`,
      `投影 ${d.visibility.projection_recorded ? "保存" : "未記録"} / renderer可視性は下表。未記録を不可視に置換しません。`,
    ]) {
      const p = document.createElement("p");
      p.textContent = value;
      $("reject-info").append(p);
    }
    const details = document.createElement("details"),
      summary = document.createElement("summary"),
      pre = document.createElement("pre");
    summary.textContent = "保存camera / target bindingの数値";
    pre.textContent = JSON.stringify(
      {
        camera_to_scene: d.camera_to_scene,
        intrinsics: d.intrinsics,
        target_binding: d.target_binding,
      },
      null,
      2,
    );
    details.append(summary, pre);
    $("reject-info").append(details);
    renderPointTable($("reject-projection"), d.visibility, d.target_court);
  } catch (error) {
    if (sequence === detailSequence)
      $("reject-info").textContent = error.message;
  }
}
$("reject-close").onclick = () => $("reject-dialog").close();
$("reject-dialog").addEventListener("close", () => ++detailSequence);
$("large-image").onerror = async () => {
  const src = $("large-image").src;
  let message = "画像を読み込めません。";
  try {
    const response = await fetch(src);
    if (!response.ok) message = (await response.json()).detail || message;
  } catch {}
  if (src === $("large-image").src) {
    $("image-error").textContent = message;
    $("image-error").hidden = false;
  }
};
$("raw-image").onerror = () => {
  $("image-error").textContent =
    "生成RGBを読み込めません。再読込してください。";
  $("image-error").hidden = false;
};
$("previous").onclick = () => {
  if (selectedImage > 0) openImage(selectedImage - 1);
};
$("next").onclick = () => {
  if (selectedImage < group.samples.length - 1) openImage(selectedImage + 1);
};
$("close").onclick = () => dialog.close();
dialog.addEventListener("close", () => {
  ++detailSequence;
  if (group) view.select(group.id);
});
dialog.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    $("previous").click();
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    $("next").click();
  }
});
function bars(id, values, colors = {}) {
  const root = $(id);
  root.replaceChildren();
  const max = Math.max(...Object.values(values), 1);
  for (const [name, value] of Object.entries(values)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("div");
    label.className = "bar-label";
    const text = document.createElement("span"),
      count = document.createElement("span");
    text.textContent = name;
    count.textContent = Number(value).toLocaleString();
    label.append(text, count);
    const bar = document.createElement("div"),
      fill = document.createElement("i");
    bar.className = "bar";
    fill.style.width = `${(value / max) * 100}%`;
    if (colors[name]) fill.style.background = colors[name];
    bar.append(fill);
    row.append(label, bar);
    root.append(row);
  }
}
function renderStats() {
  const m = data.metrics;
  const p = data.publication;
  $("publication-badge").textContent =
    `${data.schema.replace("canonical_court_dataset_", "").toUpperCase()} · ${p.storage_format} · ${p.status}`;
  $("scene-source").textContent =
    `source: ${p.source_video?.split("/").pop() ?? "未記録"} → NHT 3DGS / ${data.courts.length}コート`;
  $("schema").textContent = data.schema;
  const cards = [
    [m.accepted_frame_count.toLocaleString(), "採用画像"],
    [data.groups.length, "カメラ軌道"],
    [m.rejected_frame_count.toLocaleString(), "reject（画像未保存）"],
    [`${(m.accepted_fraction * 100).toFixed(1)}%`, "採用率"],
  ];
  $("stat-cards").replaceChildren();
  for (const [value, label] of cards) {
    const card = document.createElement("div");
    card.className = "stat-card";
    const strong = document.createElement("strong"),
      caption = document.createElement("span");
    strong.textContent = value;
    caption.textContent = label;
    card.append(strong, caption);
    $("stat-cards").append(card);
  }
  bars("splits", m.split_frame_counts, COLORS);
  bars("shapes", data.shapes);
  bars("coverage", m.coverage_counts);
  bars("visibility-classes", m.renderer_visible_points_by_class);
  const targetTable = $("split-targets");
  targetTable.replaceChildren();
  const header = targetTable.createTHead().insertRow();
  for (const label of ["split", ...data.courts.map((c) => c.id)]) {
    const cell = document.createElement("th");
    cell.textContent = label;
    header.append(cell);
  }
  const body = targetTable.createTBody();
  for (const [split, counts] of Object.entries(data.split_targets)) {
    const row = body.insertRow();
    row.insertCell().textContent = split;
    for (const court of data.courts) {
      const cell = row.insertCell();
      cell.textContent = counts[court.id].toLocaleString();
      if (counts[court.id] === 0) cell.className = "zero-count";
    }
  }
  const r = data.rejections;
  $("rejection-state").textContent =
    `${r.count} reject · 候補詳細 ${r.records_recorded ? `${r.record_count}件保存` : "未記録"} · 画像未保存`;
  bars("rejection-reasons", r.reason_counts);
  $("rejection-details").open = false;
  const rejectedBody = $("rejected-table").tBodies[0];
  rejectedBody.replaceChildren();
  for (const sample of r.samples) {
    const row = rejectedBody.insertRow();
    const cell = row.insertCell(),
      button = document.createElement("button");
    button.textContent = `${sample.id} / ${sample.frame}`;
    button.onclick = () => openRejection(sample);
    cell.append(button);
    for (const value of [
      `${sample.group} / ${sample.view} / ${sample.split}`,
      sample.target_court ?? "未記録",
      sample.reasons.join(", "),
      `${sample.projection_recorded ? "保存" : "未記録"} / ${sample.renderer_recorded ? "保存" : "未記録"}`,
    ])
      row.insertCell().textContent = value;
  }
  const dl = $("publication-details");
  dl.replaceChildren();
  for (const [key, value] of Object.entries(p)) {
    const dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent =
      value === null
        ? "未記録"
        : typeof value === "object"
          ? JSON.stringify(value)
          : String(value);
    dl.append(dt, dd);
  }
}
async function renderCatalog() {
  const response = await fetch("/api/catalog", { cache: "no-store" });
  if (!response.ok) throw Error("公開データ体系の読み込みに失敗しました");
  const entries = await response.json(),
    body = $("catalog-table").tBodies[0];
  body.replaceChildren();
  let accepted = 0,
    rejected = 0;
  for (const p of entries) {
    const row = body.insertRow();
    if (p.error) {
      row.insertCell().textContent = p.id;
      const cell = row.insertCell();
      cell.colSpan = 4;
      cell.textContent = p.error;
      continue;
    }
    accepted += p.sample_count;
    rejected += p.rejected_count;
    const sceneCell = row.insertCell(),
      button = document.createElement("button");
    button.textContent = `${p.id} / ${p.schema.replace("canonical_court_dataset_", "").toUpperCase()}`;
    button.onclick = () => {
      $("scene").value = p.id;
      loadScene();
      scrollTo({ top: 0, behavior: "smooth" });
    };
    sceneCell.append(button);
    for (const value of [
      p.source_video?.split("/").pop() ?? "未記録",
      `${p.storage_format} / ${p.status}`,
      ["train", "validation", "test"]
        .map((s) => p.split_frame_counts[s].toLocaleString())
        .join(" / "),
      `${p.sample_count.toLocaleString()} / ${p.rejected_count}（画像未保存）`,
    ])
      row.insertCell().textContent = value;
  }
  $("catalog-total").textContent =
    `${accepted.toLocaleString()}採用 · ${rejected} reject`;
}
$("scene").onchange = loadScene;
$("split-filter").onchange = applySplit;
$("image-mode").onchange = () => {
  if (group) {
    const index = selectedImage;
    selectGroup(group.id);
    if (dialog.open) openImage(index);
  }
};
$("large-mode").onchange = () => {
  $("image-mode").value = $("large-mode").value;
  $("image-mode").onchange();
};
$("reset").onclick = () => view.reset();
try {
  const response = await fetch("/api/scenes", { cache: "no-store" });
  if (!response.ok) throw Error("シーン一覧の読み込みに失敗しました");
  scenes = await response.json();
  for (const scene of scenes) {
    const option = document.createElement("option");
    option.value = scene.id;
    option.textContent = scene.id;
    $("scene").append(option);
  }
  if (scenes.length) await loadScene();
  else status("公開済みのCourtデータセットがありません。");
  await renderCatalog();
} catch (error) {
  status(error.message);
}
