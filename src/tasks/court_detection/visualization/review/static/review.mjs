// Court-only presentation; saved state and derived targets stay separate.
export const stateLabels = {
  in_frame_visibility_unknown: "画像内 · 遮蔽不明",
  renderer_visible: "renderer可視",
  renderer_not_visible: "renderer不可視",
  out_of_frame: "画面外",
  behind_camera: "camera背面",
};
export function pointStateLabel(point) {
  return stateLabels[point.state] || "状態未提供";
}
export function canFocus(point) {
  return point.in_frame && point.in_front !== false
    && Number.isFinite(point.x) && Number.isFinite(point.y);
}
export function targetDescription(kind) {
  const meanings = {
    seg: "SEG · backgroundを含む7領域の被覆率",
    line: "LINE · 通常線7.5 cm / baseline 15 cmの被覆率",
    semantic_line: "semantic LINE · backgroundを含む12種類の線の被覆率",
  };
  return meanings[kind] || "保存KP · 元画像pixel / 14点のchannel順";
}
const $ = (id) => document.getElementById(id);
function node(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function detail(title, children) {
  const el = node("details");
  el.append(node("summary", title), ...children);
  return el;
}
function table(headers, rows) {
  const el = node("table");
  const header = node("tr");
  header.append(...headers.map((x) => node("th", x)));
  const head = node("thead"); head.append(header);
  const body = node("tbody");
  for (const values of rows) {
    const row = node("tr"); row.append(...values.map((x) => node("td", x)));
    body.append(row);
  }
  el.append(head, body);
  return el;
}

export function createCourtReview(viewer, filterChanged) {
  document.body.dataset.reviewTask = "court";
  const css = node("link"); css.rel = "stylesheet"; css.href = "/task-static/review.css";
  document.head.append(css);
  const filter = node("select"); filter.id = "court-sample-filter";
  filter.setAttribute("aria-label", "Court注釈の確認候補");
  filter.append(new Option("注釈の確認候補: すべて", ""),
    new Option("画面外のKPあり", "out_of_frame"),
    new Option("画像内・renderer不可視のKPあり", "renderer_not_visible"),
    new Option("重複座標あり", "duplicate_coordinates"));
  filter.onchange = filterChanged;
  $("scene-search").after(filter);
  const section = node("section", undefined, "court-inspection"); section.id = "court-inspection";
  section.hidden = true;
  $("frame-review-section").before(section);
  const toolbar = node("div", undefined, "court-tools");
  const rgb = node("button", "画像だけ"); rgb.id = "court-rgb";
  const kp = node("button", "保存KP"); kp.id = "court-kp";
  const reference = node("label", undefined, "check");
  const toggle = node("input"); toggle.type = "checkbox"; toggle.id = "court-reference";
  reference.append(node("span", "不可視点を参考表示"), toggle);
  const notice = node("p", undefined, "court-layer-note"); notice.id = "court-layer-note";
  toolbar.append(rgb, kp);
  for (const [kind, label] of [["seg", "SEG"], ["line", "LINE"], ["semantic_line", "semantic LINE"]]) {
    const button = node("button", label);
    button.dataset.target = kind;
    button.onclick = () => setLayer(kind);
    toolbar.append(button);
  }
  toolbar.append(reference, notice);
  document.querySelector(".stage").before(toolbar);
  let current = null;
  function layerNote() {
    const kind = $("raster").value;
    const target = current?.targets.find((t) => t.kind === kind);
    notice.textContent = !$("show-gt").checked ? "画像のみ · 教師overlay非表示" : kind
      ? `${targetDescription(kind)}。保存マスクではありません。KP14→homography→原画像解像度で生成${target?.available === false ? "（このsampleでは利用不可）" : ""}。遮蔽マスクではなく、KPの誤りも引き継ぎます。`
      : targetDescription("");
  }
  function setLayer(kind) {
    const rawOnly = kind === "rgb";
    $("show-gt").checked = !rawOnly;
    $("show-pred").checked = false;
    $("raster").value = ["rgb", "kp"].includes(kind) ? "" : kind;
    $("raster").dispatchEvent(new Event("change"));
    viewer.configure({ gt: !rawOnly, pred: false, highlightPoint: null });
    layerNote();
  }
  rgb.onclick = () => setLayer("rgb");
  kp.onclick = () => setLayer("kp");
  toggle.onchange = () => viewer.configure({ referencePoints: toggle.checked });
  $("raster").addEventListener("change", layerNote);
  $("show-gt").addEventListener("change", layerNote);

  return {
    clear() {
      current = null; section.hidden = true; section.replaceChildren();
      viewer.configure({ highlightPoint: null });
    },
    dataset(dataset) {
      $("dataset-overview-open").hidden = false;
      filter.options[2].disabled = dataset?.source_kind !== "synthetic_court";
      if (filter.options[2].disabled && filter.value === "renderer_not_visible") filter.value = "";
      $("scene-review-meta").hidden = false;
      $("scene-review-meta").textContent = dataset
        ? `${dataset.source_kind === "synthetic_court" ? "合成V3 · target court限定" : "実写 · 保存KP14"} / split ${dataset.split} / 1 sample = 1画像`
        : "dataset未選択";
    },
    frame(item, scene) {
      current = item.court_review; section.hidden = !current;
      viewer.configure({ highlightPoint: null });
      if (!current) return;
      const synthetic = current.source === "synthetic_court";
      const heading = node("div", undefined, "court-section-heading");
      const raw = node("a", "raw注釈 ↗"); raw.id = "court-annotation-link";
      raw.href = `/api/court/annotation?${new URLSearchParams({ scene })}`;
      raw.target = "_blank"; raw.rel = "noopener";
      heading.append(node("h2", "保存教師の検品"), raw);
      const origin = node("p", synthetic
        ? `合成投影 · ${current.target_court} · pose教師あり`
        : "実写KP注釈 · 遮蔽/visibility未保存 · pose教師なし", "court-origin");
      const boundary = synthetic
        ? `画面外（前方）${current.counts.out_of_frame || 0} · camera背面 ${current.counts.behind_camera || 0}`
        : `画面外 ${current.counts.out_of_frame || 0}点`;
      const count = node("p", `KP教師採用 ${current.kp_supervised} / ${current.points.length}点 · ${boundary}`, "court-count");
      const why = node("p", synthetic
        ? "採用条件: in_front × in_frame × renderer_visible。不可視は未注釈とは別です。"
        : "readerは画像境界だけで採用を判定。画像内でも遮蔽がないとは断定できません。", "court-explanation");
      const rows = node("div", undefined, "court-point-list"); rows.id = "court-point-list";
      const selected = node("p", "点を選ぶと座標とphysical IDを確認できます。画像内の点は拡大します。", "court-selected");
      selected.id = "court-selected-point";
      for (const point of current.points) {
        const button = node("button", undefined, `court-point state-${point.state}`);
        button.dataset.channel = point.channel;
        button.setAttribute("aria-label", `${point.channel} ${point.name} ${pointStateLabel(point)}`);
        const name = node("span", undefined, "court-point-name");
        name.append(node("b", String(point.channel)), node("span", point.name));
        const value = node("small", `${point.x.toFixed(1)}, ${point.y.toFixed(1)} px · ${point.kp_supervised ? "KP教師" : "KP対象外"}`);
        button.append(name, value, node("span", pointStateLabel(point), "court-point-state"));
        button.onclick = () => {
          for (const row of rows.children) row.classList.toggle("selected", row === button);
          selected.textContent = `#${point.channel} ${point.name} · physical ${point.physical_index} · (${point.x.toFixed(2)}, ${point.y.toFixed(2)}) px · ${pointStateLabel(point)}${canFocus(point) ? "" : " · 画像内への移動不可"}`;
          if (canFocus(point)) viewer.focusPoint(point);
          else viewer.configure({ highlightPoint: null });
        };
        rows.append(button);
      }
      const identity = detail("schema / source / sample", [
        node("pre", JSON.stringify({
          dataset: current.dataset, split: current.split, source_split: current.source_split,
          sample: current.sample, scene: current.scene, camera: current.camera_id,
          trajectory_group: current.trajectory_group, units: current.coordinate_units,
          source_schema: current.source_schema, kp_schema: current.keypoint_schema,
          annotation: current.annotation_path, image: current.image_path,
        }, null, 2)),
      ]);
      const targets = detail("派生targetのschemaと読み方", [
        node("p", "SEG/LINE/semantic LINEは保存教師ではなく、選択した1面の幾何から生成します。previewは原画像解像度・augmentationなし。学習では変換後の解像度・paddingで同じgeneratorを使います。"),
        ...current.targets.map((t) => node("p", `${targetDescription(t.kind)} / ${t.available ? "生成可" : "利用不可"}\n${t.schema}`, "court-schema")),
        node("p", "7領域/12線のクラス分布は画素ごとに合計1。8倍格子の被覆率を表示し、画像外は外挿しません。KP heatmapも学習時の派生教師です。"),
      ]);
      section.replaceChildren(heading, origin, count, why, rows, selected,
        ...current.flags.map((f) => node("p", f, "court-flag")), identity, targets);
      layerNote();
    },
    overview(catalog) {
      const content = $("dataset-overview-content"); content.replaceChildren();
      $("dataset-overview-title").textContent = "Court datasetの体系と教師";
      const families = catalog.court_review.families;
      content.append(
        node("p", "現行consumer: 実写compact store + Synthetic Court V3 / 画像と疎な教師を検品する画面", "overview-total"),
        node("p", "RGBと保存KP14は元画像pixel。合成のcamera/target変換はmetres。KPから派生するSEG/LINEを独立した正解として扱わず、元注釈と画像を先に確認してください。"),
        table(["現行dataset", "保存教師 / 出自", "train / val / test（現行reader）", "保存件数 / 状態"], families.map((f) => [
          f.label,
          f.source === "synthetic_court" ? "合成投影KP14・renderer可視性・camera pose / 3D再構成scene" : "実写のordered KP14 / upstream TennisCourtDetector注釈",
          ["train", "val", "test"].map((s) => f.splits[s]?.available ? f.splits[s].count.toLocaleString() : "—").join(" / "),
          `${f.stored_count?.toLocaleString() ?? "不明"} / ${f.reason ? `利用不可: ${f.reason}` : "現物あり"}`,
        ])),
        node("p", "実写はsourceのtrain/valを維持し、testは提供されません。合成はtrajectory group単位のtrain/validation/testで、validationをUIのvalへ対応付けます。単画像sampleのframeは0です。", "muted"),
        node("h2", "保存と派生を分けて読む"),
        table(["教師", "保存 / 派生", "可視性・未提供の意味"], [
          ["実写KP14", "保存座標（ordered channel）", "保存visibility/遮蔽なし。readerは画像境界でKP採用を導出。画面外の座標も保持。"],
          ["合成V3 KP14", "保存投影（camera-view名→physical ID）", "in_front / in_frame / renderer_visibleを保存。target courtだけを採用。"],
          ["SEG / LINE / semantic LINE", "両sourceともKP14→homographyのオンザフライ派生", "保存マスクなし。幾何領域なので画像内の遮蔽物を取り除く教師ではない。"],
          ["camera pose", "合成のみ保存camera + target bindingから教師化", "実写は教師なし。3D再構成scene由来で独立実測GTではない。"],
        ]),
        node("h2", "除外と契約"),
        node("p", "現行readerはV3 / target_court限定です。旧V1/V2・all-courtsや廃止raw/Web形式を不足datasetとして復活させません。画面上の表示だけで全データの品質合格を保証しません。"),
      );
      for (const f of families) content.append(detail(`${f.label} · 保存schema / 除外 / パス`, [
        node("pre", JSON.stringify({ source_schema: f.schema, storage_schema: f.storage_schema, storage: f.storage, path: f.path,
          excluded_sample_ids: f.excluded_ids, trajectory_groups: f.trajectory_groups,
          rejected_proposals: f.rejected_proposals }, null, 2)),
        ...(f.excluded_ids.length ? [node("p", "除外sampleはsource設定で隔離済み。重複したKPの退化注釈を現行sample一覧へ戻しません。保存件数とreader件数の差です。")] : []),
      ]));
    },
  };
}
