// Optional ball-store review presentation. Court datasets keep their own schema.
export const pointKinds = {
  observed: { label: "観測", color: "#25db97" },
  interpolated: { label: "補間", color: "#ffc857" },
  occlusion_estimated: { label: "遮蔽推定", color: "#d594ff" },
  unresolved: { label: "位置不明（座標なし）", color: "#657b88" },
  out_of_frame: { label: "画面外（座標なし）", color: "#84998d" },
};
export const reviewLabels = {
  reference: "教師対象外（推定・位置不明）",
  unresolved: "位置不明",
  interpolated: "補間",
  occlusion_estimated: "遮蔽推定",
  scored_positive: "教師の正例",
  scored_negative: "教師の負例",
  out_of_frame: "画面外",
  unreviewed: "未レビュー",
  observed: "観測点",
  context: "文脈区間",
  segment_break: "区間の境界",
};
export const sourceLabels = {
  tracknet: "TrackNet", meiji: "Meiji", chat_annotation: "Chat annotation",
};
const $ = (id) => document.getElementById(id);
const number = (value) => Number(value).toLocaleString("ja-JP");
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};

export function ballPointColor(state) {
  // An unrecognised state remains distinct from an observed target.
  return pointKinds[state]?.color || "#9babb1";
}

export function jumpTarget(positions, frame, direction) {
  let low = 0, high = positions.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (positions[middle] < frame || (direction > 0 && positions[middle] === frame))
      low = middle + 1;
    else high = middle;
  }
  return (direction > 0 ? positions[low] : positions[low - 1]) ?? null;
}

export function renderReviewFilters(dataset, ball) {
  $("ball-review-filters").hidden = !ball;
  $("dataset-overview-open").hidden = !ball || !dataset?.overview;
  for (const [id, options, prefix] of [
    ["source-filter", (dataset?.overview?.sources || []).map((s) => [s.id, sourceLabels[s.id] || s.id]), "Source"],
    ["split-filter", Object.keys(dataset?.overview?.splits || {}).map((s) => [s, s]), "Split"],
    ["review-state-filter", Object.entries(reviewLabels), "注釈状態"],
  ]) {
    const previous = $(id).value;
    $(id).replaceChildren(new Option(`${prefix}: すべて`, ""), ...options.map(([value, label]) => new Option(label, value)));
    if (ball && options.some(([value]) => value === previous)) $(id).value = previous;
  }
}

export function renderSceneReview(item) {
  const meta = item?.review;
  $("scene-review-meta").hidden = !meta;
  $("scene-review-meta").textContent = meta
    ? `${sourceLabels[meta.source] || meta.source} · ${meta.split} · 教師対象 ${number(meta.counts.scored_positive + meta.counts.scored_negative)} / ${number(meta.counts.frames)} frame`
    : "";
}

export function renderFrameReview(item) {
  const review = item?.review;
  $("frame-review-section").hidden = !review;
  if (!review) return;
  const states = {
    scored_positive: ["採点対象 · 正例", "保存された観測点を教師として使います。"],
    scored_negative: ["採点対象 · 負例", "レビュー済みの不在・画面外。画像内の正例はありません。"],
    reference: ["採点対象外 · 参考ラベル", "推定位置または位置不明を含むため、フレーム全体を教師から除外します。"],
    unreviewed: ["採点対象外 · 未レビュー", "ボールの不在は確認されていません。教師に使いません。"],
  };
  const [label, reason] = states[review.supervision] || ["状態未提供", "教師状態が提供されていません。"];
  $("frame-supervision").textContent = label;
  $("frame-supervision").dataset.state = review.supervision;
  $("frame-review-reason").textContent = reason;
  const points = item.gt?.points || [];
  $("frame-point-kinds").replaceChildren(...points.map((point) => {
    const row = element("div", undefined, "point-kind-row");
    const dot = element("i");
    dot.style.background = ballPointColor(point.state);
    row.append(dot, element("span", `${point.label || "ball"} · ${pointKinds[point.state]?.label || point.state}`));
    return row;
  }));
  if (!points.length)
    $("frame-point-kinds").append(element("p", review.supervision === "unreviewed" ? "注釈なし" : "レビュー済み · instanceなし", "muted"));
  const eventLabels = { none: "イベントなし", hit: "打球", bounce: "バウンド", unlabeled: "イベント未提供" };
  $("frame-stored-meta").textContent =
    `frame ${item.index} · 保存時刻 ${review.time_seconds.toFixed(3)} 秒 · ${eventLabels[review.event] || review.event}${review.segment_break ? " · 区間境界" : ""}${review.context ? " · 文脈区間（採点可否とは別）" : ""}`;
}

export function renderJumpOptions(review) {
  const previous = $("review-jump-state").value;
  const options = Object.entries(reviewLabels).filter(([key]) => review?.positions[key]?.length);
  $("review-jumps").hidden = !review;
  $("review-jump-state").replaceChildren(...options.map(([key, label]) => new Option(`${label} (${number(review.positions[key].length)})`, key)));
  if (options.some(([key]) => key === previous)) $("review-jump-state").value = previous;
}

export function updateJumpButtons(review, frame) {
  const positions = review?.positions[$("review-jump-state").value] || [];
  $("review-jump-prev").disabled = jumpTarget(positions, frame, -1) === null;
  $("review-jump-next").disabled = jumpTarget(positions, frame, 1) === null;
  $("review-jump-count").textContent = `${number(positions.length)} frame · 再生は全コマ表示`;
}

function table(headers, rows) {
  const node = element("table");
  const head = element("tr");
  head.append(...headers.map((text) => element("th", text)));
  const body = element("tbody");
  for (const values of rows) {
    const row = element("tr");
    row.append(...values.map((text) => element("td", text)));
    body.append(row);
  }
  const thead = element("thead");
  thead.append(head);
  node.append(thead, body);
  const wrapper = element("div", undefined, "overview-table");
  wrapper.append(node);
  return wrapper;
}

export function renderDatasetOverview(dataset, playerDatasets = []) {
  const overview = dataset?.overview;
  const content = $("dataset-overview-content");
  content.replaceChildren();
  if (!overview) return;
  if (overview.selection?.kind === "pose_approved") {
    content.append(element("p", `固定snapshot ${number(overview.selection.parent_clips)} clipsからpose承認済みの${number(overview.clips)} clipsを選択しています。以下の件数はこのsubsetだけの集計です。`, "muted"));
  }
  content.append(
    element("p", `${overview.version} · ${number(overview.clips)} clips · ${number(overview.counts.frames)} frames`, "overview-total"),
    element("p", `コード対応: ${overview.schema} / 現物: 読み込み確認済み`, "muted"),
    element("p", "入力は1カメラの連続RGB（JPEG）。クリップの全フレームを保存し、ボール座標は保存画像の画素です。"),
    element("p", "教師は観測・補間・遮蔽推定・位置不明・画面外を区別する2Dボール注釈。学習・評価の既定方針はobserved-onlyです。用途はデータ品質の確認、検出器の学習、validation・test評価です。"),
    element("h2", "Source と split の内訳"),
    table(["Source", "clips", "frames", "train frames", "val frames", "test frames"], overview.sources.map((source) => [
      sourceLabels[source.id] || source.id, number(source.clips), number(source.counts.frames),
      ...["train", "val", "test"].map((split) => number(source.splits[split].frames)),
    ])),
    element("p", "splitはクリップ単位のランダム分割ではなく、TrackNetのgame・Meijiのvideo・Chat annotationの元動画単位です。train=学習、val=選定、test=保留評価。", "muted"),
    table(["split", "clips", "frames", "source groups"], Object.entries(overview.splits).map(([split, counts]) => [split, number(counts.clips), number(counts.frames), number(counts.groups)])),
    element("h2", "教師として使えるフレーム"),
    table(["状態", "frames"], ["scored_positive", "scored_negative", "reference", "unreviewed"].map((key) => [reviewLabels[key], number(overview.counts[key])])),
    element("p", "上の4区分は重複せず全フレームを覆います。以下はinstance数で、1フレームに複数instanceがある場合はframe数と一致しません。", "muted"),
    table(["point kind", "instances", "意味"], Object.entries(overview.point_counts).map(([key, count]) => [key, number(count), pointKinds[key]?.label || key])),
    element("h2", "派生playerデータ"),
  );
  for (const entry of overview.append_history || [])
    content.append(element("p", `同versionへの追加: ${number(entry.base_clips)} → ${number(entry.base_clips + entry.added_clips)} clips（${entry.at}）。元のclip IDとsplitを維持しています。`, "muted"));
  if (!playerDatasets.length) content.append(element("p", "対応するplayerデータは現物なし。ボール注釈の閲覧は利用できます。"));
  for (const player of playerDatasets) {
    content.append(element("p", `${player.label} · ${player.available ? `現物あり / 対応 ${number(player.clips)} clips / 採用 ${number(player.reviewed_clips)} / raw ${number(player.raw_clips)}` : `利用不可: ${player.error}`}`));
    if (player.available) {
      content.append(element("p", "入力はball storeのRGB、内容はモデル推定の2D pose・人物枠・クリップ内ID。ボールの教師とは別のレビュー用補助データです。対応範囲外の追加クリップにはposeを表示しません。", "muted"));
      content.append(element("p", `固定した元データ: ${player.ball_store_path}`, "overview-path"));
    }
  }
  content.append(element("p", `保存先: ${dataset.path}`, "overview-path"));
}
