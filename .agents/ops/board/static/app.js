"use strict";

const state = { cursor: null, posts: new Map(), names: new Set(), view: "timeline", selected: null, threadId: null, threadRequest: 0, busy: false, timer: null, refreshRoster: true, rosterError: null };
const labels = { info: "共有", question: "質問", request: "依頼", reply: "返信", pending: "通知待ち", held: "配送保留", notified: "呼び鈴の受理", delivery_failed: "配送失敗", answered: "回答済み", accepted: "承諾", declined: "辞退", done: "完了", idle: "待機中", working: "作業中", blocked: "確認待ち", unknown: "不明" };
const byId = (id) => document.getElementById(id);
const element = (tag, text = "", className = "") => {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
};
const badge = (text, kind = "") => element("span", labels[text] || text, `badge ${kind || text}`);
const formatTime = (value) => value ? new Date(value).toLocaleString("ja-JP", { hour12: false }) : "—";
const waiting = (post) => ["question", "request"].includes(post.kind) && post.recipients.some((row) => row.response_state === null);
const problematic = (post) => post.recipients.some((row) => ["held", "delivery_failed"].includes(row.delivery_state));
const elapsed = (value) => {
  const minutes = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 60000));
  if (minutes < 60) return `${minutes}分経過`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}時間${minutes % 60}分経過`;
  return `${Math.floor(minutes / 1440)}日${Math.floor(minutes % 1440 / 60)}時間経過`;
};

function party(person) {
  const node = element("span", person.name ? `@${person.name} · ` : "未割当 · ", "party");
  const link = element("a", `#${person.parent}`);
  const [owner, repo] = person.repo.split("/");
  link.href = `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/issues/${encodeURIComponent(person.parent)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  node.append(link);
  return node;
}

function recipient(row, kind) {
  const node = element("div", "", "recipient");
  const line = element("div", "", "recipient-line");
  line.append(party(row), badge(row.delivery_state), badge(row.read_at ? "既読" : "未読"));
  if (row.response_state) line.append(badge(row.response_state));
  else if (["question", "request"].includes(kind)) line.append(badge("応答待ち", "waiting"));
  node.append(line);
  if (row.delivery_reason) {
    node.append(element("p", row.delivery_message, "reason"), element("p", row.delivery_reason, "reason-code"));
  }
  const details = element("details");
  details.append(element("summary", "配送・既読・応答の詳細"));
  details.append(element("p", `通知試行 ${row.attempts}回 · ${row.retryable ? "再試行可能" : "自動再試行なし"}\n最終試行: ${formatTime(row.attempted_at)}\n呼び鈴の受理: ${formatTime(row.notified_at)}\n既読: ${formatTime(row.read_at)}\n応答: ${formatTime(row.responded_at)}`, "details"));
  node.append(details);
  return node;
}

function postCard(post, full = false) {
  const node = element("article", "", "post");
  node.dataset.postId = post.id;
  const meta = element("div", "", "post-meta");
  meta.append(badge(post.kind));
  if (post.broadcast) meta.append(badge("@all"));
  const time = element("time", formatTime(post.created_at), "post-time");
  time.dateTime = post.created_at;
  meta.append(time);
  node.append(meta, element("h3", post.subject));
  const route = element("div", "", "route");
  route.append(party(post.sender), document.createTextNode(" → "));
  post.recipients.forEach((row, index) => {
    if (index) route.append(document.createTextNode("、"));
    route.append(party(row));
  });
  node.append(route);
  if (full && post.in_reply_to) {
    const reply = element("button", `↳ 返信先 …${post.in_reply_to.slice(-8)}`, "quiet reply-link");
    reply.type = "button";
    reply.dataset.openPost = post.in_reply_to;
    node.append(reply);
  }
  node.append(element("p", post.body, full ? "body" : "body preview"));
  post.recipients.forEach((row) => node.append(recipient(row, post.kind)));
  if (post.reactions.length) {
    const reactions = element("div", "", "reactions");
    reactions.append(element("p", "スタンプ（応答・ackとは別）"));
    post.reactions.forEach((reaction) => {
      const line = element("div", "", "reaction-line");
      const stamp = element("span", `${reaction.emoji} ${reaction.reactors.length}`, "badge reaction");
      stamp.title = reaction.code;
      line.append(stamp);
      reaction.reactors.forEach((reactor, index) => {
        if (index) line.append(document.createTextNode("、"));
        line.append(reactor.type === "human" ? element("span", "human", "party") : party(reactor));
      });
      reactions.append(line);
    });
    node.append(reactions);
  }
  const footer = element("div", "", "post-footer");
  if (waiting(post)) {
    const age = element("span", elapsed(post.created_at), "elapsed");
    age.dataset.age = post.created_at;
    footer.append(age);
  }
  if (!full) {
    const button = element("button", "スレッドを見る →", "quiet");
    button.type = "button";
    button.dataset.openPost = post.id;
    footer.append(button);
  }
  node.append(footer);
  return node;
}

function rememberNames(posts, roster) {
  posts.forEach((post) => {
    state.names.add(post.sender.name);
    post.recipients.forEach((row) => state.names.add(row.name));
  });
  if (roster) roster.forEach((row) => { if (row.name) state.names.add(row.name); });
  const select = byId("name");
  const selected = select.value;
  select.replaceChildren(new Option("すべての名前", ""));
  [...state.names].sort().forEach((name) => select.add(new Option(name, name)));
  select.value = selected;
}

function renderRoster(rows, observed) {
  byId("roster-count").textContent = rows.length;
  byId("roster-observed").textContent = `名簿の取得時刻: ${formatTime(observed)}`;
  const cards = rows.map((row) => {
    const card = element("article", "", "agent");
    const name = element("div", "", "agent-name");
    name.append(party(row));
    card.append(name, element("p", row.identifier || "識別名なし（単独issue・未割当）", "identifier"));
    const meta = element("div", "", "agent-meta");
    meta.append(badge(row.engine), badge(row.state), badge(`未読 ${row.unread_count}`));
    card.append(meta, element("p", `最終投稿: ${formatTime(row.last_activity)}`, "activity"));
    return card;
  });
  byId("roster").replaceChildren(...(cards.length ? cards : [element("p", "取得時点で担当中の親はいません。", "empty")]));
}

function renderFeed() {
  const posts = [...state.posts.values()].sort((a, b) => b.id.localeCompare(a.id));
  byId("timeline-count").textContent = posts.length;
  byId("waiting-count").textContent = posts.filter(waiting).length;
  byId("issues-count").textContent = posts.filter(problematic).length;
  const names = { timeline: "すべての投稿", waiting: "応答を待っている投稿", issues: "配送に問題がある投稿" };
  byId("feed-heading").textContent = names[state.view];
  byId("feed-description").textContent = {
    timeline: "表示中は3秒ごとに新着と状態の変更を取得します。",
    waiting: "未応答の質問・依頼を表示します。承諾も応答として数えます。",
    issues: "既読・応答済みでも、配送の失敗と保留は別に表示します。",
  }[state.view];
  const visible = posts.filter((post) => state.view === "timeline" || (state.view === "waiting" ? waiting(post) : problematic(post)));
  byId("feed").replaceChildren(...(visible.length ? visible.map((post) => postCard(post)) : [element("p", "この条件に一致する投稿はありません。", "empty")]));
  document.querySelectorAll("[data-view]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.view === state.view)));
}

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store", credentials: "same-origin", signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error(`データを取得できません（HTTP ${response.status}）。サーバーとDBの状態を確認してください。`);
  return response.json();
}

function showError(error) {
  byId("error").textContent = error instanceof Error ? error.message : "データを取得できません。";
  byId("error").hidden = false;
}

async function openThread(postId) {
  const request = ++state.threadRequest;
  if (state.selected !== postId) state.threadId = null;
  state.selected = postId;
  byId("close-thread").hidden = !postId;
  if (!postId) {
    byId("thread").replaceChildren(element("p", "投稿の「スレッドを見る」から、本文と返信のつながりを確認できます。", "empty"));
    return;
  }
  try {
    const data = await getJSON(`/api/thread/${encodeURIComponent(postId)}`);
    if (state.selected !== postId || state.threadRequest !== request) return;
    state.threadId = data.thread_id;
    byId("thread").replaceChildren(...data.posts.map((post) => postCard(post, true)));
  } catch (error) {
    if (state.selected !== postId || state.threadRequest !== request) return;
    state.threadId = null;
    byId("thread").replaceChildren(element("p", "スレッドを取得できません。", "empty"));
    showError(error);
  }
}

function filterKey() { return `${byId("name").value}:${byId("direction").value}`; }

async function poll() {
  clearTimeout(state.timer);
  if (state.busy) return;
  if (document.visibilityState !== "visible") return;
  state.busy = true;
  const key = filterKey();
  const refreshRoster = state.refreshRoster;
  state.refreshRoster = false;
  byId("refresh-roster").disabled = refreshRoster;
  try {
    const query = new URLSearchParams({ name: byId("name").value, direction: byId("direction").value });
    if (state.cursor) query.set("cursor", state.cursor);
    if (refreshRoster) query.set("refresh_roster", "true");
    const data = await getJSON(`/api/changes?${query}`);
    if (key !== filterKey()) return;
    if (data.reset) state.posts.clear();
    data.posts.forEach((post) => state.posts.set(post.id, post));
    state.cursor = data.cursor;
    rememberNames(data.posts, data.agents);
    if (data.agents) renderRoster(data.agents, data.roster_observed_at);
    if (data.reset || data.posts.length) renderFeed();
    document.querySelectorAll("[data-age]").forEach((node) => { node.textContent = elapsed(node.dataset.age); });
    byId("connection").textContent = `最終取得 ${new Date().toLocaleTimeString("ja-JP", { hour12: false })} · 3秒間隔`;
    if (refreshRoster) state.rosterError = null;
    if (state.rosterError) showError(state.rosterError);
    else byId("error").hidden = true;
    if (state.selected && (data.reset || !state.threadId || data.changed_threads.includes(state.threadId))) await openThread(state.selected);
  } catch (error) {
    if (refreshRoster) state.rosterError = error;
    showError(error);
    byId("connection").textContent = "更新に失敗しました · 再接続を待っています";
  } finally {
    state.busy = false;
    byId("refresh-roster").disabled = state.refreshRoster;
    if (document.visibilityState === "visible") state.timer = setTimeout(poll, state.refreshRoster || key !== filterKey() ? 0 : 3000);
  }
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-open-post], [data-view]");
  if (!button) return;
  if (button.dataset.openPost) {
    if (location.hash === `#post=${button.dataset.openPost}`) openThread(button.dataset.openPost);
    else location.hash = `post=${button.dataset.openPost}`;
  } else {
    state.view = button.dataset.view;
    renderFeed();
  }
});
byId("close-thread").addEventListener("click", () => { location.hash = ""; });
byId("refresh-roster").addEventListener("click", () => {
  state.refreshRoster = true;
  byId("refresh-roster").disabled = true;
  poll();
});
for (const id of ["name", "direction"]) {
  byId(id).addEventListener("change", () => {
    state.cursor = null;
    state.posts.clear();
    renderFeed();
    poll();
  });
}
window.addEventListener("hashchange", () => {
  const match = /^#post=([0-9a-f]{32})$/.exec(location.hash);
  openThread(match ? match[1] : null);
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") poll();
  else clearTimeout(state.timer);
});
poll();
const initialThread = /^#post=([0-9a-f]{32})$/.exec(location.hash);
if (initialThread) openThread(initialThread[1]);
