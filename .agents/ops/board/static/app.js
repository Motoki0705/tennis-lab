"use strict";

const state = {
  cursor: null,
  posts: new Map(),
  roster: [],
  human: null,
  name: "",
  view: "timeline",
  selected: null,
  threadId: null,
  threadPosts: [],
  threadRequest: 0,
  busy: false,
  timer: null,
  refreshRoster: true,
  rosterError: null,
  toastTimer: null,
  forcePoll: false,
  reactionBusy: new Set(),
  actionError: null,
};
const labels = {
  info: "共有",
  question: "質問",
  request: "依頼",
  reply: "返信",
  pending: "通知待ち",
  held: "配送保留",
  notified: "通知受理",
  delivery_failed: "配送失敗",
  answered: "回答済み",
  accepted: "承諾",
  declined: "辞退",
  done: "完了",
  available: "Webで閲覧可能",
  idle: "待機中",
  working: "作業中",
  blocked: "確認待ち",
  unknown: "状態不明",
};
const emojis = {
  "+1": ["👍", "いいね"],
  eyes: ["👀", "見ています"],
  white_check_mark: ["✅", "チェック"],
  tada: ["🎉", "お祝い"],
  pray: ["🙏", "ありがとう"],
  question: ["❓", "質問"],
  warning: ["⚠️", "注意"],
};
const byId = (id) => document.getElementById(id);
const element = (tag, text = "", className = "") => {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
};
const button = (text, className, action) => {
  const node = element("button", text, className);
  node.type = "button";
  if (action) node.addEventListener("click", action);
  return node;
};
const formatTime = (value) =>
  value ? new Date(value).toLocaleString("ja-JP", { hour12: false }) : "—";
const clockTime = (value) =>
  new Date(value).toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
const waiting = (post) =>
  ["question", "request"].includes(post.kind) &&
  post.recipients.some((row) => row.response_state === null);
const problematic = (post) =>
  post.recipients.some((row) =>
    ["held", "delivery_failed"].includes(row.delivery_state),
  );
const sortedPosts = () =>
  [...state.posts.values()].sort((a, b) => a.id.localeCompare(b.id));
const nearBottom = (node) =>
  node.scrollHeight - node.scrollTop - node.clientHeight < 100;
const scrollBottom = (node) => {
  node.scrollTop = node.scrollHeight;
};

function avatar(name, presence) {
  const code =
    [...(name || "?")].reduce(
      (hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0,
      0,
    ) % 8;
  const node = element(
    "span",
    (name || "?")[0].toUpperCase(),
    `avatar ${name === "human" ? "color-human" : `color-${code}`}`,
  );
  node.setAttribute("aria-hidden", "true");
  if (presence) node.append(element("span", "", `presence ${presence}`));
  return node;
}

function party(person) {
  if (person.name === "human") return element("span", "human", "party");
  if (!person.repo || !person.parent)
    throw new Error("親のリポジトリまたはissue番号が不正です。");
  const node = element(
    "span",
    person.name ? `${person.name} · ` : "未割当 · ",
    "party",
  );
  const link = element("a", `#${person.parent}`);
  const [owner, repo] = person.repo.split("/");
  link.href = `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/issues/${encodeURIComponent(person.parent)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  node.append(link);
  return node;
}

function mentionBody(post) {
  const node = element("p", "", "body");
  const known = new Set([
    "all",
    "human",
    post.sender.name.toLowerCase(),
    ...post.recipients.map((p) => p.name.toLowerCase()),
    ...state.roster.filter((p) => p.name).map((p) => p.name.toLowerCase()),
  ]);
  const regex =
    /(?<![A-Za-z0-9_.+@-])@([A-Za-z]+)(?![A-Za-z0-9_@-]|\.[A-Za-z0-9])/g;
  const mentioned = new Set(
    [...post.body.matchAll(regex)].map((m) => m[1].toLowerCase()),
  );
  // Older CLI posts, channel posts and replies can have explicit recipients
  // without @ text. Project those frozen recipients into the message itself.
  const destinations = post.broadcast
    ? [
        "all",
        ...post.recipients.filter((p) => p.name === "human").map((p) => p.name),
      ]
    : post.recipients.map((p) => p.name);
  for (const name of destinations) {
    if (!mentioned.has(name.toLowerCase())) {
      const mention = element("span", `@${name}`, "mention recipient-mention");
      mention.title = "この投稿の宛先";
      node.append(mention, document.createTextNode(" "));
    }
  }
  let start = 0;
  for (const match of post.body.matchAll(regex)) {
    if (!known.has(match[1].toLowerCase())) continue;
    node.append(
      document.createTextNode(post.body.slice(start, match.index)),
      element("span", match[0], "mention"),
    );
    start = match.index + match[0].length;
  }
  node.append(document.createTextNode(post.body.slice(start)));
  return node;
}

function reactionControls(post) {
  const node = element("div", "", "reactions");
  for (const reaction of post.reactions) {
    const mine = reaction.reactors.some((person) => person.type === "human");
    const stamp = button(
      `${reaction.emoji} ${reaction.reactors.length}`,
      "reaction",
      () => react(post.id, reaction.code, mine),
    );
    stamp.setAttribute("aria-pressed", String(mine));
    stamp.setAttribute(
      "aria-label",
      `${emojis[reaction.code][1]} ${reaction.reactors.length}件${mine ? "（自分の反応を外す）" : "（反応する）"}`,
    );
    stamp.title = reaction.reactors
      .map((p) => (p.name === "human" ? "human" : `${p.name} #${p.parent}`))
      .join("、");
    stamp.disabled = state.reactionBusy.has(`${post.id}:${reaction.code}`);
    node.append(stamp);
  }
  const readers = post.recipients.filter((p) => p.read_at);
  if (readers.length) {
    const seen = element(
      "span",
      `👀 ${readers.length}`,
      "reaction read-receipt",
    );
    seen.title = `既読: ${readers.map((p) => p.name).join("、")}`;
    seen.setAttribute("aria-label", seen.title);
    seen.tabIndex = 0;
    node.append(seen);
  }
  return node;
}

function postActions(post) {
  const node = element("div", "", "post-actions");
  node.setAttribute("role", "toolbar");
  node.setAttribute("aria-label", "メッセージの操作");
  const picker = element("div", "", "reaction-picker");
  picker.hidden = true;
  picker.setAttribute("role", "group");
  picker.setAttribute("aria-label", "スタンプを選択");
  for (const [code, [emoji, label]] of Object.entries(emojis)) {
    const mine = post.reactions.some(
      (r) => r.code === code && r.reactors.some((p) => p.type === "human"),
    );
    const choice = button(emoji, "", () => {
      picker.hidden = true;
      react(post.id, code, mine);
    });
    choice.title = label;
    choice.setAttribute("aria-label", label);
    choice.setAttribute("aria-pressed", String(mine));
    picker.append(choice);
  }
  const add = button("☺", "reaction-add", () => {
    const open = picker.hidden;
    document.querySelectorAll(".reaction-picker").forEach((p) => {
      p.hidden = true;
    });
    picker.hidden = !open;
    add.setAttribute("aria-expanded", String(open));
    if (open) picker.firstElementChild.focus();
  });
  add.setAttribute("aria-label", "スタンプを追加");
  add.setAttribute("aria-expanded", "false");
  picker.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      picker.hidden = true;
      add.focus();
    }
  });
  node.append(add, picker);
  const reply = button("↩", "reply-action", () => selectThread(post.id));
  reply.setAttribute("aria-label", "スレッドで返信");
  reply.title = "スレッドで返信";
  node.append(reply);
  if (post.recipients.some((p) => p.name === "human" && !p.read_at)) {
    const read = button("👀", "read-action", () => markRead(post.id));
    read.setAttribute("aria-label", "既読にする");
    read.title = "既読にする";
    node.append(read);
  }
  return node;
}

function postCard(post, full = false) {
  const node = element(
    "article",
    "",
    `post${post.sender.name === "human" ? " human" : ""}`,
  );
  node.dataset.postId = post.id;
  node.tabIndex = 0;
  node.append(avatar(post.sender.name));
  const content = element("div", "", "post-content");
  const meta = element("div", "", "post-meta");
  const time = element("time", clockTime(post.created_at), "post-time");
  time.dateTime = post.created_at;
  time.title = formatTime(post.created_at);
  meta.append(party(post.sender), time);
  content.append(meta);
  if (full && post.in_reply_to) {
    const parent = state.threadPosts.find((p) => p.id === post.in_reply_to);
    content.append(
      button(
        `↳ ${parent ? parent.sender.name : "元の投稿"} への返信`,
        "reply-link",
        () => selectThread(post.in_reply_to),
      ),
    );
  }
  const bubble = element("div", "", "bubble");
  if (!post.in_reply_to && post.subject !== post.body.split("\n")[0])
    bubble.append(element("h3", post.subject));
  const body = mentionBody(post);
  if (!full) body.classList.add("preview");
  bubble.append(body);
  content.append(bubble);
  const reactions = reactionControls(post);
  if (reactions.children.length) content.append(reactions);
  const replies = sortedPosts().filter(
    (p) => p.thread_id === post.thread_id && p.id !== post.thread_id,
  );
  if (!full && replies.length) {
    const open = button(
      `${replies.length}件の返信 · ${replies[replies.length - 1].sender.name}`,
      "thread-link",
      () => selectThread(post.id),
    );
    open.dataset.openPost = post.id;
    content.append(open);
  }
  node.append(content, postActions(post));
  return node;
}

function renderRoster() {
  byId("roster-count").textContent = String(state.roster.length + 1);
  byId("roster-summary").textContent =
    `Herdrの親 ${state.roster.length} · human 1`;
  const card = (row) => {
    const node = button(
      "",
      `roster-button${state.name === row.name ? " active" : ""}`,
      () => setChannel(row.name),
    );
    node.setAttribute("aria-pressed", String(state.name === row.name));
    node.disabled = !row.name;
    node.title = `${row.identifier || "識別名なし"} · ${labels[row.state] || row.engine || "human"}\n最終投稿: ${formatTime(row.last_activity)}`;
    node.append(avatar(row.name, row.state));
    const info = element("span", "", "roster-info");
    info.append(
      element("span", row.name || "未割当", "roster-title"),
      element(
        "span",
        row.name === "human"
          ? "あなた · Webから参加"
          : `#${row.parent} · ${row.identifier || "識別名なし"}`,
        "roster-sub",
      ),
    );
    node.append(info);
    if (row.unread_count)
      node.append(element("span", row.unread_count, "count"));
    return node;
  };
  byId("roster").replaceChildren(...state.roster.map(card));
  byId("human-roster").replaceChildren(card(state.human || { name: "human" }));
  byId("all-channel").classList.toggle("active", !state.name);
}

function matches(post) {
  const direction = byId("direction").value;
  if (
    state.name &&
    !(
      (post.sender.name === state.name && direction !== "received") ||
      (post.recipients.some((p) => p.name === state.name) &&
        direction !== "sent")
    )
  )
    return false;
  const query = byId("search").value.trim().toLowerCase();
  return (
    !query ||
    [
      post.subject,
      post.body,
      post.sender.name,
      ...post.recipients.map((p) => p.name),
    ]
      .join(" ")
      .toLowerCase()
      .includes(query)
  );
}

function replaceMessages(container, children) {
  const opened = new Set(
    [...container.querySelectorAll(".reaction-picker:not([hidden])")].map(
      (node) => node.closest("article").dataset.postId,
    ),
  );
  const active = document.activeElement;
  const focusedPost =
    active && container.contains(active)
      ? active.closest("article")?.dataset.postId
      : null;
  const focusedLabel = focusedPost ? active.getAttribute("aria-label") : null;
  container.replaceChildren(...children);
  for (const post of container.querySelectorAll("article")) {
    if (opened.has(post.dataset.postId)) {
      post.querySelector(".reaction-picker").hidden = false;
      post.querySelector(".reaction-add").setAttribute("aria-expanded", "true");
    }
    if (post.dataset.postId === focusedPost) {
      const target = [...post.querySelectorAll("[aria-label]")].find(
        (node) => node.getAttribute("aria-label") === focusedLabel,
      );
      (target || post).focus({ preventScroll: true });
    }
  }
}

function renderFeed() {
  const posts = sortedPosts();
  const matching = posts.filter(matches);
  byId("timeline-count").textContent = String(
    posts.filter((p) => !p.in_reply_to).length,
  );
  byId("waiting-count").textContent = String(matching.filter(waiting).length);
  byId("issues-count").textContent = String(
    matching.filter(problematic).length,
  );
  const threads = new Set(matching.map((p) => p.thread_id));
  const visible =
    state.view === "timeline"
      ? posts.filter((p) => !p.in_reply_to && threads.has(p.id))
      : matching.filter(state.view === "waiting" ? waiting : problematic);
  const nodes = [];
  let previousDay = "";
  for (const post of visible) {
    const day = new Date(post.created_at).toLocaleDateString("ja-JP", {
      year: "numeric",
      month: "long",
      day: "numeric",
      weekday: "short",
    });
    if (day !== previousDay) {
      nodes.push(element("div", day, "day-separator"));
      previousDay = day;
    }
    nodes.push(postCard(post));
  }
  if (!nodes.length) {
    const empty = element("div", "", "empty");
    empty.append(
      element(
        "strong",
        state.view === "timeline"
          ? "ここから会話をはじめましょう"
          : "該当する投稿はありません",
      ),
      document.createTextNode(
        state.view === "timeline"
          ? "@名前で宛先を選び、メッセージを送れます。"
          : "絞り込みや検索条件を変えると、ほかの会話が表示されます。",
      ),
    );
    nodes.push(empty);
  }
  replaceMessages(byId("feed"), nodes);
  document
    .querySelectorAll("[data-view]")
    .forEach((node) =>
      node.setAttribute(
        "aria-pressed",
        String(node.dataset.view === state.view),
      ),
    );
}

function setView(view) {
  state.view = view;
  const executing = view === "execution";
  byId("execution-panel").hidden = !executing;
  byId("feed-scroll").hidden = executing;
  byId("feed-composer").hidden = executing;
  byId("new-message").hidden = executing;
  byId("direction-filter").hidden = executing;
  byId("search").disabled = executing;
  byId("workspace").classList.toggle("execution-open", executing);
  byId("workspace").classList.toggle(
    "thread-open",
    !executing && Boolean(state.selected),
  );
  byId("thread-pane").hidden = executing || !state.selected;
  byId("channel-title").textContent = executing
    ? "実行状況"
    : state.name
      ? `@${state.name}`
      : "# 全体";
  clearTimeout(state.timer);
  executionView.setActive(executing);
  renderFeed();
  if (!executing) poll();
}

function setChannel(name) {
  const composer = composers.compose;
  composer.drafts.set(state.name, {
    body: byId("compose-body").value,
    subject: byId("compose-subject").value,
    kind: byId("compose-kind").value,
  });
  const draft = composer.drafts.get(name || "") || {
    body: "",
    subject: "",
    kind: "info",
  };
  byId("compose-body").value = draft.body;
  resizeComposer(byId("compose-body"));
  byId("compose-subject").value = draft.subject;
  byId("compose-kind").value = draft.kind;
  byId("compose-suggestions").hidden = true;
  byId("compose-body").setAttribute("aria-expanded", "false");
  byId("compose-error").hidden = true;
  state.name = name || "";
  if (state.view === "execution") setView("timeline");
  byId("channel-title").textContent = state.name ? `@${state.name}` : "# 全体";
  byId("channel-description").textContent = state.name
    ? `${state.name} が参加する会話`
    : "親たちとhumanが集まる、共有の会話スペース";
  byId("compose-hint").textContent =
    state.name && state.name !== "human"
      ? `宛先 @${state.name} · @ で追加`
      : "@ で宛先を選択 · Shift + Enter で改行";
  renderRoster();
  renderFeed();
  scrollBottom(byId("feed-scroll"));
  byId("latest").hidden = true;
}

async function getJSON(url, timeout = 15000) {
  const response = await fetch(url, {
    cache: "no-store",
    credentials: "same-origin",
    ...(timeout === null ? {} : { signal: AbortSignal.timeout(timeout) }),
  });
  if (!response.ok)
    throw new Error(
      `データを取得できません（HTTP ${response.status}）。接続と掲示板の状態を確認してください。`,
    );
  return response.json();
}

async function writeJSON(url, method, payload) {
  let response;
  try {
    response = await fetch(url, {
      method,
      credentials: "same-origin",
      cache: "no-store",
      signal: AbortSignal.timeout(30000),
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": document.querySelector('meta[name="board-csrf-token"]')
          .content,
      },
      ...(payload === undefined ? {} : { body: JSON.stringify(payload) }),
    });
  } catch {
    throw new Error(
      "送信結果を確認できません。会話を更新して、投稿の有無を確かめてから再操作してください。",
    );
  }
  if (response.status === 403)
    throw new Error(
      "接続の期限が切れました。入力を控えてからページを再読み込みしてください。",
    );
  if (!response.ok)
    throw new Error(
      `操作できませんでした（HTTP ${response.status}）。宛先が有効か、本文に私的な識別子や絶対パスがないか確認してください。`,
    );
  return response.json();
}

function showError(error) {
  byId("error").textContent = error.message;
  byId("error").hidden = false;
}
function toast(message) {
  clearTimeout(state.toastTimer);
  byId("toast").textContent = message;
  byId("toast").hidden = false;
  state.toastTimer = setTimeout(() => {
    byId("toast").hidden = true;
  }, 4000);
}
function selectThread(postId) {
  if (location.hash === `#post=${postId}`) openThread(postId);
  else location.hash = `post=${postId}`;
}

function renderThread() {
  const nodes = [];
  state.threadPosts.forEach((post, index) => {
    if (index === 1)
      nodes.push(
        element(
          "div",
          `${state.threadPosts.length - 1}件の返信`,
          "thread-divider",
        ),
      );
    nodes.push(postCard(post, true));
  });
  replaceMessages(byId("thread"), nodes);
  byId("thread-count").textContent =
    `${Math.max(0, state.threadPosts.length - 1)}件の返信`;
  const target = state.threadPosts.find((p) => p.id === state.selected);
  byId("reply-target").textContent =
    target?.sender.name === "human"
      ? "自分の投稿への追記 · @で宛先を指定"
      : `${target?.sender.name || "投稿者"} への返信`;
}

async function openThread(postId) {
  const request = ++state.threadRequest;
  const switching = state.selected !== postId;
  if (switching) {
    // Keep a separate draft for each reply target; switching cannot send a
    // draft intended for a different person.
    composers.reply.drafts.set(state.selected, byId("reply-body").value);
  }
  if (switching) {
    byId("reply-body").value = composers.reply.drafts.get(postId) || "";
    byId("reply-error").hidden = true;
    byId("reply-suggestions").hidden = true;
    byId("reply-body").setAttribute("aria-expanded", "false");
  }
  state.selected = postId;
  byId("thread-pane").hidden = !postId || state.view === "execution";
  byId("workspace").classList.toggle(
    "thread-open",
    Boolean(postId) && state.view !== "execution",
  );
  resizeComposer(byId("reply-body"));
  if (!postId) {
    state.threadId = null;
    state.threadPosts = [];
    return;
  }
  if (switching) {
    state.threadId = null;
    state.threadPosts = [];
    byId("thread").replaceChildren(
      element("p", "スレッドを読み込んでいます…", "empty"),
    );
    byId("reply-send").disabled = true;
  }
  const bottom = switching || nearBottom(byId("thread"));
  try {
    const data = await getJSON(`/api/thread/${encodeURIComponent(postId)}`);
    if (request !== state.threadRequest || postId !== state.selected) return;
    state.threadId = data.thread_id;
    state.threadPosts = data.posts;
    renderThread();
    byId("reply-send").disabled = composers.reply.busy;
    if (bottom) scrollBottom(byId("thread"));
  } catch (error) {
    if (request !== state.threadRequest) return;
    showError(error);
  }
}

function applyPost(post) {
  state.posts.set(post.id, post);
  renderFeed();
  if (state.threadId === post.thread_id) {
    const index = state.threadPosts.findIndex((p) => p.id === post.id);
    if (index < 0) state.threadPosts.push(post);
    else state.threadPosts[index] = post;
    state.threadPosts.sort((a, b) => a.id.localeCompare(b.id));
    renderThread();
  }
}

async function react(postId, code, remove) {
  const key = `${postId}:${code}`;
  if (state.reactionBusy.has(key)) return;
  state.reactionBusy.add(key);
  try {
    const data = await writeJSON(
      `/api/posts/${postId}/reactions/${encodeURIComponent(code)}`,
      remove ? "DELETE" : "PUT",
    );
    applyPost(data.post);
    state.actionError = null;
  } catch (error) {
    state.actionError = error;
    showError(error);
  } finally {
    state.reactionBusy.delete(key);
    renderFeed();
    if (state.threadId) renderThread();
  }
}
async function markRead(postId) {
  try {
    applyPost((await writeJSON(`/api/posts/${postId}/read`, "POST")).post);
    requestPoll();
  } catch (error) {
    state.actionError = error;
    showError(error);
  }
}

const composers = {};
function resizeComposer(input) {
  input.style.height = "0px";
  input.style.height = `${input.scrollHeight}px`;
}
function setupComposer(prefix) {
  const form = byId(prefix),
    input = byId(`${prefix}-body`),
    list = byId(`${prefix}-suggestions`);
  const composer = {
    busy: false,
    candidates: [],
    index: 0,
    match: null,
    drafts: new Map(),
  };
  composers[prefix] = composer;
  const hide = () => {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  };
  const choose = (name) => {
    const match = composer.match;
    if (!match) return;
    const end = input.selectionStart;
    input.setRangeText(`@${name} `, match.index, end, "end");
    resizeComposer(input);
    hide();
    input.focus();
  };
  const suggest = () => {
    const before = input.value.slice(0, input.selectionStart);
    const match = /(?:^|[^A-Za-z0-9_.+@-])@([A-Za-z]*)$/.exec(before);
    if (!match) {
      hide();
      return;
    }
    composer.match = { index: before.length - match[1].length - 1 };
    const rows = [
      { name: "all", identifier: "稼働中の親全員" },
      ...state.roster.filter((p) => p.name),
    ];
    composer.candidates = rows.filter((p) =>
      p.name.toLowerCase().startsWith(match[1].toLowerCase()),
    );
    composer.index = 0;
    list.replaceChildren(
      ...composer.candidates.map((row, index) => {
        const option = button("", "suggestion", () => choose(row.name));
        option.id = `${prefix}-option-${index}`;
        option.setAttribute("role", "option");
        option.setAttribute("aria-selected", String(index === 0));
        option.append(avatar(row.name));
        const label = element("span", `@${row.name}`);
        label.append(
          element(
            "small",
            row.name === "all"
              ? row.identifier
              : `#${row.parent} · ${row.identifier || "識別名なし"}`,
          ),
        );
        option.append(label);
        return option;
      }),
    );
    list.hidden = !composer.candidates.length;
    input.setAttribute("aria-expanded", String(!list.hidden));
    if (!list.hidden)
      input.setAttribute("aria-activedescendant", `${prefix}-option-0`);
  };
  input.addEventListener("input", () => {
    resizeComposer(input);
    suggest();
  });
  let width = 0;
  new ResizeObserver((entries) => {
    const next = entries[0].contentRect.width;
    if (next !== width) {
      width = next;
      resizeComposer(input);
    }
  }).observe(input);
  resizeComposer(input);
  input.addEventListener("click", suggest);
  input.addEventListener("blur", () => setTimeout(hide, 150));
  input.addEventListener("keydown", (event) => {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Escape") {
      hide();
      return;
    }
    if (
      !list.hidden &&
      ["ArrowDown", "ArrowUp", "Enter", "Tab"].includes(event.key)
    ) {
      event.preventDefault();
      if (["Enter", "Tab"].includes(event.key))
        choose(composer.candidates[composer.index].name);
      else {
        composer.index =
          (composer.index +
            (event.key === "ArrowDown" ? 1 : -1) +
            composer.candidates.length) %
          composer.candidates.length;
        [...list.children].forEach((node, i) =>
          node.setAttribute("aria-selected", String(i === composer.index)),
        );
        input.setAttribute(
          "aria-activedescendant",
          `${prefix}-option-${composer.index}`,
        );
        list.children[composer.index].scrollIntoView({ block: "nearest" });
      }
    } else if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const reply = prefix === "reply";
    if (composer.busy || !input.value.trim() || (reply && !state.threadId))
      return;
    const body = input.value,
      target = state.selected,
      channel = state.name;
    const payload = reply
      ? { body }
      : {
          body,
          kind: byId("compose-kind").value,
          subject:
            byId("compose-subject").value.trim() ||
            body.trim().split("\n")[0].slice(0, 200),
          to: channel && channel !== "human" ? [channel] : [],
        };
    composer.busy = true;
    byId(`${prefix}-send`).disabled = true;
    input.readOnly = true;
    byId(`${prefix}-error`).hidden = true;
    hide();
    try {
      const data = await writeJSON(
        reply ? `/api/posts/${target}/replies` : "/api/posts",
        "POST",
        payload,
      );
      applyPost(data.post);
      composer.drafts.delete(reply ? target : channel);
      if (reply ? target === state.selected : channel === state.name) {
        input.value = "";
        resizeComposer(input);
        if (!reply) byId("compose-subject").value = "";
      }
      toast(reply ? "返信しました" : "投稿しました");
      requestPoll();
      if (reply && target === state.selected) scrollBottom(byId("thread"));
      if (!reply) {
        if (state.view !== "execution") state.view = "timeline";
        byId("search").value = "";
        renderFeed();
        scrollBottom(byId("feed-scroll"));
      }
    } catch (error) {
      byId(`${prefix}-error`).textContent = error.message;
      byId(`${prefix}-error`).hidden = false;
    } finally {
      composer.busy = false;
      byId(`${prefix}-send`).disabled = reply && !state.threadId;
      input.readOnly = false;
    }
  });
}

function requestPoll() {
  state.forcePoll = true;
  poll();
}
// Herdr is read on the first visible display and then only by the button.
let runtimeShown = false;
function showRuntimeOnce() {
  if (runtimeShown || document.visibilityState !== "visible") return;
  runtimeShown = true;
  refreshRuntime();
}
async function refreshRuntime() {
  byId("refresh-runtime").disabled = true;
  try {
    const data = await getJSON("/api/runtime");
    const cards = data.agents.map((row) => {
      const card = element("div", "", "runtime-card");
      card.title = `${row.label} · ${labels[row.state] || row.state}\n${row.title}`;
      card.append(avatar(row.engine, row.state));
      const info = element("div", "", "runtime-info");
      info.append(
        element("strong", `${row.label} · ${labels[row.state] || row.state}`),
        element("span", row.title_status === "redacted"
          ? "タイトル非表示（ローカル情報を含むため）" : row.title || "タイトルなし"),
      );
      card.append(info);
      return card;
    });
    byId("runtime-agents").replaceChildren(...(cards.length ? cards : [element("p", "稼働中のエージェントはいません。") ]));
    const breakdown = ["working", "blocked", "idle", "done", "unknown"]
      .map((name) => [name, data.agents.filter((row) => row.state === name).length])
      .filter(([, count]) => count)
      .map(([name, count]) => `${labels[name]} ${count}`)
      .join(" · ");
    byId("runtime-observed").textContent = `${data.agents.length}エージェント${breakdown ? `（${breakdown}）` : ""} · ${formatTime(data.observed_at)} 取得 · Herdr全体（掲示板への参加登録とは別）`;
    byId("runtime-error").hidden = true;
  } catch (error) {
    byId("runtime-error").textContent = "取得できませんでした（前回の表示を保持）。「状態を更新」で再試行してください。";
    byId("runtime-error").hidden = false;
  } finally {
    byId("refresh-runtime").disabled = false;
  }
}
async function poll() {
  clearTimeout(state.timer);
  if (
    state.busy ||
    document.visibilityState !== "visible" ||
    state.view === "execution"
  )
    return;
  state.busy = true;
  state.forcePoll = false;
  const refreshRoster = state.refreshRoster;
  state.refreshRoster = false;
  byId("refresh-roster").disabled = refreshRoster;
  try {
    const query = new URLSearchParams();
    if (state.cursor) query.set("cursor", state.cursor);
    if (refreshRoster) query.set("refresh_roster", "true");
    const data = await getJSON(`/api/changes?${query}`);
    const bottom = !state.cursor || nearBottom(byId("feed-scroll"));
    if (data.reset) state.posts.clear();
    data.posts.forEach((post) => state.posts.set(post.id, post));
    state.cursor = data.cursor;
    state.human = data.human;
    if (data.agents) state.roster = data.agents;
    renderRoster();
    byId("roster-observed").textContent =
      `名簿更新 ${formatTime(data.roster_observed_at)}`;
    if (data.reset || data.posts.length) {
      renderFeed();
      if (bottom) scrollBottom(byId("feed-scroll"));
      else if (data.posts.length) byId("latest").hidden = false;
    }
    byId("connection").textContent = `接続中 · ${clockTime(new Date())} 更新`;
    if (refreshRoster) state.rosterError = null;
    if (state.rosterError) showError(state.rosterError);
    else if (state.actionError) showError(state.actionError);
    else byId("error").hidden = true;
    if (
      state.selected &&
      state.view !== "execution" &&
      (data.reset ||
        !state.threadId ||
        data.changed_threads.includes(state.threadId))
    )
      await openThread(state.selected);
  } catch (error) {
    if (refreshRoster) state.rosterError = error;
    showError(error);
    byId("connection").textContent = "更新に失敗 · 再接続を待っています";
  } finally {
    state.busy = false;
    byId("refresh-roster").disabled = state.refreshRoster;
    if (document.visibilityState === "visible" && state.view !== "execution")
      state.timer = setTimeout(
        poll,
        state.refreshRoster || state.forcePoll ? 0 : 3000,
      );
  }
}

const executionView = new ExecutionView();
setupComposer("compose");
setupComposer("reply");
byId("refresh-runtime").addEventListener("click", refreshRuntime);
// The card list opens on demand so the conversation keeps its height.
byId("toggle-runtime").addEventListener("click", () => {
  const list = byId("runtime-agents");
  list.hidden = !list.hidden;
  byId("toggle-runtime").setAttribute("aria-expanded", String(!list.hidden));
  byId("toggle-runtime").textContent = list.hidden ? "一覧を表示" : "一覧を閉じる";
});
showRuntimeOnce();
byId("compose-options").addEventListener("click", () => {
  const settings = byId("compose-settings");
  settings.hidden = !settings.hidden;
  byId("compose-options").setAttribute(
    "aria-expanded",
    String(!settings.hidden),
  );
});
byId("all-channel").addEventListener("click", () => setChannel(""));
byId("close-thread").addEventListener("click", () => {
  location.hash = "";
});
byId("refresh-roster").addEventListener("click", () => {
  state.refreshRoster = true;
  byId("refresh-roster").disabled = true;
  poll();
});
byId("new-message").addEventListener("click", () =>
  byId("compose-body").focus(),
);
byId("latest").addEventListener("click", () => {
  scrollBottom(byId("feed-scroll"));
  byId("latest").hidden = true;
});
byId("feed-scroll").addEventListener("scroll", () => {
  if (nearBottom(byId("feed-scroll"))) byId("latest").hidden = true;
});
byId("search").addEventListener("input", renderFeed);
byId("direction").addEventListener("change", renderFeed);
document.querySelectorAll("[data-view]").forEach((node) =>
  node.addEventListener("click", () => {
    setView(node.dataset.view);
  }),
);
document.addEventListener("keydown", (event) => {
  if (
    event.key === "/" &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)
  ) {
    event.preventDefault();
    byId("search").focus();
  }
});
window.addEventListener("hashchange", () => {
  const match = /^#post=([0-9a-f]{32})$/.exec(location.hash);
  openThread(match ? match[1] : null);
});
document.addEventListener("visibilitychange", () => {
  clearTimeout(state.timer);
  showRuntimeOnce();
  executionView.setActive(state.view === "execution");
  if (document.visibilityState === "visible" && state.view !== "execution")
    poll();
});
poll();
const initialThread = /^#post=([0-9a-f]{32})$/.exec(location.hash);
if (initialThread) openThread(initialThread[1]);
