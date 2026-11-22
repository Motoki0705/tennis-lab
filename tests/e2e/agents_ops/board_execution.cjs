"use strict";

// Production scripts, in-memory DOM and a manually advanced timer queue. No
// elapsed-time sleeps, browser packages, remote services or agent processes.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.text = "";
    this.value = "";
    this.hidden = false;
    this.style = {};
    this.events = {};
    this.classList = { toggle() {}, add() {} };
  }
  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }
  get textContent() {
    return this.text + this.children.map((n) => n.textContent).join("");
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  replaceChildren(...nodes) {
    this.text = "";
    this.children = nodes;
  }
  addEventListener(name, callback) {
    this.events[name] = callback;
  }
  setAttribute(name, value) {
    this[name] = value;
  }
  removeAttribute(name) {
    delete this[name];
  }
  querySelectorAll() {
    return [];
  }
  contains() {
    return false;
  }
}
const elements = new Map();
const tabs = ["timeline", "execution", "waiting", "issues"].map((view) => {
  const node = new Element("button");
  node.dataset.view = view;
  return node;
});
const document = {
  visibilityState: "hidden",
  events: {},
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => {
    const node = new Element("#text");
    node.textContent = text;
    return node;
  },
  getElementById: (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  querySelectorAll: (query) => (query === "[data-view]" ? tabs : []),
  querySelector: () => ({ content: "test-csrf" }),
  addEventListener(name, callback) {
    this.events[name] = callback;
  },
};
const timers = new Map(),
  requests = [];
let serial = 0,
  fail = false;
let holdRefresh = false,
  finishRefresh = null,
  cacheAvailable = false,
  herdrReads = 0;
let serverRefresh = structuredClone(input.local.github_refresh);
const response = (body, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => structuredClone(body),
});
const localData = () => ({
  ...structuredClone(input.local),
  github_refresh: { ...serverRefresh },
});
const completeRefresh = () => {
  serverRefresh = {
    epoch: serverRefresh.epoch,
    status: "idle",
    revision: serverRefresh.revision + 1,
    result: fail ? "failed" : "succeeded",
  };
  if (!fail && !cacheAvailable) {
    input.local = structuredClone(input.refreshed);
    cacheAvailable = true;
  }
  return response(
    {
      ...structuredClone(input.refreshed),
      github_refresh: { ...serverRefresh },
    },
    fail ? 503 : 200,
  );
};
const context = vm.createContext({
  pendingSubmission: null,
  post: input.post,
  document,
  URLSearchParams,
  AbortSignal,
  window: { addEventListener() {} },
  location: { hash: "" },
  ResizeObserver: class {
    observe() {}
  },
  setTimeout(callback, delay) {
    const id = ++serial;
    timers.set(id, { callback, delay });
    return id;
  },
  clearTimeout(id) {
    timers.delete(id);
  },
  fetch: async (url) => {
    requests.push(url);
    if (url === "/api/runtime")
      return response({
        agents: [
          {
            label: "codex 1",
            engine: "codex",
            state: "working",
            title: "t",
            title_status: "available",
          },
          {
            label: "claude 1",
            engine: "claude",
            state: "idle",
            title: "",
            title_status: "redacted",
          },
          {
            label: "agy 1",
            engine: "agy",
            state: "working",
            title: "u",
            title_status: "available",
          },
        ],
        observed_at: input.local.observed_at,
      });
    if (url.startsWith("/api/execution")) {
      if (url.includes("refresh=true")) {
        if (serverRefresh.status === "running")
          return response(localData(), 202);
        serverRefresh = {
          ...serverRefresh,
          status: "running",
          revision: serverRefresh.revision + 1,
        };
        if (holdRefresh)
          return new Promise((resolve) => {
            finishRefresh = () => resolve(completeRefresh());
          });
        return completeRefresh();
      }
      if (url.includes("herdr=true")) herdrReads += 1;
      return response(localData(), fail ? 503 : 200);
    }
    return {
      ok: !fail,
      status: fail ? 503 : 200,
      json: async () => {
        return {
          reset: true,
          cursor: "cursor",
          posts: [],
          agents: [],
          human: { name: "human", unread_count: 0 },
          changed_threads: [],
          roster_observed_at: input.local.observed_at,
        };
      },
    };
  },
});
// index.html starts the runtime card list closed.
document.getElementById("runtime-agents").hidden = true;
for (const name of ["execution.js", "app.js"])
  vm.runInContext(
    fs.readFileSync(path.join(process.argv[2], name), "utf8"),
    context,
  );
const settle = async () => {
  await new Promise(setImmediate);
  await new Promise(setImmediate);
};
const descendants = (node) => [node, ...node.children.flatMap(descendants)];
const view = (name) =>
  tabs.find((node) => node.dataset.view === name).events.click();
const executionRequests = () =>
  requests.filter((url) => url.startsWith("/api/execution"));
const runtimeRequests = () => requests.filter((url) => url === "/api/runtime");
const tick = async (delay = 10000) => {
  const [id, timer] = [...timers].find(([, timer]) => timer.delay === delay);
  timers.delete(id);
  timer.callback();
  await settle();
};
const graph = () => elements.get("execution-graph");
const drawer = () => elements.get("execution-drawer-content");
const nodeButton = (key) =>
  descendants(graph()).find(
    (node) => node.tagName === "BUTTON" && node.dataset.node === key,
  );
const foldButton = (key) =>
  descendants(graph()).find((node) => node.dataset.fold === key);
const graphText = () => graph().textContent;

async function main() {
  const parent = input.local.parents[0];
  const session = input.local.sessions.find((row) => row.title === "CI確認");
  parent.summary.messages = [{ status: "visible", text: "親の途中報告" }];
  session.summary.messages = [{ status: "visible", text: "未登録の途中報告" }];
  assert.equal(requests.length, 0, "hidden pages do not poll");
  document.visibilityState = "visible";
  view("execution");
  await settle();
  assert.deepEqual(
    executionRequests(),
    ["/api/execution?herdr=true"],
    "the first display reads Herdr once",
  );
  assert.deepEqual(
    [...timers.values()].map((t) => t.delay),
    [10000],
  );
  assert.equal(elements.get("feed-composer").hidden, true);
  assert.equal(elements.get("execution-panel").hidden, false);
  assert.equal(tabs[1]["aria-pressed"], "true");
  const parentKey = `parent:${parent.repo}#${parent.parent}`;
  const childKey = (number) =>
    `child:${parent.repo}#${parent.parent}/${number}`;
  const rowKeys = graph().children.map((row) => row.dataset.row);
  // Running child, working parent and working unregistered session precede idle/done.
  assert.deepEqual(
    rowKeys.slice(0, 3),
    [parentKey, "parent:o/r#2", `session:${session.key}`],
  );
  assert.ok(rowKeys.includes(`session:${input.local.sessions.find((row) => row.state === "done").key}`));
  assert.ok(!rowKeys.includes("parent:o/r#5"), "absent parent is hidden");
  assert.ok(!rowKeys.includes("parent:o/r#3"), "an unobserved generation is hidden after Herdr was read");
  assert.equal(vm.runInContext("executionView.visibleParents().some(parent => parent.parent === 3)", context), false);
  const handoffSession = input.local.sessions.find((row) => row.title === "引継ぎ後の観測待ち");
  assert.ok(nodeButton(`session:${handoffSession.key}`));
  // Joined parent: Herdr terminal title, then the board name and number.
  const parentText = nodeButton(parentKey).textContent;
  assert.ok(parentText.includes("子の進行管理"), parentText);
  assert.ok(parentText.includes(`${parent.roster.name} · #1`), parentText);
  const children = `${parentKey}|children`;
  const undelegated = `${parentKey}|undelegated`;
  assert.equal(foldButton(children).textContent, "+7 子worker（実行中 1）");
  assert.equal(foldButton(children)["aria-expanded"], "false");
  assert.equal(nodeButton(childKey(11)), undefined);
  assert.equal(nodeButton(childKey(17)), undefined);
  foldButton(undelegated).events.click();
  assert.ok(nodeButton(childKey(17)));
  assert.equal(nodeButton(childKey(11)), undefined);
  foldButton(children).events.click();
  for (const label of [
    "#11 子11の作業",
    "gpt-6.1-sol · max",
    "#12 （題名未取得）",
    "gpt-6.1-sol（指定値） · max（指定値）",
    "未起動",
    "実行中",
    "ターン終了",
    "失敗",
    "停止中",
    "依存待ち",
    "委任予約済み・未起動",
    "状態不明",
    "更新途絶",
    "調査メモ",
    "CI確認",
    "未登録セッション",
  ])
    assert.ok(graphText().includes(label), label);
  assert.ok(!graphText().includes("undefined"));
  assert.ok(graphText().includes("#17 子17の作業"));
  assert.equal(foldButton(undelegated)["aria-expanded"], "true");
  assert.equal(
    descendants(nodeButton(childKey(13))).find((node) => node.tagName === "IMG").src,
    "/static/icon-claude.svg",
  );
  assert.equal(elements.get("execution-drawer").hidden, true);
  // Every node opens the same drawer; previous card details live there.
  nodeButton(childKey(11)).events.click();
  assert.equal(elements.get("execution-drawer").hidden, false);
  assert.equal(nodeButton(childKey(11))["aria-pressed"], "true");
  assert.equal(
    elements.get("execution-drawer-heading").textContent,
    "#11 子11の作業",
  );
  for (const label of [
    "Issue 未取得",
    "codex · gpt-6.1-sol · max",
    "停止とは限りません",
    "非表示（秘匿検査）",
    "読取上限",
    "<img src=x onerror=alert(1)> を文字として表示",
  ])
    assert.ok(drawer().textContent.includes(label), label);
  const steps = descendants(drawer()).find((node) => node.className === "execution-steps");
  assert.equal(steps.tagName, "OL");
  assert.deepEqual(
    steps.children.map((step) => step.children.at(-1).textContent),
    ["先に入力を確認します", "非表示（秘匿検査）", "<img src=x onerror=alert(1)> を文字として表示"],
  );
  for (const removed of ["最後の発言", "直近のツール", "取得元・判定根拠", "委任・run"])
    assert.ok(!drawer().textContent.includes(removed));
  assert.equal(
    descendants(drawer()).filter((node) => node.tagName === "IMG").length,
    0,
  );
  nodeButton(childKey(14)).events.click();
  assert.ok(drawer().textContent.includes("親が差分を確認する"));
  nodeButton(childKey(12)).events.click();
  assert.ok(drawer().textContent.includes("gpt-6.1-sol（指定値） · max（指定値）"));
  nodeButton(parentKey).events.click();
  assert.equal(elements.get("execution-drawer-heading").textContent, "子の進行管理");
  assert.ok(drawer().textContent.includes("親の途中報告"));
  assert.ok(!drawer().textContent.includes("#17"));
  nodeButton(`session:${session.key}`).events.click();
  assert.equal(elements.get("execution-drawer-heading").textContent, "CI確認");
  assert.ok(drawer().textContent.includes("未登録の途中報告"));
  elements.get("close-execution-drawer").events.click();
  assert.equal(elements.get("execution-drawer").hidden, true);
  // Selection, opened folds and the drawer itself survive the 10s refresh.
  // A new local timestamp alone does not rebuild the drawer.
  nodeButton(childKey(11)).events.click();
  const opened = drawer().children[0];
  input.local.observed_at = "2026-10-30T12:00:10+00:00";
  input.local.parents[0].children[0].observed_at = input.local.observed_at;
  await tick();
  assert.deepEqual(executionRequests().slice(-1), ["/api/execution"]);
  assert.equal(herdrReads, 1, "local polls never read Herdr");
  assert.equal(drawer().children[0], opened);
  assert.equal(nodeButton(childKey(11))["aria-pressed"], "true");
  assert.ok(graphText().includes("#17 子17の作業"));
  input.local.parents[0].children[0].summary.messages.push({ status: "visible", text: "新しい発言" });
  await tick();
  assert.notEqual(drawer().children[0], opened);
  assert.ok(drawer().textContent.includes("新しい発言"));
  const tree = graph();
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(
    executionRequests().filter((url) => url.includes("refresh=true")).length,
    1,
  );
  assert.ok(drawer().textContent.includes("Issue 未完了（OPEN）"));
  assert.ok(drawer().textContent.includes("PR #88"));
  nodeButton(childKey(14)).events.click();
  assert.ok(drawer().textContent.includes("GitHubの停止記録"));
  assert.ok(!drawer().textContent.includes("ローカルの停止記録"));
  // Completion comes from GitHub only: merged PR (#12) and closed Issue (#17).
  assert.ok(graphText().includes("+2 完了"), graphText());
  assert.ok(nodeButton(childKey(19)));
  assert.ok(!graphText().includes("#12"));
  const done = `${parentKey}|done`;
  foldButton(done).events.click();
  assert.ok(graphText().includes("#12 GitHub上の題名12"));
  assert.ok(graphText().includes("完了（GitHub）"));
  nodeButton(childKey(17)).events.click();
  assert.ok(drawer().textContent.includes("Issue 完了（CLOSED）"));
  foldButton(children).events.click();
  assert.equal(nodeButton(childKey(11)), undefined);
  assert.ok(nodeButton(childKey(19)) && nodeButton(childKey(12)));
  foldButton(children).events.click();
  holdRefresh = true;
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(elements.get("refresh-execution").disabled, true);
  assert.equal(elements.get("execution-refresh").hidden, false);
  assert.ok(
    elements
      .get("execution-refresh")
      .textContent.includes("GitHub・Herdr取得中"),
  );
  const pendingCount = executionRequests().length;
  const refreshCount = executionRequests().filter((url) =>
    url.includes("refresh=true"),
  ).length;
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(
    executionRequests().length,
    pendingCount,
    "pending refresh cannot be submitted twice",
  );
  input.local.parents[0].children[0].label = "実行中（新しい観測）";
  for (let ticks = 1; ticks <= 3; ticks++) {
    await tick();
    assert.equal(
      executionRequests().length,
      pendingCount + ticks,
      "10s local polling continues during the long refresh",
    );
    assert.equal(elements.get("refresh-execution").disabled, true);
    assert.ok(tree.textContent.includes("実行中（新しい観測）"));
  }
  assert.equal(
    executionRequests().filter((url) => url.includes("refresh=true")).length,
    refreshCount,
  );
  finishRefresh();
  await settle();
  holdRefresh = false;
  assert.equal(elements.get("refresh-execution").disabled, false);
  assert.equal(elements.get("execution-refresh").hidden, true);
  assert.ok(
    tree.textContent.includes("実行中（新しい観測）"),
    "old local data in the GitHub response never rolls back the display",
  );

  // Another browser already started the refresh: HTTP 202 supplies progress,
  // and subsequent local observations release the button on completion.
  serverRefresh = {
    ...serverRefresh,
    status: "running",
    revision: serverRefresh.revision + 1,
  };
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(elements.get("refresh-execution").disabled, true);
  assert.equal(elements.get("execution-refresh").hidden, false);
  context.staleRefresh = {
    epoch: serverRefresh.epoch,
    status: "idle",
    revision: 0,
    result: "succeeded",
  };
  vm.runInContext("executionView.observeRefresh(staleRefresh)", context);
  assert.equal(
    elements.get("refresh-execution").disabled,
    true,
    "late completion from an older refresh cannot enable the button",
  );
  serverRefresh = {
    epoch: serverRefresh.epoch,
    status: "idle",
    revision: serverRefresh.revision + 1,
    result: "succeeded",
  };
  await tick();
  assert.equal(elements.get("refresh-execution").disabled, false);
  assert.equal(elements.get("execution-refresh").hidden, true);
  const saved = tree.textContent;
  // Simulate the server cache on subsequent local polls, without freezing the
  // local observation. Both snapshots are real HTTP responses supplied by pytest.
  fail = true;
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(tree.textContent, saved);
  assert.equal(elements.get("execution-error").hidden, false);
  assert.ok(
    elements.get("execution-error").textContent.includes("前回の表示を保持"),
  );
  fail = false;
  await tick();
  assert.equal(
    elements.get("execution-error").hidden,
    false,
    "refresh error survives a local success",
  );
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(elements.get("execution-error").hidden, true);
  // A failed first Herdr read is shown, then local records load at once.
  const cached = input.local;
  input.local = input.unobserved;
  vm.runInContext("executionView.herdrRequested = false", context);
  fail = true;
  await vm.runInContext("executionView.load()", context);
  assert.ok(
    elements
      .get("execution-error")
      .textContent.includes("Herdrの取得に失敗しました"),
  );
  assert.ok([...timers.values()].some((t) => t.delay === 0));
  fail = false;
  await tick(0);
  assert.ok(nodeButton("parent:o/r#5"), "unobserved parents remain visible");
  assert.ok(graphText().includes("Herdr未取得"));
  input.local = cached;
  assert.deepEqual(executionRequests().slice(-2), [
    "/api/execution?herdr=true",
    "/api/execution",
  ]);
  assert.equal(
    elements.get("execution-error").hidden,
    false,
    "the Herdr failure stays visible until an explicit refresh succeeds",
  );
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(elements.get("execution-error").hidden, true);
  document.visibilityState = "hidden";
  document.events.visibilitychange();
  assert.equal(timers.size, 0, "hide cancels the execution timer");
  const count = executionRequests().length;
  await vm.runInContext("executionView.load(true)", context);
  assert.equal(executionRequests().length, count);
  document.visibilityState = "visible";
  document.events.visibilitychange();
  await settle();
  assert.equal(executionRequests().length, count + 1);
  assert.deepEqual(
    runtimeRequests(),
    ["/api/runtime"],
    "Herdr overview loads once, on the first visible display",
  );
  // The overview is one summary line until the user opens the card list.
  assert.ok(
    elements
      .get("runtime-observed")
      .textContent.startsWith("3エージェント（作業中 2 · 待機中 1）"),
  );
  assert.equal(elements.get("runtime-agents").children.length, 3);
  assert.deepEqual(
    descendants(elements.get("runtime-agents")).filter((node) => node.tagName === "IMG").map((node) => node.src),
    ["/static/icon-codex.svg", "/static/icon-claude.svg", "/static/icon-gemini.svg"],
  );
  assert.equal(elements.get("runtime-agents").hidden, true);
  elements.get("toggle-runtime").events.click();
  assert.equal(elements.get("runtime-agents").hidden, false);
  assert.equal(elements.get("toggle-runtime")["aria-expanded"], "true");
  elements.get("toggle-runtime").events.click();
  assert.equal(elements.get("runtime-agents").hidden, true);
  assert.equal(elements.get("toggle-runtime")["aria-expanded"], "false");
  view("timeline");
  await settle();
  assert.equal(elements.get("execution-panel").hidden, true);
  assert.equal(elements.get("feed-composer").hidden, false);
  assert.ok(![...timers.values()].some((t) => t.delay === 10000));
  view("execution");
  await settle();
  serverRefresh = {
    ...serverRefresh,
    status: "running",
    revision: serverRefresh.revision + 1,
  };
  await tick();
  assert.equal(elements.get("refresh-execution").disabled, true);
  serverRefresh = {
    epoch: "new-web-process",
    status: "idle",
    revision: 0,
    result: null,
  };
  await tick();
  assert.equal(
    elements.get("refresh-execution").disabled,
    false,
    "a server restart resets the process-local revision",
  );
  view("timeline");
  await settle();
  // A conversation POST can finish after the user has switched tabs. Its
  // completion must not change the active view or restart conversation polls.
  vm.runInContext(
    "writeJSON = async () => new Promise(resolve => { pendingSubmission = resolve; })",
    context,
  );
  elements.get("compose-body").value = "送信中の投稿";
  const submission = elements
    .get("compose")
    .events.submit({ preventDefault() {} });
  view("execution");
  await settle();
  context.pendingSubmission({ post: input.post });
  await submission;
  await settle();
  assert.equal(vm.runInContext("state.view", context), "execution");
  assert.equal(elements.get("execution-panel").hidden, false);
  assert.ok(![...timers.values()].some((t) => t.delay === 3000));
  view("timeline");
  await settle();
  assert.ok([...timers.values()].some((t) => t.delay === 3000));
  const before = executionRequests().length;
  await vm.runInContext("executionView.load()", context);
  assert.equal(
    executionRequests().length,
    before,
    "conversation tab never polls execution",
  );
  // A pending local response must not restart polling after leaving the tab.
  view("execution");
  view("timeline");
  await settle();
  assert.ok(![...timers.values()].some((t) => t.delay === 10000));
  document.visibilityState = "hidden";
  document.events.visibilitychange();
  document.visibilityState = "visible";
  document.events.visibilitychange();
  await settle();
  assert.equal(runtimeRequests().length, 1, "re-showing does not poll Herdr");
  console.log(
    "execution graph, drawer persistence, explicit refresh, failure retention and virtual-clock polling passed",
  );
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
