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
const tabs = ["timeline", "waiting", "issues", "execution"].map((view) => {
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
  cacheAvailable = false;
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
        agents: [],
        observed_at: input.local.roster_observed_at,
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
          roster_observed_at: input.local.roster_observed_at,
        };
      },
    };
  },
});
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
const tick = async () => {
  const [id, timer] = [...timers].find(([, timer]) => timer.delay === 10000);
  timers.delete(id);
  timer.callback();
  await settle();
};

async function main() {
  assert.equal(requests.length, 0, "hidden pages do not poll");
  document.visibilityState = "visible";
  view("execution");
  await settle();
  assert.deepEqual(executionRequests(), ["/api/execution"]);
  assert.deepEqual(
    [...timers.values()].map((t) => t.delay),
    [10000],
  );
  assert.equal(elements.get("feed-composer").hidden, true);
  assert.equal(elements.get("execution-panel").hidden, false);
  assert.equal(tabs[3]["aria-pressed"], "true");
  const tree = elements.get("execution-parents");
  for (const label of [
    "親 · Herdr 待機中",
    "実行中",
    "ターン終了",
    "失敗",
    "停止中",
    "依存待ち",
    "委任予約済み・未起動",
    "未委任",
    "状態不明",
    "Issue 未取得",
    "停止とは限りません",
    "非表示（秘匿検査）",
    "runner.result",
    "identity_verified",
    "lookup · search",
    "変更 2件",
    "親が差分を確認する",
  ])
    assert.ok(tree.textContent.includes(label), label);
  assert.ok(!tree.textContent.includes("undefined"));
  assert.equal(
    descendants(tree).filter((node) => node.tagName === "IMG").length,
    0,
  );
  assert.ok(tree.textContent.includes("<img src=x onerror=alert(1)>"));
  // Evidence remains a collector projection: new contract codes render through
  // the same text-only path, without Web-side state inference or field values.
  for (const [code, status, label] of [
    ["contract_verified", "turn_completed", "ターン終了"],
    ["contract_missing", "unknown", "状態不明"],
    ["contract_mismatch", "unknown", "状態不明"],
  ]) {
    context.contractParent = input.local.parents[0];
    context.contractChild = {
      ...structuredClone(input.local.parents[0].children[0]),
      status,
      label,
      evidence: [
        { source: "runner.result", code, fields: ["contract.branch"] },
      ],
    };
    const card = vm.runInContext(
      "executionView.childCard(contractParent, contractChild)",
      context,
    );
    assert.ok(card.textContent.includes(`${code} · contract.branch`));
    assert.ok(card.textContent.includes(label));
    assert.ok(!card.textContent.includes("undefined"));
  }
  await tick();
  assert.equal(executionRequests().length, 2);
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(
    executionRequests().filter((url) => url.includes("refresh=true")).length,
    1,
  );
  assert.ok(tree.textContent.includes("Issue 未完了（OPEN）"));
  assert.ok(
    tree.textContent.includes("ターン終了はIssueの受入完了を意味しません。"),
  );
  assert.ok(tree.textContent.includes("PR #88"));
  holdRefresh = true;
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(elements.get("refresh-execution").disabled, true);
  assert.equal(elements.get("execution-refresh").hidden, false);
  assert.ok(
    elements.get("execution-refresh").textContent.includes("GitHub取得中"),
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
    "execution rendering, explicit refresh, failure retention and virtual-clock polling passed",
  );
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
