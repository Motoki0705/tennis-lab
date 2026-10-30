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
    return {
      ok: !fail,
      status: fail ? 503 : 200,
      json: async () => {
        if (url.startsWith("/api/execution"))
          return structuredClone(
            url.includes("refresh=true") ? input.refreshed : input.local,
          );
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
  await tick();
  assert.equal(executionRequests().length, 2);
  elements.get("refresh-execution").events.click();
  await settle();
  assert.equal(executionRequests().at(-1), "/api/execution?refresh=true");
  assert.ok(tree.textContent.includes("Issue 未完了（OPEN）"));
  assert.ok(
    tree.textContent.includes("ターン終了はIssueの受入完了を意味しません。"),
  );
  assert.ok(tree.textContent.includes("PR #88"));
  const saved = tree.textContent;
  // Simulate the server cache on subsequent local polls, without freezing the
  // local observation. Both snapshots are real HTTP responses supplied by pytest.
  input.local = input.refreshed;
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
  view("timeline");
  await settle();
  assert.equal(elements.get("execution-panel").hidden, true);
  assert.equal(elements.get("feed-composer").hidden, false);
  assert.ok(![...timers.values()].some((t) => t.delay === 10000));
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
  console.log(
    "execution rendering, explicit refresh, failure retention and virtual-clock polling passed",
  );
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
