"use strict";

// Run the complete production script against a small in-memory DOM, with no
// packages or live server. Python supplies a real /api/changes post on stdin.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.text = "";
    this.value = "";
    this.style = {};
    this.scrollHeight = 38;
    this.classList = { toggle() {}, add() {} };
  }
  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }
  get textContent() {
    return this.text + this.children.map((child) => child.textContent).join("");
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this.text = "";
    this.children = [...children];
  }
  addEventListener() {}
  setAttribute(name, value) {
    this[name] = value;
  }
  querySelectorAll() {
    return [];
  }
  removeAttribute(name) {
    delete this[name];
  }
}

const elements = new Map();
const document = {
  // Suppress the initial poll; rendering and thread loading run explicitly.
  visibilityState: "hidden",
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => {
    const node = new Element("#text");
    node.textContent = text;
    return node;
  },
  getElementById: (id) => {
    if (!elements.has(id)) elements.set(id, new Element("div"));
    return elements.get(id);
  },
  querySelectorAll: () => [],
  addEventListener() {},
};

const post = JSON.parse(fs.readFileSync(0, "utf8"));
const context = vm.createContext({
  document,
  window: { addEventListener() {} },
  location: { hash: "" },
  clearTimeout() {},
  setTimeout() {},
  AbortSignal,
  ResizeObserver: class {
    observe() {}
  },
  post,
  fetch: async (url) => {
    assert.equal(url, `/api/thread/${post.id}`);
    return {
      ok: true,
      json: async () => ({ thread_id: post.thread_id, posts: [post] }),
    };
  },
});
vm.runInContext(
  fs.readFileSync(
    require("node:path").join(
      require("node:path").dirname(process.argv[2]),
      "execution.js",
    ),
    "utf8",
  ),
  context,
);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);

const descendants = (node) => [node, ...node.children.flatMap(descendants)];
function checkCard(card) {
  assert.equal(card.tagName, "ARTICLE");
  assert.equal(card.dataset.postId, post.id);
  assert.ok(card.textContent.includes(post.body));
  const parties = descendants(card).filter(
    (node) => node.className === "party",
  );
  assert.equal(parties.length, 1, "only the author remains in the header");
  const humans = parties.filter((node) => node.textContent === "human");
  assert.equal(humans.length, post.sender.name === "human" ? 1 : 0);
  for (const human of humans) {
    assert.equal(
      descendants(human).filter((node) => node.tagName === "A").length,
      0,
    );
  }
  const parents = parties.filter((node) => !humans.includes(node));
  assert.equal(parents.length, post.sender.name === "human" ? 0 : 1);
  const parent =
    post.sender.name === "human" ? post.recipients[0] : post.sender;
  for (const party of parents) {
    const links = descendants(party).filter((node) => node.tagName === "A");
    assert.equal(links.length, 1);
    assert.equal(
      links[0].href,
      `https://github.com/${parent.repo}/issues/${parent.parent}`,
    );
  }
  assert.ok(!card.textContent.includes("#null"));
  assert.ok(
    descendants(card).some(
      (node) =>
        (node.className || "").includes("mention") &&
        node.textContent === `@${post.recipients[0].name}`,
    ),
  );
  assert.equal(
    descendants(card).filter((node) =>
      ["route", "delivery-summary", "post-footer"].includes(node.className),
    ).length,
    0,
  );
}

async function main() {
  vm.runInContext("state.posts.set(post.id, post)", context);
  context.renderFeed();
  const cards = elements
    .get("feed")
    .children.filter((node) => node.tagName === "ARTICLE");
  assert.equal(cards.length, 1);
  checkCard(cards[0]);

  await context.openThread(post.id);
  assert.equal(elements.get("thread").children.length, 1);
  checkCard(elements.get("thread").children[0]);
  assert.ok(!elements.get("error")?.textContent);

  const parent =
    post.sender.name === "human" ? post.recipients[0] : post.sender;
  for (const key of ["repo", "parent"]) {
    const invalid = { ...parent, [key]: null };
    const error = {
      name: "Error",
      message: "親のリポジトリまたはissue番号が不正です。",
    };
    assert.throws(() => context.party(invalid), error);
    delete invalid[key];
    assert.throws(() => context.party(invalid), error);
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
