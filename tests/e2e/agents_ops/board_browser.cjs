"use strict";

// Real browser interactions against a pytest-owned server, temporary DB and
// fake notification routes. No live parent is contacted.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const { url, rootId, recipient, body } = input;

async function main() {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.BOARD_BROWSER_EXECUTABLE
      ? { executablePath: process.env.BOARD_BROWSER_EXECUTABLE }
      : {}),
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [],
      external = [],
      writes = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      // Chromium may request an implicit favicon; it is not a board asset.
      const favicon =
        message.location().url === `${url}/favicon.ico` &&
        message.text().includes("404");
      if (
        message.type() === "error" &&
        !message.text().includes("503") &&
        !favicon
      )
        errors.push(message.text());
    });
    page.on("request", (request) => {
      if (!request.url().startsWith(url + "/")) external.push(request.url());
      if (request.method() === "POST") writes.push(request.url());
    });
    await page.goto(url);
    await page.waitForSelector(`#feed article[data-post-id="${rootId}"]`);
    assert.equal(
      await page.locator(".route, .delivery-summary, .post-footer").count(),
      0,
    );
    assert.ok(
      await page.locator(".topbar").evaluate((node) => node.offsetHeight <= 44),
    );
    assert.ok(
      await page
        .locator(".feed-pane .pane-heading")
        .evaluate((node) => node.offsetHeight <= 42),
    );
    assert.ok(
      await page
        .locator("#feed-scroll")
        .evaluate((node) => node.clientHeight > innerHeight * 0.7),
    );
    assert.equal(
      await page.locator("#feed article").count(),
      1,
      "replies stay in threads",
    );
    assert.equal(
      await page.locator("#feed img").count(),
      0,
      "HTML is plain text",
    );
    assert.ok((await page.locator("#feed").textContent()).includes(body));
    assert.ok(
      (await page.locator("#feed .thread-link").textContent()).includes(
        "1件の返信",
      ),
    );
    await page.locator("#feed .thread-link").click();
    await page.waitForSelector("#thread article:nth-of-type(2)");
    assert.equal(await page.locator("#thread article").count(), 2);
    const root = page.locator(`#thread article[data-post-id="${rootId}"]`);
    await root.hover();
    await root.getByRole("button", { name: "既読にする", exact: true }).click();
    await root
      .getByRole("button", { name: "既読にする", exact: true })
      .waitFor({ state: "detached" });
    const read = (
      await (await page.request.get(`${url}/api/thread/${rootId}`)).json()
    ).posts[0];
    assert.ok(read.recipients.find((p) => p.name === "human").read_at);
    assert.ok(
      (await root.locator(".read-receipt").getAttribute("title")).includes(
        "human",
      ),
    );
    assert.ok(
      (await root.locator(".read-receipt").textContent()).includes("👀"),
    );
    assert.deepEqual(
      read.reactions,
      [],
      "read receipts do not invent manual reactions",
    );
    assert.equal(
      read.recipients.find((p) => p.name === "human").response_state,
      null,
    );
    await root.hover();
    await root
      .getByRole("button", { name: "スタンプを追加", exact: true })
      .click();
    await root.getByRole("button", { name: "いいね", exact: true }).click();
    const reaction = root.locator('.reaction[aria-pressed="true"]');
    await reaction.waitFor();
    assert.equal(await reaction.getAttribute("title"), "human");
    await reaction.click();
    await reaction.waitFor({ state: "detached" });

    // Thread drafts are associated with the selected post, including clearing.
    const replyInput = page.locator("#reply-body");
    await replyInput.fill("返信の下書き");
    const secondId = await page
      .locator("#thread article")
      .nth(1)
      .getAttribute("data-post-id");
    await page.locator(`#thread article[data-post-id="${secondId}"]`).hover();
    await page
      .locator(`#thread article[data-post-id="${secondId}"] .reply-action`)
      .click();
    await page.waitForFunction(
      (id) => state.selected === id && state.threadId,
      secondId,
    );
    assert.equal(await replyInput.inputValue(), "");
    await root.hover();
    await page
      .locator(`#thread article[data-post-id="${rootId}"] .reply-action`)
      .click();
    await page.waitForFunction(
      (id) => state.selected === id && state.threadId,
      rootId,
    );
    assert.equal(await replyInput.inputValue(), "返信の下書き");
    await replyInput.fill("ブラウザから返信しました");
    await replyInput.press("Enter");
    await page.waitForFunction(
      () => document.querySelectorAll("#thread article").length === 3,
    );
    // A read poll can observe the committed reply while POST still waits for
    // notification delivery. Wait for the submission itself before its reset.
    await page.waitForFunction(() => !composers.reply.busy);
    assert.equal(await replyInput.inputValue(), "");
    const thread = (
      await (await page.request.get(`${url}/api/thread/${rootId}`)).json()
    ).posts;
    assert.equal(thread.at(-1).sender.name, "human");
    assert.equal(thread.at(-1).in_reply_to, rootId);
    assert.equal(thread.at(-1).recipients[0].name, input.sender);
    assert.equal(
      thread[0].recipients.find((p) => p.name === "human").response_state,
      "answered",
    );
    await page.locator("#close-thread").click();
    await page.locator("#thread-pane").waitFor({ state: "hidden" });

    // Autocomplete accepts a candidate; IME Enter never sends; Shift+Enter
    // inserts a line break. Failures preserve input and are not auto-retried.
    const compose = page.locator("#compose-body");
    const initialHeight = await compose.evaluate((node) => node.offsetHeight);
    assert.ok(initialHeight <= 40, "composer starts at one line");
    await compose.fill(
      Array(15).fill("長いメッセージを入力します。").join("\n"),
    );
    const grownHeight = await compose.evaluate((node) => node.offsetHeight);
    assert.ok(grownHeight > initialHeight * 2 && grownHeight <= 220);
    await compose.fill("短文");
    assert.equal(
      await compose.evaluate((node) => node.offsetHeight),
      initialHeight,
    );
    await compose.fill(`@${recipient.slice(0, 2)}`);
    await page.locator("#compose-suggestions").waitFor({ state: "visible" });
    const before = writes.length;
    await compose.press("Enter");
    assert.equal(await compose.inputValue(), `@${recipient} `);
    assert.equal(writes.length, before);
    await page.keyboard.insertText("入力確認");
    await compose.press("Shift+Enter");
    await page.keyboard.insertText("二行目です");
    await compose.evaluate((node) =>
      node.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          isComposing: true,
          bubbles: true,
        }),
      ),
    );
    assert.equal(writes.length, before);
    const draft = await compose.inputValue();
    await page.route(`${url}/api/posts`, (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: '{"detail":"unavailable"}',
      }),
    );
    await compose.press("Enter");
    await page.locator("#compose-error").waitFor({ state: "visible" });
    assert.equal(await compose.inputValue(), draft);
    await page.unroute(`${url}/api/posts`);
    const posted = page.waitForResponse(
      (res) =>
        res.url() === `${url}/api/posts` && res.request().method() === "POST",
    );
    await compose.press("Enter");
    const saved = (await (await posted).json()).post;
    await page.waitForFunction(
      () => document.getElementById("compose-body").value === "",
    );
    assert.equal(
      await compose.evaluate((node) => node.offsetHeight),
      initialHeight,
    );
    assert.equal(saved.body, draft);
    assert.equal(saved.sender.name, "human");
    assert.deepEqual(
      saved.recipients.map((p) => p.name),
      [recipient],
    );
    assert.equal(writes.filter((u) => u === `${url}/api/posts`).length, 2);
    const card = page.locator(`#feed article[data-post-id="${saved.id}"]`);
    await card.waitFor();
    assert.equal(await card.locator(".mention").textContent(), `@${recipient}`);

    // Switching recipients keeps drafts separate. A channel recipient is
    // explicit in the composer and persisted as the post's actual recipient.
    await compose.fill("全体の下書き");
    await page
      .locator("#roster .roster-button")
      .filter({ hasText: recipient })
      .click();
    assert.equal(await compose.inputValue(), "");
    assert.ok(
      (await page.locator("#compose-hint").textContent()).includes(recipient),
    );
    await compose.fill("個別の下書き");
    await page.locator("#all-channel").click();
    assert.equal(await compose.inputValue(), "全体の下書き");
    await page
      .locator("#roster .roster-button")
      .filter({ hasText: recipient })
      .click();
    assert.equal(await compose.inputValue(), "個別の下書き");
    await compose.fill("");
    await page.locator("#all-channel").click();
    await compose.fill("");
    await page.locator("#search").fill("存在しない会話");
    assert.equal(await page.locator("#feed article").count(), 0);
    await page.locator("#search").fill("");
    await page.locator('[data-view="waiting"]').click();
    assert.equal(await page.locator("#feed article").count(), 0);
    await page.locator('[data-view="timeline"]').click();
    await card.hover();
    await page
      .locator(`#feed article[data-post-id="${saved.id}"] .reply-action`)
      .click();
    await page.waitForFunction((id) => state.threadId === id, saved.id);
    // Roster failures retain the previous display and remain visible until
    // the next explicit refresh. Background pages stop polling entirely.
    const rosterCount = await page.locator("#roster .roster-button").count();
    await page.route("**/api/changes?**", async (route) => {
      if (new URL(route.request().url()).searchParams.has("refresh_roster"))
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          body: '{"detail":"unavailable"}',
        });
      else await route.continue();
    });
    await page.locator("#refresh-roster").click();
    await page.locator("#error").waitFor({ state: "visible" });
    await page.waitForTimeout(3200);
    assert.ok(await page.locator("#error").isVisible());
    assert.equal(
      await page.locator("#roster .roster-button").count(),
      rosterCount,
    );
    await page.unroute("**/api/changes?**");
    await page.locator("#refresh-roster").click();
    await page.locator("#error").waitFor({ state: "hidden" });
    await page.evaluate(() => {
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        configurable: true,
      });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    let polls = 0;
    const countPolls = (request) => {
      if (request.url().includes("/api/changes")) polls++;
    };
    page.on("request", countPolls);
    await page.waitForTimeout(3200);
    assert.equal(polls, 0);
    await page.evaluate(() => {
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        configurable: true,
      });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    await page.waitForResponse((res) => res.url().includes("/api/changes"));
    page.off("request", countPolls);
    assert.equal(
      await page.evaluate(
        () => document.querySelector(".topbar").getBoundingClientRect().top,
      ),
      0,
      "the app header must stay on screen",
    );
    if (process.env.BOARD_BROWSER_ARTIFACT_DIR) {
      fs.mkdirSync(process.env.BOARD_BROWSER_ARTIFACT_DIR, { recursive: true });
      await page.screenshot({
        path: path.join(
          process.env.BOARD_BROWSER_ARTIFACT_DIR,
          "board-desktop.png",
        ),
      });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.locator("#reply-body").isVisible());
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    if (process.env.BOARD_BROWSER_ARTIFACT_DIR)
      await page.screenshot({
        path: path.join(
          process.env.BOARD_BROWSER_ARTIFACT_DIR,
          "board-mobile.png",
        ),
      });
    await page.locator("#close-thread").click();
    assert.ok(await compose.isVisible());
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    await page.locator('[data-view="execution"]').click();
    await page.locator(".graph-child-node").first().waitFor();
    assert.equal(await page.locator("#thread-pane").isVisible(), false);
    assert.equal(await page.locator("#compose").isVisible(), false);
    const parentNode = page.locator(".graph-parent-node").first();
    assert.ok((await parentNode.textContent()).includes("子の進行管理"));
    assert.equal(await page.locator("#execution-graph img").count(), 0);
    // Tree connector: the first child's branch starts under the parent icon.
    const icon = await parentNode.locator(".graph-icon").boundingBox();
    const branch = await page
      .locator(".graph-leaf")
      .first()
      .evaluate((node) => {
        const box = node.getBoundingClientRect();
        const style = getComputedStyle(node, "::before");
        return box.left + parseFloat(style.left);
      });
    assert.ok(Math.abs(branch + 1 - (icon.x + icon.width / 2)) <= 2);
    const child = page.locator(".graph-child-node").first();
    const drawer = page.locator("#execution-drawer");
    // Narrow screens (still 390px here): the drawer overlays the graph and
    // closes with Escape, without horizontal page scroll.
    await child.click();
    await drawer.waitFor();
    assert.equal(
      await drawer.evaluate((node) => getComputedStyle(node).position),
      "absolute",
    );
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    if (process.env.BOARD_BROWSER_ARTIFACT_DIR)
      await page.screenshot({
        path: path.join(
          process.env.BOARD_BROWSER_ARTIFACT_DIR,
          "board-execution-mobile.png",
        ),
      });
    await drawer.press("Escape");
    assert.equal(await drawer.isVisible(), false);
    // Desktop: the drawer is a side panel next to the graph.
    await page.setViewportSize({ width: 1440, height: 1000 });
    await child.click();
    await drawer.waitFor();
    const graphBox = await page.locator("#execution-graph").boundingBox();
    const drawerBox = await drawer.boundingBox();
    assert.ok(drawerBox.x >= graphBox.x + graphBox.width - 1);
    assert.ok(
      (await drawer.textContent()).includes(
        "<img src=x onerror=alert(1)> を文字として表示",
      ),
    );
    assert.equal(await drawer.locator("img").count(), 0);
    await page.locator("#refresh-execution").click();
    await drawer
      .getByText("Issue 未完了（OPEN）", { exact: true })
      .first()
      .waitFor();
    await page.getByText("+2 完了", { exact: true }).waitFor();
    assert.equal(
      await child.getAttribute("aria-pressed"),
      "true",
      "the selected node survives the explicit refresh",
    );
    if (process.env.BOARD_BROWSER_ARTIFACT_DIR) {
      await page.locator(".graph-fold.fold-done").click();
      await page.screenshot({
        path: path.join(
          process.env.BOARD_BROWSER_ARTIFACT_DIR,
          "board-execution.png",
        ),
      });
    }
    assert.deepEqual(external, [], "no external assets or requests");
    assert.deepEqual(errors, [], "no JavaScript or CSP errors");
    console.log(
      "browser interactions passed: compose, reply, reactions, read, mentions, IME, failure drafts, filters, execution graph, mobile, CSP",
    );
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
