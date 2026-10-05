/* Read-only browser regression for the PLCS dataset scene review UI. */
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const PORT = Number(process.env.PLCS_REVIEW_PORT || 8780);
const BASE = process.env.PLCS_REVIEW_URL || `http://127.0.0.1:${PORT}`;
const DATA_ROOT =
  process.env.DATASET_REVIEW_DATA_ROOT || "/home/kamimura/projects/tennis-lab/data";

const PLAYER_COLORS = [
  [31, 138, 112],
  [209, 98, 63],
  [58, 111, 176],
  [199, 154, 18],
  [142, 92, 196],
  [192, 57, 43],
  [15, 139, 141],
  [179, 84, 138],
  [91, 107, 127],
  [122, 139, 47],
];

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForServer(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      // server not up yet
    }
    await delay(500);
  }
  throw new Error(`review server did not start at ${url}`);
}

async function startServer() {
  if (process.env.PLCS_REVIEW_URL) return null;
  const child = spawn(
    "bash",
    [
      "./scripts/run_in_repo_venv.sh",
      "python",
      "-m",
      "src.tasks.plcs.scripts.review_dataset",
      "--data-root",
      DATA_ROOT,
      "--port",
      String(PORT),
    ],
    { cwd: process.cwd(), stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  await waitForServer(BASE, 120000);
  return child;
}

async function countPixels(page, target, tolerance) {
  const colors = Array.isArray(target[0]) ? target : [target];
  return page.evaluate(
    ([targets, tol]) => {
      const canvas = document.getElementById("view");
      // The viewport is a WebGL canvas; read the rendered frame back directly.
      const context =
        canvas.getContext("webgl2") || canvas.getContext("webgl");
      const width = canvas.width;
      const height = canvas.height;
      const data = new Uint8Array(width * height * 4);
      context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, data);
      let count = 0;
      for (let index = 0; index < data.length; index += 4) {
        const matched = targets.some(
          ([r, g, b]) =>
            Math.abs(data[index] - r) < tol &&
            Math.abs(data[index + 1] - g) < tol &&
            Math.abs(data[index + 2] - b) < tol,
        );
        if (matched) {
          count += 1;
        }
      }
      return count;
    },
    [colors, tolerance],
  );
}

(async () => {
  const server = await startServer();
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    const loaded = () =>
      page.waitForFunction(
        () => {
          const status = document.getElementById("status");
          const transport = document.getElementById("transport");
          return status.hidden && !transport.hidden;
        },
        {},
        { timeout: 120000 },
      );

    await page.goto(BASE);
    await page.waitForSelector(".scene", { timeout: 120000 });
    await loaded();

    assert.equal(await page.locator(".form").count(), 1);
    assert.match(await page.locator("#scene-dataset").textContent(), /plcs\//);

    await page.waitForFunction(() => document.getElementById("inspection-status").hidden, {}, { timeout: 120000 });
    await page.click("#play");
    await page.locator("#scrub").evaluate(el => { el.value = "100"; el.dispatchEvent(new Event("input", { bubbles: true })); });
    assert.match(await page.locator("#inspection-frame").textContent(), /frame 100 /);
    assert.equal(await page.locator("#human-visible").textContent(), "17 / 17");
    assert.match(await page.locator("#motion-source").textContent(), /ACCAD/);
    assert.match(await page.locator(".synthetic-note").textContent(), /合成投影/);
    assert.equal(await page.locator("#visibility-mismatch").textContent(), "0点");
    await page.click('[data-camera-index="3"]');
    assert.equal(await page.locator("#human-visible").textContent(), "0 / 17");
    assert.match(await page.locator("#frame-check").textContent(), /有効観測なし/);
    await page.click('[data-camera-index="0"]');
    await page.check("#all-court-points");
    assert.match(await page.locator("#court-visible").textContent(), /\/ 20/);
    await page.uncheck("#all-court-points");
    await page.click("#next");
    assert.match(await page.locator("#inspection-frame").textContent(), /frame 101 /);
    assert.equal(await page.locator("#hud-frame").textContent(), "101");
    await page.click('[data-preset="overhead"]');
    await page.locator("#view").hover();
    await page.mouse.wheel(0, 400);

    await page.waitForTimeout(600);
    const court = await countPixels(page, [[63, 125, 100]], 26);
    // A tight tolerance keeps the green court (63, 125, 100) from matching the
    // nearest player shade (31, 138, 112), so this counts real player pixels.
    const player = await countPixels(page, PLAYER_COLORS, 14);
    assert.ok(court > 1500, `court pixels: ${court}`);
    assert.ok(player > 40, `player pixels: ${player}`);

    const before = await page.locator(".scene").count();
    await page.fill("#query", "scene_00012");
    await page.waitForTimeout(600);
    const after = await page.locator(".scene").count();
    assert.ok(after > 0 && after < before, `search count ${before} -> ${after}`);
    assert.match(await page.locator("#result-count").textContent(), /一致/);
    await page.click("#clear");
    await page.waitForTimeout(200);

    const cameras = page.locator("#cameras");
    await cameras.click();
    assert.equal(await cameras.getAttribute("aria-pressed"), "false");
    await cameras.click();
    assert.equal(await cameras.getAttribute("aria-pressed"), "true");

    await page.click("#play");
    await page.click("#play");
    assert.equal(await page.getAttribute("#play", "title"), "再生");
    await page.click("#play");
    assert.equal(await page.getAttribute("#play", "title"), "一時停止");

    await page.click("#follow");
    assert.equal(await page.getAttribute("#follow", "aria-pressed"), "true");
    await page.click("[data-preset='overhead']");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("Home");

    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    assert.ok(
      overflow.scrollWidth <= overflow.clientWidth + 1,
      `horizontal overflow: ${JSON.stringify(overflow)}`,
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(300);
    const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
    assert.equal(mobileOverflow, false, "mobile horizontal overflow");
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.route("**/api/scene/observations?**", route => route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Scene file is missing." }) }));
    await page.click('[data-scene="scene_000001"]');
    await page.waitForFunction(() => document.getElementById("inspection-status").classList.contains("error"), {}, { timeout: 120000 });
    assert.equal(await page.locator("#inspection-body").isVisible(), false);
    assert.match(await page.locator("#inspection-status").textContent(), /検品は利用できません/);
    // The intentionally rejected request must be the only browser console error.
    const unexpectedErrors = errors.filter(message => !message.includes("404"));
    assert.deepEqual(unexpectedErrors, []);
    console.log(
      `plcs dataset review browser regression passed (court=${court} player=${player})`,
    );
  } finally {
    await browser.close();
    if (server) server.kill("SIGTERM");
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
