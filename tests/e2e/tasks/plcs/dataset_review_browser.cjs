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

    assert.equal(await page.locator(".form").count(), 6);
    assert.match(await page.locator("#scene-dataset").textContent(), /plcs\//);

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
    assert.deepEqual(errors, []);
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
