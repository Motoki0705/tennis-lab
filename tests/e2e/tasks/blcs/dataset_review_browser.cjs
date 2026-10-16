/* Read-only browser regression for the BLCS dataset scene review UI. */
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const PORT = Number(process.env.BLCS_REVIEW_PORT || 8781);
const BASE = process.env.BLCS_REVIEW_URL || `http://127.0.0.1:${PORT}`;
const DATA_ROOT =
  process.env.DATASET_REVIEW_DATA_ROOT || "/home/kamimura/projects/tennis-lab/data";

const BALL_COLORS = [
  [224, 97, 31],
  [47, 127, 209],
  [31, 158, 107],
  [192, 57, 43],
  [142, 92, 196],
  [212, 160, 23],
  [15, 139, 141],
  [200, 90, 142],
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
  if (process.env.BLCS_REVIEW_URL) return null;
  const child = spawn(
    "bash",
    [
      "./scripts/run_in_repo_venv.sh",
      "python",
      "-m",
      "src.tasks.blcs.scripts.review_dataset",
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

async function countPixels(page, colors, tolerance) {
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
        if (matched) count += 1;
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
    assert.match(await page.locator("#scene-dataset").textContent(), /blcs\//);
    await page.waitForSelector("#blcs-content:not([hidden])");
    assert.equal(await page.locator(".blcs-camera").count(), 6);
    assert.match(await page.locator(".blcs-inspection header").textContent(), /合成truth.*RGB/s);
    assert.match(await page.locator("#blcs-normalization").textContent(), /11\.885.*位置 一致.*速度 一致/s);

    // Pause, then seek across the timeline. Ball tracks are ball-present only
    // for part of the scene, so scan until a frame decodes a live ball and
    // assert both the court and the ball marker are actually drawn. The
    // tolerance keeps the green court (63, 125, 100) out of the ball palette.
    await page.click("#play");
    assert.equal(await page.getAttribute("#play", "title"), "再生");
    const court = await countPixels(page, [[63, 125, 100]], 26);
    assert.ok(court > 1500, `court pixels: ${court}`);
    let ball = 0;
    let liveFrame = -1;
    for (let frame = 0; frame < 1024 && liveFrame < 0; frame += 24) {
      await page.evaluate((value) => {
        const scrub = document.getElementById("scrub");
        scrub.value = String(value);
        scrub.dispatchEvent(new Event("input", { bubbles: true }));
      }, frame);
      await page.waitForTimeout(50);
      ball = await countPixels(page, BALL_COLORS, 14);
      const hud = (await page.locator("#hud-pos").textContent())?.trim();
      if (ball >= 3 && hud && hud !== "–") liveFrame = frame;
    }
    assert.ok(liveFrame >= 0, "no frame rendered a present ball");
    assert.ok(ball >= 3, `ball pixels at frame ${liveFrame}: ${ball}`);

    // Actual saved coordinates outside the image are retained, and the panel
    // follows the same zero-based frame as the shared 3D transport.
    await page.evaluate(() => {
      const scrub = document.getElementById("scrub");
      scrub.value = "248";
      scrub.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForFunction(() => document.querySelector(".blcs-inspection").dataset.frame === "248");
    assert.equal((await page.locator("#hud-frame").textContent()).trim(), "248");
    assert.match(await page.locator("#blcs-frame").textContent(), /8\.267 s/);
    assert.match(await page.locator('[data-camera="cam_0"] .blcs-badge').textContent(), /非可視.*画像外/);
    assert.match(await page.locator("#blcs-event-now").textContent(), /shot 5 hit/);
    assert.match(await page.locator("#blcs-event-summary").textContent(), /区間外候補 19件/);
    await page.selectOption("#blcs-camera-select", "cam_0");
    assert.equal(await page.locator(".blcs-camera:visible").count(), 1);
    assert.equal(await page.locator(".blcs-cameras").getAttribute("data-single"), "true");
    await page.selectOption("#blcs-camera-select", "all");
    assert.equal(await page.locator(".blcs-camera:visible").count(), 6);
    const imageOutside = await page.locator('[data-camera="cam_0"] canvas').evaluate(canvas => canvas.toDataURL());
    await page.click('#blcs-event-buttons button[data-frame="315"]');
    assert.equal(await page.locator(".blcs-inspection").getAttribute("data-frame"), "315");
    assert.equal((await page.locator("#hud-frame").textContent()).trim(), "315");
    const imageInside = await page.locator('[data-camera="cam_0"] canvas').evaluate(canvas => canvas.toDataURL());
    assert.notEqual(imageOutside, imageInside);

    const before = await page.locator(".scene").count();
    await page.fill("#query", "scene_00055");
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
    assert.equal(await page.getAttribute("#play", "title"), "一時停止");

    await page.click("#trail");
    assert.equal(await page.getAttribute("#trail", "aria-pressed"), "false");
    await page.click("#trail");
    await page.click("[data-preset='broadcast']");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Home");

    // Scene changes clear previous evidence and bind the inspection to the new
    // scene/revision. Fast consecutive selection must not render stale UVs.
    await page.locator('.scene[data-scene="scene_000001"]').click();
    await page.locator('.scene[data-scene="scene_000002"]').click();
    await loaded();
    await page.waitForFunction(() => document.querySelector(".blcs-inspection").dataset.scene === "scene_000002");

    await page.setViewportSize({width:600,height:900});
    await page.waitForTimeout(200);

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
      `blcs dataset review browser regression passed (court=${court} ball=${ball})`,
    );
  } finally {
    await browser.close();
    if (server) server.kill("SIGTERM");
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
