/* Read-only SLCS browser smoke: mixed 3D labels, browsing and playback. */
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const port = Number(process.env.SLCS_REVIEW_PORT || 8784);
const base = process.env.SLCS_REVIEW_URL || `http://127.0.0.1:${port}`;
const root = process.env.SLCS_REVIEW_DATASET_ROOT || "/home/kamimura/projects/tennis-lab/data/slcs/real_rgb_v1";

(async () => {
  let server;
  let browser;
  try {
    if (!process.env.SLCS_REVIEW_URL) {
      server = spawn(".venv/bin/python", ["-m", "src.tasks.slcs.scripts.review_dataset", "--dataset-root", root, "--port", String(port)], { stdio: ["ignore", "ignore", "pipe"] });
      server.stderr.on("data", (chunk) => process.stderr.write(chunk));
      let ready = false;
      for (let attempt = 0; attempt < 120; attempt += 1) {
        if (server.exitCode !== null) throw new Error(`server exited: ${server.exitCode}`);
        try { ready = (await fetch(`${base}/api/catalog`)).ok; } catch {}
        if (ready) break;
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      assert.ok(ready, "review server did not start");
    }
    browser = await chromium.launch({
      headless: true,
      ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
      args: ["--no-sandbox"],
    });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const loaded = () => page.waitForFunction(() => document.getElementById("status").hidden && !document.getElementById("transport").hidden);
    await page.goto(base);
    await loaded();
    assert.equal(await page.title(), "SLCS Dataset Review");
    assert.match(await page.locator("#legend").textContent(), /選手 2.*ボール 1/);
    assert.match(await page.locator("#scene-note").textContent(), /疑似ラベル.*未校正/);
    assert.ok(await page.locator("#cameras").isDisabled());
    assert.ok(await page.locator("#open-camera").isDisabled());
    await page.click("#play");
    assert.equal(await page.locator("#play").getAttribute("title"), "再生");

    // Teacher-quality masks may hide the start of a real clip. Seek to a
    // labeled frame before inspecting pixels, with playback paused so the
    // sampled frame cannot race into a masked gap.
    const visibleFrame = await page.evaluate(async () => {
      const selected = document.querySelector('.scene[aria-current="true"]');
      const query = new URLSearchParams({ form: selected.dataset.form, scene: selected.dataset.scene });
      const sceneResponse = await fetch(`/api/scene?${query}`);
      if (!sceneResponse.ok) throw new Error(`scene HTTP ${sceneResponse.status}`);
      const scene = await sceneResponse.json();
      query.set("revision", scene.revision);
      const bufferResponse = await fetch(`/api/scene/buffer?${query}`);
      if (!bufferResponse.ok) throw new Error(`buffer HTTP ${bufferResponse.status}`);
      const { decodeBuffers } = await import("/static/model.mjs");
      const groups = decodeBuffers(await bufferResponse.arrayBuffer(), scene);
      for (let frame = 0; frame < scene.frame_count; frame += 1) {
        if (groups.every((group) => !group.presence || Array.from({ length: group.entity.slots }, (_, slot) => group.presence[slot * scene.frame_count + frame]).every((value) => value === 1))) return frame;
      }
      throw new Error("The browser fixture needs a frame with visible players and ball.");
    });
    await page.locator("#scrub").evaluate((scrub, frame) => {
      scrub.value = String(frame);
      scrub.dispatchEvent(new Event("input", { bubbles: true }));
    }, visibleFrame);
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.locator("#hud-frame").textContent(), String(visibleFrame));

    const pixels = await page.evaluate(() => {
      const canvas = document.getElementById("view");
      const gl = canvas.getContext("webgl2");
      const data = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
      let players = 0;
      let ball = 0;
      for (let i = 0; i < data.length; i += 4) {
        if ([[58, 111, 176], [209, 98, 63]].some((rgb) => rgb.every((value, axis) => Math.abs(data[i + axis] - value) < 8))) players += 1;
        if ([224, 120, 40].every((value, axis) => Math.abs(data[i + axis] - value) < 8)) ball += 1;
      }
      return { players, ball };
    });
    assert.ok(pixels.players > 40, JSON.stringify(pixels));
    assert.ok(pixels.ball > 0, JSON.stringify(pixels));
    await page.click("#to-start");
    await page.click("#next");
    assert.equal(await page.locator("#hud-frame").textContent(), "1");
    await page.click("#to-end");
    assert.equal(await page.locator("#scrub").inputValue(), await page.locator("#scrub").getAttribute("max"));
    await page.click("#follow");
    assert.equal(await page.locator("#follow").getAttribute("aria-pressed"), "true");
    await page.click("#trail");
    assert.equal(await page.locator("#trail").getAttribute("aria-pressed"), "false");
    await page.click("[data-preset='overhead']");
    await page.fill("#query", "no-such-clip");
    await page.waitForFunction(() => document.getElementById("result-count").textContent.startsWith("0 シーン"));
    await page.click("#clear");
    const first = await page.locator(".scene").first().textContent();
    await page.locator(".scene").first().click();
    await loaded();
    assert.equal(await page.locator("#scene-title").textContent(), first);
    await page.click("#play");
    await page.click("[data-preset='corner']");
    if (process.env.SLCS_REVIEW_SCREENSHOT) await page.screenshot({ path: process.env.SLCS_REVIEW_SCREENSHOT });
    await page.setViewportSize({ width: 720, height: 900 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    assert.deepEqual(errors, []);
    console.log(`SLCS review browser passed: ${JSON.stringify(pixels)}`);
  } finally {
    if (browser) await browser.close();
    if (server) server.kill("SIGTERM");
  }
})().catch((error) => { console.error(error); process.exit(1); });
