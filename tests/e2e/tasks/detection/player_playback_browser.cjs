// Opt-in local-data performance/overlay regression. Starts no inference jobs.
// Start ball review_dataset, then set PLAYWRIGHT_MODULE / DETECTION_URL as needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.DETECTION_URL || "http://127.0.0.1:8776";
const out = process.env.SCREENSHOT_DIR || "/tmp/ball-player-playback";
const duration = Number(process.env.PLAYBACK_DURATION_MS || 20000);
const samples = [
  {
    clip: "meiji/video_001/clip_019/cam1",
    mode: "reviewed",
    frame: 149,
    people: 2,
  },
  {
    clip: "meiji/video_001/clip_010/cam0",
    mode: "reviewed",
    frame: 129,
    people: 4,
  },
  {
    clip: "chat_annotation/hSFGMSfwiCE__b68cb7d349dc3aa9__f000001370-000002020",
    mode: "raw",
    frame: 74,
    people: 22,
  },
];
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(
      () => document.getElementById("empty").hidden,
      null,
      { timeout: 60000 },
    );
    const select = async (clip, mode, frame) => {
      await page.fill("#scene-search", clip);
      await page.locator(".scene").filter({ hasText: clip }).click();
      await page.waitForFunction(
        (clip) =>
          document.getElementById("view").dataset.scene?.endsWith(`::${clip}`),
        clip,
        { timeout: 60000 },
      );
      await page.selectOption("#player-mode", mode);
      await page.waitForFunction(
        (mode) => document.getElementById("view").dataset.playerMode === mode,
        mode,
        { timeout: 60000 },
      );
      await page.locator("#seek").evaluate((el, frame) => {
        el.value = String(frame);
        el.dispatchEvent(new Event("input"));
      }, frame);
      await page.waitForFunction(
        (frame) =>
          Number(document.getElementById("view").dataset.frame) === frame,
        frame,
        { timeout: 60000 },
      );
    };
    const measurements = [];
    for (const [i, sample] of samples.entries()) {
      await select(sample.clip, sample.mode, sample.frame);
      assert.equal(
        Number(await page.locator("#view").getAttribute("data-people")),
        sample.people,
      );
      const before = await page.locator("#view").screenshot();
      await page.uncheck("#show-players");
      assert.notDeepEqual(
        await page.locator("#view").screenshot(),
        before,
        "pose overlay changes canvas pixels",
      );
      await page.check("#show-players");
      await page.screenshot({ path: path.join(out, `overlay-${i}.png`) });
      await page.selectOption("#fps", "25");
      await page.evaluate(() => {
        window.presented = [];
        if (!window.playbackListener) {
          document
            .getElementById("view")
            .addEventListener("frame-presented", (event) =>
              window.presented.push(event.detail),
            );
          window.playbackListener = true;
        }
        performance.setResourceTimingBufferSize(10000);
        performance.clearResourceTimings();
      });
      await page.click("#play");
      await page.waitForTimeout(duration);
      await page.click("#play");
      const measurement = await page.evaluate(() => {
        const frames = window.presented;
        const total = Number(document.getElementById("seek").max) + 1;
        const intervals = frames
          .slice(1)
          .map((f, i) => f.at - frames[i].at)
          .sort((a, b) => a - b);
        const requests = performance.getEntriesByType("resource");
        return {
          updates: frames.length,
          fps: ((frames.length - 1) * 1000) / (frames.at(-1).at - frames[0].at),
          missing: frames
            .slice(1)
            .filter((f, i) => f.frame !== (frames[i].frame + 1) % total).length,
          interval_p95_ms: intervals[Math.floor(intervals.length * 0.95)],
          preview_requests: requests.filter((r) =>
            r.name.includes("/api/preview?"),
          ).length,
          player_requests: requests.filter((r) =>
            r.name.includes("/api/players?"),
          ).length,
        };
      });
      measurements.push({ ...sample, ...measurement });
      console.log(JSON.stringify(measurements.at(-1)));
      assert.ok(
        measurement.fps >= 24.5 && measurement.fps <= 25.5,
        `25fps contract: ${measurement.fps}`,
      );
      assert.equal(measurement.missing, 0);
      assert.ok(measurement.preview_requests < measurement.updates / 4);
      assert.ok(measurement.player_requests < measurement.updates / 4);
    }
    // Stall an annotation block. Display must wait at 31, then resume at 32,
    // without advancing labels ahead of RGB or skipping the delayed frames.
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    await page.route("**/api/players?*", async (route) => {
      if (new URL(route.request().url()).searchParams.get("start") === "32")
        await gate;
      await route.continue();
    });
    await select(samples[0].clip, "reviewed", 28);
    await page.evaluate(() => {
      window.presented = [];
    });
    await page.click("#play");
    await page.waitForFunction(
      () =>
        document.getElementById("view").dataset.frame === "31" &&
        document
          .getElementById("buffer-status")
          .textContent.includes("読み込み待ち"),
    );
    await page.waitForTimeout(250);
    assert.equal(await page.locator("#view").getAttribute("data-frame"), "31");
    release();
    await page.waitForFunction(
      () => Number(document.getElementById("view").dataset.frame) >= 37,
    );
    await page.click("#play");
    assert.deepEqual(
      await page.evaluate(() =>
        window.presented.slice(0, 9).map((f) => f.frame),
      ),
      [29, 30, 31, 32, 33, 34, 35, 36, 37],
    );
    await page.unroute("**/api/players?*");
    for (const sample of [
      { clip: "tracknet/game2/Clip5", status: "要確認", frame: 53, raw: true },
      {
        clip: "chat_annotation/hY1epQDmhGQ__b7f76f5507f3c345__f000008956-000009735",
        status: "レビュー待ち",
        frame: 377,
        raw: true,
      },
      {
        clip: "chat_annotation/-PyWNn6abXA__bbe89f372dea9715__f000002150-000002475",
        status: "未生成",
        frame: 49,
        raw: false,
      },
      {
        clip: "chat_annotation/-6UwVW0DeO4__f056b9d6649bee3a__f000001811-000002136",
        status: "対象外",
        frame: 0,
        raw: false,
      },
    ]) {
      await select(sample.clip, "reviewed", sample.frame);
      assert.match(
        await page.locator("#player-status").textContent(),
        new RegExp(sample.status),
      );
      assert.equal(
        await page.locator("#view").getAttribute("data-people"),
        "0",
      );
      await page.selectOption("#player-mode", "raw");
      await page.waitForFunction(
        () => document.getElementById("view").dataset.playerMode === "raw",
      );
      assert.equal(
        Number(await page.locator("#view").getAttribute("data-frame")),
        sample.frame,
        "mode switch preserves frame",
      );
      assert.equal(
        Number(await page.locator("#view").getAttribute("data-people")) > 0,
        sample.raw,
      );
    }
    for (const [width, height] of [
      [1440, 1000],
      [390, 844],
    ]) {
      await page.setViewportSize({ width, height });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      path.join(out, "measurements.json"),
      JSON.stringify({ duration_ms: duration, measurements, errors }, null, 2),
    );
    console.log(
      "PASS player overlays, status distinctions, frame synchronization and 25fps",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
