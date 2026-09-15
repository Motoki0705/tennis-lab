// Live local server smoke; inference is opt-in and uses the server GPU queue.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.DETECTION_URL || "http://127.0.0.1:8776";
const out = process.env.SCREENSHOT_DIR || "/tmp/detection-live";
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
      { timeout: 120000 },
    );
    const catalog = await (await page.request.get(`${url}/api/catalog`)).json();
    for (const dataset of catalog.datasets.filter(
      (d) => d.available && process.env.SKIP_DATASET_SWEEP !== "1",
    )) {
      const button = page
        .locator(".dataset")
        .filter({ has: page.locator(".name", { hasText: dataset.label }) })
        .first();
      await button.click();
      await page.waitForFunction(
        () => document.getElementById("empty").hidden,
        null,
        { timeout: 120000 },
      );
      assert.notEqual(await page.locator("#frame-name").textContent(), "");
      const rasterNames = await page
        .locator("#raster option")
        .evaluateAll((options) => options.map((o) => o.value).filter(Boolean));
      for (const name of rasterNames) {
        await page.selectOption("#raster", name);
        await page.screenshot({
          path: `${out}/${catalog.task}-${dataset.id.replace(/[^a-z0-9_-]/gi, "_")}-${name}.png`,
        });
      }
      console.log("rendered", dataset.id, rasterNames);
    }
    if (catalog.mode === "inference" && process.env.RUN_INFERENCE === "1") {
      const selected = process.env.CHECKPOINT_ID
        ? catalog.checkpoints.find((c) => c.id === process.env.CHECKPOINT_ID)
        : catalog.checkpoints.find(
            (c) => !c.error && c.compatible_datasets.length,
          );
      assert.ok(selected, "available compatible checkpoint");
      await page.selectOption("#checkpoint", selected.id);
      await page.waitForFunction(
        () => document.getElementById("empty").hidden,
        null,
        { timeout: 120000 },
      );
      if (process.env.DATASET_ID) {
        const dataset = catalog.datasets.find(
          (d) => d.id === process.env.DATASET_ID,
        );
        assert.ok(dataset);
        await page
          .locator(".dataset")
          .filter({ has: page.locator(".name", { hasText: dataset.label }) })
          .first()
          .click();
        await page.waitForFunction(
          () => document.getElementById("empty").hidden,
          null,
          { timeout: 120000 },
        );
      }
      if (process.env.START_FRAME)
        await page.fill("#start", process.env.START_FRAME);
      await page.selectOption(
        "#device",
        process.env.INFERENCE_DEVICE || "cuda",
      );
      const response = page.waitForResponse(
        (r) =>
          r.url().endsWith("/api/infer") && r.request().method() === "POST",
        { timeout: 1200000 },
      );
      await page.click("#infer");
      const result = await response;
      const data = await result.json();
      assert.equal(result.status(), 200, JSON.stringify(data));
      fs.writeFileSync(
        `${out}/${catalog.task}-result.json`,
        JSON.stringify(data),
      );
      await page.waitForFunction(
        () =>
          document.getElementById("status").textContent === "GT + Prediction",
        null,
        { timeout: 120000 },
      );
      console.log("inference", selected.id, data.metrics);
    }
    for (const [w, h] of [
      [1440, 1000],
      [390, 844],
    ]) {
      await page.setViewportSize({ width: w, height: h });
      await page.waitForTimeout(100);
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      await page.screenshot({
        path: `${out}/${catalog.task}-${catalog.mode}-${w}.png`,
        fullPage: true,
      });
    }
    assert.deepEqual(errors, []);
    console.log("PASS", catalog.task, catalog.mode);
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
