// Deterministic UI regression with real source image pixels and mocked inference.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname, "../../../..");
const staticRoot = path.join(
  root,
  "src/tasks/base/visualization/detection/static",
);
const imagePath =
  process.env.DETECTION_IMAGE ||
  "/home/kamimura/projects/tennis-lab/data/court/images/PuXlxKdUIes_2450.png";
const outputDir = process.env.SCREENSHOT_DIR || "/tmp/detection-ui";
fs.mkdirSync(outputDir, { recursive: true });
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
    let pending = null;
    let inferCount = 0;
    const scenes = [
      { id: "scene1", label: "game1 / Clip1", frames: 150 },
      { id: "scene2", label: "game2 / Clip2", frames: 150 },
    ];
    await page.route("http://detection.test/**", async (route) => {
      const url = new URL(route.request().url());
      const p = url.pathname;
      const json = (body) => route.fulfill({ json: body });
      if (p === "/api/catalog")
        return json({
          task: "ball_detection",
          title: "Ball Detection",
          mode: "inference",
          cuda_available: true,
          datasets: [
            {
              id: "tracknet",
              label: "tennis / tracknet",
              path: "/data/tennis/tracknet",
              available: true,
              count: 2,
            },
            {
              id: "web",
              label: "tennis / web",
              path: "/data/tennis/web",
              available: true,
              count: 1,
            },
          ],
          checkpoints: [
            {
              id: "test.ckpt",
              label: "run-epoch13.ckpt",
              path: "ckpt/ball_detection/run-epoch13.ckpt",
              model: "conv_next_unet",
              compatible_datasets: ["tracknet"],
              settings: { count: 8, threshold: 0.5 },
            },
          ],
          warnings: [],
        });
      if (p === "/api/scenes")
        return json({ items: scenes, total: scenes.length });
      if (p === "/api/preview") {
        const index = Number(url.searchParams.get("start"));
        return json({
          scene: url.searchParams.get("scene"),
          label: "Clip",
          frames: 150,
          start: index,
          width: 1280,
          height: 720,
          items: [
            {
              index,
              name: `frame_${index}.jpg`,
              gt: { points: [{ x: 600, y: 400, label: "b001" }], rasters: [] },
            },
          ],
          warnings: [],
        });
      }
      if (p === "/api/image")
        return route.fulfill({
          contentType: "image/png",
          body: fs.readFileSync(imagePath),
        });
      if (p === "/api/infer") {
        inferCount++;
        pending = { route, request: route.request().postDataJSON() };
        return;
      }
      const file = p === "/" ? "index.html" : p.replace("/static/", "");
      if (
        ![
          "index.html",
          "app.js",
          "viewer.mjs",
          "icons.mjs",
          "style.css",
        ].includes(file)
      )
        return route.fulfill({ status: 404, body: "" });
      return route.fulfill({
        contentType: file.endsWith(".html")
          ? "text/html"
          : file.endsWith(".css")
            ? "text/css"
            : "text/javascript",
        body: fs.readFileSync(path.join(staticRoot, file)),
      });
    });
    await page.goto("http://detection.test/");
    await page.locator(".scene").first().waitFor();
    assert.equal(await page.locator(".dataset").count(), 2);
    await page.selectOption("#checkpoint", "test.ckpt");
    await page.locator(".scene").first().click();
    await page.waitForFunction(() => document.getElementById("empty").hidden);
    assert.equal(await page.locator(".dataset").count(), 1);
    const canvas = page.locator("#view");
    const image1 = await canvas.screenshot();
    await page.click("#zoom-in");
    assert.notDeepEqual(await canvas.screenshot(), image1);
    const box = await canvas.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      box.x + box.width / 2 + 50,
      box.y + box.height / 2 + 40,
    );
    await page.mouse.up();
    assert.notDeepEqual(await canvas.screenshot(), image1);
    await page.click("#fit");
    await page.locator("#seek").evaluate((el) => {
      el.value = "100";
      el.dispatchEvent(new Event("input"));
    });
    await page.waitForFunction(
      () =>
        document.getElementById("frame-name").textContent === "frame_100.jpg",
    );
    await page.click("#play");
    await page.waitForFunction(
      () => Number(document.getElementById("seek").value) > 100,
    );
    await page.click("#play");
    await page.click("#infer");
    await page.waitForFunction(() => document.getElementById("infer").disabled);
    while (!pending) await page.waitForTimeout(10);
    await page.locator(".scene").nth(1).click();
    await pending.route.fulfill({
      json: {
        scene: pending.request.scene,
        start: 0,
        items: [{ index: 0, pred: { points: [], rasters: [] } }],
        metrics: { error: 0 },
        warnings: [],
      },
    });
    pending = null;
    await page.waitForFunction(
      () => !document.getElementById("infer").disabled,
    );
    assert.equal(
      await page.locator("#scene-title").textContent(),
      "game2 / Clip2",
    );
    await page.click("#infer");
    while (!pending) await page.waitForTimeout(10);
    await pending.route.fulfill({
      json: {
        scene: pending.request.scene,
        start: 0,
        items: [
          {
            index: 0,
            pred: { points: [{ x: 610, y: 403, label: "pred" }], rasters: [] },
          },
        ],
        metrics: { distance_px: 10.44 },
        warnings: [],
      },
    });
    pending = null;
    await page.waitForFunction(
      () => document.getElementById("status").textContent === "GT + Prediction",
    );
    assert.equal(inferCount, 2);
    for (const [width, height] of [
      [1440, 1000],
      [800, 1000],
      [390, 844],
    ]) {
      await page.setViewportSize({ width, height });
      await page.waitForTimeout(100);
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        true,
        `overflow ${width}`,
      );
      const bounds = await page.locator(".transport").evaluate((el) =>
        [...el.children].map((child) => {
          const p = el.getBoundingClientRect(),
            r = child.getBoundingClientRect();
          return (
            r.left >= p.left &&
            r.right <= p.right + 1 &&
            r.top >= p.top &&
            r.bottom <= p.bottom + 1
          );
        }),
      );
      assert.ok(bounds.every(Boolean), `transport bounds ${width}`);
      await page.screenshot({
        path: path.join(outputDir, `detection-${width}.png`),
        fullPage: true,
      });
    }
    assert.deepEqual(errors, []);
    console.log(
      "PASS: source image, zoom/pan, filtering, playback seek, inference selection race, retry, responsive bounds",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
