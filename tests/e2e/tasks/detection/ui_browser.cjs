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
    let mode = "inference";
    let holdPlayScene = null;
    let pendingPlay = null;
    let failPlay = false;
    const annotations = scene => Array.from({length: scene === "long" ? 10000 : 150}, (_, i) => {
      const reference = scene === "scene1" && i < 20;
      const absent = scene === "scene1" && (i >= 80 || (i >= 45 && i < 50));
      const unresolved = scene === "scene2" || i === 58;
      return {kinds: absent ? [] : [unresolved ? "unresolved" : i === 22 ? "interpolated" : "observed"],
        located_count: absent || unresolved ? 0 : 1, reviewed: true, is_target: !reference,
        evidence: !reference && !absent,
        exclusion_reasons: reference ? ["reference_only"] : absent ? ["no_ball"] : []};
    });
    const proposal = (scene) => scene === "long" ? {
      ...proposal("scene1"), scene, frames: 10000,
      timestamps: Array.from({length: 10000}, (_, i) => i / 30),
      play: [[0, 10000]], excluded: [], training: [[0, 10000]], presence: [[0, 10000]],
      bridged: [], annotations: annotations(scene),
      counts: {play: 10000, excluded: 0, training: 10000, windows: 624},
    } : ({
      scene, frames: 150, pose_approved: true,
      config: {window_length: 32, window_stride: 16, max_gap_seconds: 0.4,
        min_presence_fraction: 0.5, min_observed_frames: 8},
      timestamps: Array.from({length: 150}, (_, i) => i / 30),
      play: scene === "scene1" ? [[20, 80]] : [[0, 150]],
      excluded: scene === "scene1" ? [[0, 20], [80, 150]] : [],
      training: scene === "scene1" ? [[20, 80]] : [],
      presence: scene === "scene1" ? [[20, 45], [50, 80]] : [[0, 150]],
      bridged: scene === "scene1" ? [[45, 50]] : [],
      annotations: annotations(scene),
      counts: {play: scene === "scene1" ? 60 : 150,
        excluded: scene === "scene1" ? 90 : 0,
        training: scene === "scene1" ? 60 : 0, windows: scene === "scene1" ? 3 : 0},
    });
    const scenes = [
      { id: "scene1", label: "game1 / Clip1", frames: 150 },
      { id: "scene2", label: "game2 / Clip2", frames: 150 },
      { id: "long", label: "long clip / one frame without coordinates", frames: 10000 },
    ];
    await page.route("http://detection.test/**", async (route) => {
      const url = new URL(route.request().url());
      const p = url.pathname;
      const json = (body) => route.fulfill({ json: body });
      if (p === "/api/catalog")
        return json({
          task: "ball_detection",
          title: "Ball Detection",
          mode,
          play_intervals_available: mode === "review",
          cuda_available: true,
          datasets: [
            {
              id: "store/ball-mix-v2",
              label: "Ball store (ball-mix-v2)",
              path: "/data/ball_detection/ball-mix-v2",
              mode: "temporal",
              available: true,
              count: 2,
            },
            {
              id: "store/short-clips",
              label: "Ball store (short-clips)",
              path: "/data/ball_detection/short-clips",
              mode: "temporal",
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
              compatible_datasets: ["store/ball-mix-v2"],
              settings: { count: 8, threshold: 0.5 },
            },
          ],
          warnings: [],
        });
      if (p === "/api/scenes")
        return json({ items: scenes, total: scenes.length });
      if (p === "/api/play-intervals") {
        const scene = url.searchParams.get("scene");
        if (scene === holdPlayScene) {
          pendingPlay = {route, scene};
          return;
        }
        if (failPlay) return route.fulfill({status: 422, json: {detail: "snapshot mismatch"}});
        return json(proposal(scene));
      }
      if (p === "/api/preview") {
        const index = Number(url.searchParams.get("start"));
        const scene = url.searchParams.get("scene");
        const annotation = annotations(scene)[index];
        return json({
          scene: url.searchParams.get("scene"),
          label: "Clip",
          frames: scene === "long" ? 10000 : 150,
          start: index,
          width: 1280,
          height: 720,
          items: [
            {
              index,
              name: `frame_${index}.jpg`,
              gt: { points: mode === "review" && !annotation.located_count ? [] : [{ x: 600, y: 400, label: "b001" }], rasters: [] },
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
          "play_intervals.mjs",
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
    assert.equal(await page.locator("#play-intervals").isVisible(), false);
    mode = "review";
    await page.reload();
    await page.locator("#play-frame-state").waitFor();
    assert.equal(await page.locator(".play-track").count(), 4);
    assert.match(await page.locator("#play-frame-state").textContent(), /除外候補/);
    assert.match(await page.locator("#play-annotation-state").textContent(), /位置注釈 1個.*参照用フレーム/);
    assert.ok(await page.locator('.play-detail-row[data-row="annotations"] .play-cell[data-frame="0"]').evaluate(el => el.classList.contains("located")));
    assert.ok(await page.locator('.play-detail-row[data-row="evidence"] .play-cell[data-frame="0"]').evaluate(el => el.classList.contains("missing")));
    assert.ok(await page.locator('.play-detail-row[data-row="annotations"] .play-cell[data-frame="22"]').evaluate(el => el.classList.contains("estimated")));
    await page.getByRole("button", {name: "次の区間境界"}).click();
    await page.waitForFunction(() => document.getElementById("frame-name").textContent === "frame_20.jpg");
    assert.match(await page.locator("#play-frame-state").textContent(), /プレイ候補 · 教師窓内/);
    const track = page.locator(".play-track").first();
    let trackBox = await track.boundingBox();
    await track.click({position: {x: trackBox.width * 90.5 / 150, y: 8}});
    await page.waitForFunction(() => document.getElementById("frame-name").textContent === "frame_90.jpg");
    assert.match(await page.locator("#play-frame-state").textContent(), /除外候補 · 教師窓外/);
    await track.press("ArrowRight");
    await page.waitForFunction(() => document.getElementById("frame-name").textContent === "frame_91.jpg");
    await page.locator("#seek").evaluate((el) => {
      el.value = "45"; el.dispatchEvent(new Event("input"));
    });
    await page.waitForFunction(() => document.getElementById("play-frame-state").textContent.includes("frame 45"));
    assert.match(await page.locator("#play-frame-state").textContent(), /欠損を連結/);
    assert.equal(await track.getAttribute("aria-valuenow"), "45");
    await page.click("#play");
    await page.waitForFunction(() => Number(document.querySelector(".play-track").getAttribute("aria-valuenow")) > 45);
    await page.click("#play");
    holdPlayScene = "scene1";
    await page.locator(".scene").first().click();
    while (!pendingPlay) await page.waitForTimeout(10);
    assert.equal(await page.locator(".play-track").count(), 0);
    await page.locator(".scene").nth(1).click();
    await page.waitForFunction(() => document.getElementById("play-frame-state")?.textContent.includes("プレイ候補 · 教師窓外"));
    await pendingPlay.route.fulfill({json: proposal(pendingPlay.scene)});
    pendingPlay = null;
    holdPlayScene = null;
    await page.waitForTimeout(100);
    assert.equal(await page.locator(".play-segment.excluded").count(), 0);
    assert.match(await page.locator("#play-frame-state").textContent(), /プレイ候補 · 教師窓外/);
    failPlay = true;
    await page.locator(".scene").first().click();
    await page.waitForFunction(() => document.getElementById("play-intervals").textContent.includes("snapshot mismatch"));
    assert.equal(await page.locator(".play-track").count(), 0);
    failPlay = false;
    await page.locator(".scene").first().click();
    await page.locator("#play-frame-state").waitFor();
    // A one-frame gap is subpixel in the 10,000-frame overview but remains
    // a distinct >=16px cell in detail, even though evidence stays continuous.
    await page.locator(".scene").nth(2).click();
    await page.waitForFunction(() => document.getElementById("seek").max === "9999" && document.getElementById("play-frame-state"));
    await page.locator("#seek").evaluate(el => {el.value = "58"; el.dispatchEvent(new Event("input"));});
    await page.waitForFunction(() => document.getElementById("play-frame-state").textContent.startsWith("frame 58 "));
    assert.match(await page.locator("#play-annotation-state").textContent(), /位置注釈 0個（位置未確定）.*証拠 あり/);
    const gap = page.locator('.play-detail-row[data-row="annotations"] .play-cell[data-frame="58"]');
    assert.ok(await gap.evaluate(el => el.classList.contains("missing")));
    assert.ok((await gap.boundingBox()).width >= 16);
    const evidence = page.locator('.play-detail-row[data-row="evidence"] .play-cell[data-frame="58"]');
    assert.ok(await evidence.evaluate(el => el.classList.contains("evidence")));
    await page.locator('.play-detail-row[data-row="annotations"] .play-cell[data-frame="59"]').click();
    await page.waitForFunction(() => document.getElementById("frame-name").textContent === "frame_59.jpg");
    assert.match(await page.locator("#play-annotation-state").textContent(), /位置注釈 1個（実測）/);
    await page.getByRole("button", {name: "前の注釈変化"}).click();
    await page.waitForFunction(() => document.getElementById("play-frame-state").textContent.startsWith("frame 58 "));
    await page.selectOption("#play-detail-count", "16");
    assert.equal(await page.locator('.play-detail-row[data-row="annotations"] .play-cell').count(), 16);
    await page.selectOption("#play-detail-count", "64");
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
      const panel = await page.locator("#play-intervals").boundingBox();
      const center = await page.locator(".center").boundingBox();
      assert.ok(panel.y + panel.height <= center.y + center.height + 1, `timeline bounds ${width}`);
      assert.ok((await page.locator('.play-detail-row[data-row="annotations"] .play-cell[data-frame="58"]').boundingBox()).width >= 16, `one-frame detail ${width}`);
      await page.screenshot({
        path: path.join(outputDir, `detection-${width}.png`),
        fullPage: true,
      });
    }
    assert.deepEqual(errors, []);
    console.log(
      "PASS: image/GT, inference regression, play timeline seek/keyboard/playback, stale proposals/error handling, responsive bounds",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
