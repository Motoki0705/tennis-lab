// Real local ball-store dataset review. Never launches inference or modifies data.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.DETECTION_URL || "http://127.0.0.1:8776";
const out = process.env.SCREENSHOT_DIR || "/tmp/ball-dataset-review";
const samples = [
  { name: "interpolated", clip: "meiji/video_000/clip_000/cam0", frame: 21, state: "interpolated", supervision: "reference" },
  { name: "unresolved", clip: "meiji/video_000/clip_000/cam0", frame: 86, state: "unresolved", supervision: "reference" },
  { name: "estimated", clip: "tracknet/game1/Clip1", frame: 83, state: "occlusion_estimated", supervision: "reference" },
];
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH || chromium.executablePath(),
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    let inferenceRequests = 0;
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (request) => { if (request.url().endsWith("/api/infer")) inferenceRequests++; });
    await page.goto(url);
    await page.waitForFunction(() => document.getElementById("empty").hidden, null, {timeout: 60000});
    const catalog = await (await page.request.get(`${url}/api/catalog`)).json();
    const dataset = catalog.datasets.find((d) => d.id === "store/ball-mix-v2");
    assert.ok(dataset?.available, "requires the real ball-mix-v2 store");
    await page.click("#dataset-overview-open");
    const overview = await page.locator("#dataset-overview").textContent();
    for (const value of [dataset.overview.clips, dataset.overview.counts.frames])
      assert.ok(overview.includes(value.toLocaleString("ja-JP")));
    assert.match(overview, /frame数|フレーム/);
    assert.match(overview, /instance数/);
    assert.match(overview, /ボールの教師とは別/);
    await page.screenshot({path: path.join(out, "after-overview.png")});
    await page.click("#dataset-overview-close");
    const select = async (sample) => {
      await page.fill("#scene-search", sample.clip);
      await page.locator(".scene").filter({hasText: sample.clip}).first().click();
      await page.waitForFunction((clip) => document.getElementById("view").dataset.scene?.endsWith(`::${clip}`), sample.clip, {timeout: 60000});
      await page.locator("#seek").evaluate((el, frame) => {
        el.value = String(frame); el.dispatchEvent(new Event("input"));
      }, sample.frame);
      await page.waitForFunction((frame) => document.getElementById("view").dataset.frame === String(frame), sample.frame, {timeout: 60000});
    };
    const findings = [];
    for (const sample of samples) {
      await select(sample);
      assert.equal(await page.locator("#view").getAttribute("data-supervision"), sample.supervision);
      const preview = await (await page.request.get(`${url}/api/preview?${new URLSearchParams({scene: `store/ball-mix-v2::${sample.clip}`, start: sample.frame, count: 1})}`)).json();
      const item = preview.items[0];
      assert.ok(item.review.point_kinds.includes(sample.state));
      assert.match(await page.locator("#frame-supervision").textContent(), /採点対象外/);
      await page.screenshot({path: path.join(out, `after-${sample.name}.png`)});
      if (sample.state === "unresolved")
        assert.ok(item.gt.points.filter((p) => p.state === "unresolved").every((p) => p.visible === false));
      const review = await (await page.request.get(`${url}/api/review?${new URLSearchParams({scene: `store/ball-mix-v2::${sample.clip}`})}`)).json();
      await page.selectOption("#review-jump-state", sample.state);
      const next = review.positions[sample.state].find((f) => f > sample.frame);
      if (next !== undefined) {
        await page.click("#review-jump-next");
        await page.waitForFunction((frame) => document.getElementById("view").dataset.frame === String(frame), next);
      }
      findings.push({...sample, annotation: item.review, next});
    }
    await page.fill("#scene-search", "");
    await page.selectOption("#source-filter", "meiji");
    await page.selectOption("#split-filter", "val");
    await page.selectOption("#review-state-filter", "unresolved");
    await page.waitForFunction(() => document.getElementById("view").dataset.scene?.includes("::meiji/video_000/") && document.getElementById("frame-point-kinds").textContent.includes("位置不明"), null, {timeout: 60000});
    assert.match(await page.locator("#scene-review-meta").textContent(), /Meiji · val/);
    const result = await (await page.request.get(`${url}/api/scenes?${new URLSearchParams({dataset: dataset.id, source: "meiji", split: "val", review_state: "unresolved"})}`)).json();
    assert.equal(Number((await page.locator("#scene-count").textContent()).replaceAll(",", "")), result.total);
    for (const [width, height] of [[1440, 1000], [390, 844]]) {
      await page.setViewportSize({width, height});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.click("#dataset-overview-open");
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.click("#dataset-overview-close");
    }
    assert.equal(inferenceRequests, 0);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(out, "dataset-review-browser.json"), JSON.stringify({url, viewport: {width:1440,height:1000}, dataset: dataset.id, findings, filter_count: result.total, errors, inference_requests: inferenceRequests}, null, 2));
    console.log("PASS real dataset composition, supervision states, filters, jumps, mobile bounds; no inference");
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
