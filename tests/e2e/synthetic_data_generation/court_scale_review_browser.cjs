/* Read-only acceptance against explicitly published scale versions. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
  const base = process.env.COURT_REVIEW_URL || "http://127.0.0.1:8906";
  const screenshots = process.env.COURT_SCALE_SCREENSHOT_DIR || "tmp/court-scale-review/screenshots";
  fs.mkdirSync(screenshots, { recursive: true });
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox"],
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1520, height: 1000}});
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const scenes = await (await page.request.get(`${base}/api/scenes`)).json();
    const catalog = await (await page.request.get(`${base}/api/catalog`)).json();
    assert(catalog.length >= 20);
    assert.deepEqual(new Set(catalog.map(s => s.method)), new Set([
      "grid_budget_replay", "stratified_counter", "sobol", "capture_local", "open_bezier"]));
    await page.goto(base);
    await page.waitForSelector(".thumbnail img", {timeout: 90000});
    const versions = [...new Set(scenes.map(s => s.dataset_version))];
    for (const version of versions) {
      await page.selectOption("#method-filter", version);
      for (const selected of scenes.filter(s => s.dataset_version === version)) {
        await page.selectOption("#scene", selected.id);
        await page.waitForFunction(key => {
          const image = document.querySelector(".thumbnail img");
          return document.querySelector("#status").hidden && image?.complete && image.naturalWidth > 0 && image.src.includes(key);
        }, selected.id, {timeout: 90000});
        const response = await page.request.get(`${base}/api/scenes/${selected.id}?revision=${selected.revision}`);
        assert.equal(response.status(), 200);
        const summary = await response.json();
        assert.equal(summary.scene_id, selected.scene_id);
        assert.equal(summary.scale.method, selected.method);
        assert(summary.groups.some(g => g.samples.length));
        for (const group of summary.groups) {
          if (group.closed === false && group.planned_points.length) {
            assert.notDeepEqual(group.planned_points[0], group.planned_points.at(-1));
          }
        }
        const empty = summary.groups.find(g => !g.samples.length);
        if (empty) {
          await page.locator(`.trajectory[data-group="${empty.id}"]`).click();
          assert.equal(await page.locator(".thumbnail").count(), 0);
          assert.match(await page.locator("#sample-info").textContent(), /採択0件/);
          await page.locator(`.trajectory[data-group="${summary.groups.find(g => g.samples.length).id}"]`).click();
        }
        await page.locator(".thumbnail").first().click();
        await page.waitForFunction(() => document.querySelectorAll("#point-table tr").length === 14);
        assert.equal(await page.locator("#label-scope").inputValue(), "target");
        await page.selectOption("#large-mode", "compare");
        await page.waitForFunction(() => ["large-image", "raw-image"].every(id => {
          const img = document.getElementById(id); return img.complete && img.naturalWidth > 0;
        }));
        await page.keyboard.press("Escape");
        await page.selectOption("#image-mode", "overlay");
        if (selected.scene_id === "B00") {
          await page.screenshot({path: path.join(screenshots, `${selected.method}-${version.slice(0, 10)}.png`)});
        }
        console.log(JSON.stringify({method: selected.method, scene: selected.scene_id,
          candidates: summary.scale.counts.candidate_unique, accepted: summary.metrics.accepted_frame_count,
          empty_trajectories: summary.scale.empty_trajectory_count, status: "passed"}));
      }
    }
    // A delayed response from a previous version must never repopulate the new selection.
    const old = scenes.find(s => s.method === "stratified_counter" && s.scene_id === "B00");
    const current = scenes.find(s => s.method === "sobol" && s.scene_id === "B00");
    await page.route(`**/api/scenes/${old.id}?*`, async route => {
      const response = await route.fetch();
      await new Promise(resolve => setTimeout(resolve, 600));
      await route.fulfill({response}).catch(() => {});
    });
    const pending = page.waitForRequest(r => r.url().includes(`/api/scenes/${old.id}?`));
    await page.selectOption("#method-filter", old.dataset_version);
    await page.selectOption("#scene", old.id);
    await pending;
    await page.selectOption("#method-filter", current.dataset_version);
    await page.waitForFunction(key => document.querySelector(".thumbnail img")?.src.includes(key), current.id, {timeout: 90000});
    await page.waitForTimeout(900);
    assert((await page.locator(".thumbnail img").first().getAttribute("src")).includes(current.id));
    assert.deepEqual(errors, []);
    console.log("method/scene selection, target teachers, empty paths, open endpoints and stale-response rejection: passed");
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exit(1);});
