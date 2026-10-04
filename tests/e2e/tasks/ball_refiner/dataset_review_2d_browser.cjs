/* Explicit real-data browser check. No fixture generation or model execution. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const modulePath = process.env.PLAYWRIGHT_CORE_PATH;
if (!modulePath) throw new Error("Set PLAYWRIGHT_CORE_PATH to the installed playwright-core module");
const { chromium } = require(modulePath);
const [baseURL, output] = process.argv.slice(2);
if (!baseURL || !output) throw new Error("Usage: node dataset_review_2d_browser.cjs <running-review-URL> <capture-directory>");

(async () => {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1250 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  const shots = [];
  async function open(clip, frame, condition = "observed") {
    await page.goto(`${baseURL}/?${new URLSearchParams({ clip, frame: String(frame), condition })}`, { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.body.dataset.ready === "true" && document.body.dataset.busy === "false");
    assert.equal(await page.locator("#error").isVisible(), false, await page.locator("#error").textContent());
    const response = await page.request.get(`${baseURL}/api/frame?${new URLSearchParams({ clip, frame: String(frame), condition })}`);
    assert.equal(response.status(), 200);
    const data = await response.json();
    assert.equal(data.clip_id, clip);
    assert.equal(data.frame, frame);
    return data;
  }
  async function capture(name, data, note) {
    await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });
    shots.push({ name, note, kind: "new", url: page.url(), clip_id: data.clip_id, source: data.source,
      split: data.split, camera: data.camera, frame: data.frame, pts: data.pts, condition: data.condition,
      viewport: { width: 1600, height: 1250, device_scale_factor: 1 }, source_size_wh: data.source_size_wh,
      stored_size_wh: data.stored_size_wh, rgb: data.rgb, target: data.target,
      gap_active: data.gap_active, context_available: data.context !== null,
      context_used_by_gmm: data.context?.used_by_saved_gmm, provenance: data.provenance });
  }
  try {
    const meiji = "meiji/video_000/clip_001/cam1";
    let data = await open(meiji, 148);
    assert.equal(data.target.reason, "observed");
    assert.equal(data.rgb.available, true);
    assert.equal(data.context.court_valid.filter(Boolean).length, 14);
    assert.equal(data.context.used_by_saved_gmm, false);
    assert.equal(data.effective_candidates.length, 8);
    await capture("01-observed-context", data, "候補・観測教師・旧比較pose/court・保存GMMを同frameで確認");

    data = await open(meiji, 148, "evidence_gap");
    assert.equal(data.gap_active, true);
    assert.equal(data.effective_candidates.length, 0);
    assert.equal(data.candidates.length, 8);
    assert.equal(await page.locator(".gap-disabled").count(), 8);
    await capture("02-artificial-evidence-gap", data, "同clip/frameの人工候補dropout。保存patchは参考で実効入力0件");

    data = await open(meiji, 63);
    assert.equal(data.target.reason, "interpolated");
    assert.equal(data.target.position_valid, false);
    assert.equal(data.target.presence_valid, false);
    assert.notEqual(data.target.uv, null);
    await capture("03-estimated-reference", data, "補間参考位置は橙の◇。教師maskは位置/存在とも0");

    data = await open(meiji, 66);
    assert.equal(data.target.reason, "unresolved");
    assert.equal(data.target.presence, null);
    await capture("04-unknown-presence", data, "unknownを不存在の負例にしない。保存GMMの存在確率は別欄");

    const absent = "chat_annotation/2Fa16bdg8pI__403208cbe9abb73c__f000001212-000001537";
    data = await open(absent, 234);
    assert.equal(data.target.reason, "out_of_frame");
    assert.equal(data.target.position_valid, false);
    assert.equal(data.target.presence_valid, true);
    assert.equal(data.target.presence, false);
    await capture("05-explicit-out-of-frame", data, "明示画面外: 位置mask0 / 存在mask1 / 存在教師0");

    // Actual frame controls, playback, toggles, state navigation and filtering.
    await open(meiji, 148);
    await page.locator("#next").click();
    await page.waitForFunction(() => document.querySelector("#scene-title").textContent.endsWith("frame 149") && document.body.dataset.busy === "false");
    await page.locator("#previous").click();
    await page.waitForFunction(() => document.querySelector("#scene-title").textContent.endsWith("frame 148") && document.body.dataset.busy === "false");
    await page.locator("#show-context").uncheck();
    await page.locator("#show-context").check();
    await page.locator("#show-rgb").uncheck();
    assert.match(await page.locator("#rgb-badge").textContent(), /座標面/);
    await page.locator("#show-rgb").check();
    await page.locator("#jump").selectOption("estimated");
    await page.waitForFunction(() => document.querySelector("#teacher-state").textContent === "補間位置" && document.body.dataset.busy === "false");
    await page.locator("#play").click();
    await page.waitForTimeout(500);
    await page.locator("#play").click();
    assert.equal(await page.locator("#play").textContent(), "再生");
    await page.locator("#source").selectOption("tracknet");
    await page.locator("#search").fill("game1/Clip1");
    await page.waitForFunction(() => document.querySelector("#teacher-state").textContent === "未提供" && document.body.dataset.busy === "false");
    assert.match(await page.locator("#teacher-description").textContent(), /unknown教師とは別/);
    assert.equal(await page.locator("#condition").evaluate(select => select.options[1].disabled), true);
    await page.locator("#search").fill("no-such-real-clip");
    assert.equal(await page.locator(".review-grid").isVisible(), false);
    assert.equal(errors.length, 0, errors.join("\n"));
    await fs.writeFile(path.join(output, "browser_capture.json"), JSON.stringify({ status: "passed", screenshot_type: "real browser / real local caches",
      no_before_reason: "New cache/teacher review UI; existing component renderer is not a comparable browser screen",
      browser_version: browser.version(), captured_at: new Date().toISOString(), screenshots: shots,
      checks: ["same-frame observed/evidence_gap", "observed/estimated/unknown/absent masks", "verified RGB", "actual playback/steps",
        "layer toggles", "state jump", "source/search filtering", "missing teacher distinct from unknown", "no browser errors"] }, null, 2) + "\n");
    console.log(`Passed real-data browser checks; ${shots.length} screenshots captured in ${output}`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
