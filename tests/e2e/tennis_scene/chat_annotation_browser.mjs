// Read-only browser checks against a prepared annotation root.
// ANNOTATION_REVIEW_URL=http://127.0.0.1:8769 PLAYWRIGHT_MODULE=/.../playwright/index.mjs node ...
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
const { chromium } = await import(
  pathToFileURL(process.env.PLAYWRIGHT_MODULE).href
);
const base = process.env.ANNOTATION_REVIEW_URL;
const artifacts = process.env.ANNOTATION_REVIEW_ARTIFACTS || "outputs/chat_annotation_web_checks";
assert.ok(base, "ANNOTATION_REVIEW_URL is required");
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH
    ? { executablePath: process.env.CHROMIUM_PATH }
    : {}),
});
const page = await browser.newPage({ viewport: { width: 1540, height: 1080 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
await page.addInitScript(() => {
  window.previewURLs = { created: [], revoked: [] };
  const create = URL.createObjectURL.bind(URL),
    revoke = URL.revokeObjectURL.bind(URL);
  URL.createObjectURL = (blob) => {
    const url = create(blob);
    if (blob.type === "video/mp4") window.previewURLs.created.push(url);
    return url;
  };
  URL.revokeObjectURL = (url) => {
    window.previewURLs.revoked.push(url);
    revoke(url);
  };
});
try {
  await page.goto(base);
  await page.waitForSelector("#summary .metric", { timeout: 30000 });
  const catalog = await (await page.request.get(`${base}/api/catalog`)).json();
  assert.ok(catalog.summary.clips > 0);
  assert.equal(await page.locator("#summary .metric").count(), 4);
  const missing = catalog.clips.find(
    (r) => r.state === "missing" && r.video_available,
  );
  const complete = catalog.clips.find(
    (r) => r.state === "completed" && r.video_available,
  );
  assert.ok(
    missing && complete,
    "Fixture/root needs missing and completed clips",
  );
  await page.locator("#search").fill(missing.id);
  assert.equal(await page.locator("#clips tr").count(), 1);
  await page.locator("[data-open]").click();
  await page.waitForSelector("#versions select");
  assert.equal(await page.locator("#render-preview").isDisabled(), true);
  assert.ok(
    (await page.locator("#quality").textContent()).includes("JSONなし"),
  );
  await page.locator("#handoff-one").click();
  await page.waitForSelector("#handoff-dialog[open]");
  const request = await page.locator("#handoff-text").inputValue();
  assert.ok(request.includes(missing.id) && request.includes("JSON Schema"));
  assert.ok(request.includes("frame_ranges_half_open"));
  assert.ok(request.includes("対象は ball のみ"));
  await page.locator("#handoff-target").selectOption("player");
  const playerRequest = await page.locator("#handoff-text").inputValue();
  assert.ok(playerRequest.includes("対象は player のみ"));
  assert.ok(!playerRequest.includes("対象は ball のみ"));
  await page.locator("#close-handoff").click();
  await page.locator("#close-detail").click();
  await page.locator('[data-filter="all"]').click();
  await page.locator("#search").fill(complete.id);
  await page.locator("[data-open]").click();
  await page.waitForFunction(
    () => document.querySelector("#render-preview")?.disabled === false,
  );
  await page.locator("#video").evaluate(
    (video) =>
      new Promise((resolve) => {
        if (video.readyState >= 1) resolve();
        else video.addEventListener("loadedmetadata", resolve, { once: true });
      }),
  );
  const requestEvent = page.waitForResponse(
    (r) => r.url().endsWith("/preview"),
    { timeout: 120000 },
  );
  await page.locator("#render-preview").click();
  const response = await requestEvent;
  if (response.status() !== 200) {
    throw new Error(
      `Preview HTTP ${response.status()}: ${await response.text()}`,
    );
  }
  assert.equal(response.headers()["x-preview-storage"], "memory-only");
  assert.equal(response.headers()["cache-control"], "no-store");
  await page.waitForFunction(
    () =>
      document.querySelector("#video").src.startsWith("blob:") &&
      document.querySelector("#video").readyState >= 1,
    {},
    { timeout: 30000 },
  );
  assert.equal(
    await page.locator("#video").evaluate((v) => v.videoHeight),
    complete.height + 80,
  );
  assert.equal(await page.locator("#video-mode").textContent(), "一時overlay");
  await page.locator("#frame-index").fill("10");
  await page.locator("#frame-index").dispatchEvent("change");
  await page.waitForFunction(() =>
    document.querySelector("#frame-data").textContent.startsWith("frame 10 "),
  );
  await page.locator("#video").evaluate(async (v) => {
    await v.play();
  });
  await page.waitForTimeout(300);
  await page.locator("#video").evaluate((v) => v.pause());
  assert.ok(await page.locator("#video").evaluate((v) => v.currentTime > 0.4));
  const preview = await page.locator("#video").evaluate((v) => v.src);
  await mkdir(artifacts, { recursive: true });
  await page.screenshot({
    path: join(artifacts, "desktop.png"),
    fullPage: true,
  });
  await page.locator("#original").click();
  assert.ok(
    await page.evaluate(
      (url) => window.previewURLs.revoked.includes(url),
      preview,
    ),
  );
  await page.locator("#render-preview").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#video").src.startsWith("blob:") &&
      document.querySelector("#video").readyState >= 1,
    {},
    { timeout: 120000 },
  );
  const secondPreview = await page.locator("#video").evaluate((v) => v.src);
  await page.locator("#close-detail").click();
  assert.ok(
    await page.evaluate(
      (url) => window.previewURLs.revoked.includes(url),
      secondPreview,
    ),
  );
  await page.locator("[data-open]").click();
  await page.waitForFunction(
    () => document.querySelector("#render-preview")?.disabled === false,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: join(artifacts, "mobile.png"),
    fullPage: true,
  });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  );
  await page.locator("#close-detail").click();
  assert.equal(await page.locator("#detail").isHidden(), true);
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      clips: catalog.summary.clips,
      missing: missing.id,
      preview: complete.id,
      checks: [
        "filters",
        "missing-vs-reviewed",
        "handoff REQUEST",
        "memory MP4 playback",
        "frame seek",
        "blob revocation",
        "responsive layout",
        "no page errors",
      ],
    }),
  );
} finally {
  await browser.close();
}
