/* Run against the local PLCS inference server; no inference or GPU is used. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
    args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(process.env.PLCS_INFERENCE_URL || "http://127.0.0.1:8771");
    await page.locator(".family").first().click();
    await page.locator(".scene").first().click();
    await page.waitForFunction(() => Number(document.querySelector("#scrub").max) > 0);
    await page.waitForTimeout(700);

    await page.locator("#scrub").evaluate((input) => {
      input.value = "100";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.locator("#play").click();
    await page.waitForTimeout(100);
    await page.locator("#play").click();
    const afterSeek = Number(await page.locator("#scrub").inputValue());
    assert.ok(afterSeek >= 100 && afterSeek < 200, `seek resumed at ${afterSeek}`);
    await page.locator("#next").click();
    const stepped = Number(await page.locator("#scrub").inputValue());
    await page.locator("#play").click();
    await page.waitForTimeout(100);
    await page.locator("#play").click();
    const afterStep = Number(await page.locator("#scrub").inputValue());
    assert.ok(afterStep >= stepped && afterStep < stepped + 100,
      `frame step resumed at ${afterStep}, expected >= ${stepped}`);

    const canvas = page.locator("canvas");
    const digest = () => canvas.evaluate((node) => node.toDataURL());
    const before = await digest();
    assert.ok(before.length > 10000, "rendered scene must not be blank");
    const box = await canvas.boundingBox();
    const x = box.x + box.width / 2;
    const y = box.y + box.height / 2;
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 100, y + 45, { steps: 12 });
    await page.mouse.up();
    await page.waitForTimeout(700);
    const rotated = await digest();
    assert.notEqual(rotated, before, "orbit drag must change the rendered view");
    await page.mouse.wheel(0, -250);
    await page.waitForTimeout(700);
    assert.notEqual(await digest(), rotated, "wheel must change the rendered view");

    for (const width of [390, 800, 1440]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 900 });
      await page.waitForTimeout(250);
      const clipped = await page.evaluate(() => {
        const transport = document.querySelector("#transport").getBoundingClientRect();
        return [...document.querySelectorAll("#transport button, #transport input, #clock")]
          .filter((element) => {
            const box = element.getBoundingClientRect();
            return box.left < transport.left - 1 || box.right > transport.right + 1
              || box.top < transport.top - 1 || box.bottom > transport.bottom + 1;
          }).map((element) => element.id || element.textContent.trim());
      });
      assert.deepEqual(clipped, [], `clipped transport controls at ${width}px`);
      if (width === 390 && process.env.SCREENSHOT_PATH) {
        await page.screenshot({ path: process.env.SCREENSHOT_PATH });
      }
    }
    // Hold a mocked prediction while another scene's real GT preview loads.
    // No prediction request reaches the server or starts a GPU job.
    let releaseFirst;
    let firstStarted;
    let calls = 0;
    const held = new Promise((resolve) => { releaseFirst = resolve; });
    const started = new Promise((resolve) => { firstStarted = resolve; });
    await page.route("**/api/predict", async (route) => {
      calls += 1;
      if (calls === 1) {
        firstStarted();
        await held;
        await route.fulfill({ status: 200, contentType: "application/octet-stream", body: Buffer.alloc(4) });
      } else {
        await route.fulfill({ status: 422, json: { detail: "mock inference failure" } });
      }
    });
    await page.locator(".checkpoint:not(:disabled)").first().click();
    await page.locator(".family:not(:disabled)").first().click();
    await page.locator(".scene").first().click();
    await page.waitForFunction(() => !document.querySelector("#run").disabled);
    await page.locator("#run").click();
    await started;
    const nextScene = await page.locator(".scene").nth(1).getAttribute("data-scene");
    await page.locator(".scene").nth(1).click();
    await page.waitForFunction((id) => document.querySelector("#scene-title").textContent === id
      && Number(document.querySelector("#scrub").max) > 0
      && document.querySelector("#status").hidden, nextScene);
    releaseFirst();
    await page.waitForFunction(() => !document.querySelector("#run").disabled);
    assert.equal(await page.locator("#scene-title").textContent(), nextScene);
    assert.ok(Number(await page.locator("#scrub").getAttribute("max")) > 0);
    await page.locator("#run").click();
    await page.waitForFunction(() => !document.querySelector("#run").disabled
      && document.querySelector("#status").textContent.includes("mock inference failure"));
    assert.equal(calls, 2, "a new inference must be possible after the scene switch");
    assert.deepEqual(errors, []);
    console.log("PLCS orbit/zoom, transport bounds, seek resume and in-flight scene switch passed");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
