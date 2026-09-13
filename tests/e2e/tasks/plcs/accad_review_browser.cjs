/* Read-only browser regression for the ACCAD motion review UI. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_PATH
      ? { executablePath: process.env.CHROMIUM_PATH }
      : {}),
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 900 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    const loaded = () =>
      page.waitForFunction(
        () => {
          const status = document.getElementById("status");
          const transport = document.getElementById("transport");
          return status.hidden && !transport.hidden;
        },
        {},
        { timeout: 90000 },
      );

    await page.goto(process.env.ACCAD_REVIEW_URL || "http://127.0.0.1:8769");
    await page.waitForSelector(".motion", { timeout: 90000 });
    await loaded();

    assert.equal(await page.locator(".subject").count(), 20);
    assert.equal(await page.locator(".motion").count(), 252);

    // The canvas must actually contain the drawn skeleton, not just background.
    const painted = await page.evaluate(() => {
      const canvas = document.getElementById("view");
      const context = canvas.getContext("2d");
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
      const near = (value, target) => Math.abs(value - target) < 26;
      let left = 0;
      let right = 0;
      for (let index = 0; index < data.length; index += 4) {
        const [r, g, b] = [data[index], data[index + 1], data[index + 2]];
        if (near(r, 0x11) && near(g, 0x82) && near(b, 0x7b)) left += 1;
        if (near(r, 0xd1) && near(g, 0x62) && near(b, 0x3f)) right += 1;
      }
      return { left, right };
    });
    assert.ok(painted.left > 200, `left limb pixels: ${painted.left}`);
    assert.ok(painted.right > 200, `right limb pixels: ${painted.right}`);

    await page.fill("#query", "Walk B10");
    await page.waitForTimeout(250);
    assert.equal(await page.locator(".motion").count(), 3);
    assert.match(await page.locator("#result-count").textContent(), /3 件/);
    await page.locator(".motion").first().click();
    await loaded();
    assert.match(await page.locator("#motion-title").textContent(), /walk turn left/i);
    assert.match(await page.locator("#hud-x").textContent(), /m$/);
    assert.match(await page.locator("#clock").textContent(), /^\d+ \/ \d+$/);

    await page.click("#follow");
    assert.equal(await page.getAttribute("#follow", "aria-pressed"), "true");
    await page.click("#trail");
    assert.equal(await page.getAttribute("#trail", "aria-pressed"), "false");
    await page.click("#reset");
    await page.click("#play");
    await page.waitForTimeout(250);
    assert.equal(await page.getAttribute("#play", "title"), "再生");

    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    assert.ok(
      overflow.scrollWidth <= overflow.clientWidth + 1,
      `horizontal overflow: ${JSON.stringify(overflow)}`,
    );
    assert.deepEqual(errors, []);
    console.log("accad motion review browser regression passed");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
