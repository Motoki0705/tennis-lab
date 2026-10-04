/* Shared lifecycle contract, against any running real dataset review app. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.DATASET_REVIEW_URL;
if (!url) throw new Error("DATASET_REVIEW_URL must point to a running review app");
(async () => {
  const browser = await chromium.launch({headless:true,args:["--no-sandbox"]});
  try {
    const page = await browser.newPage();
    await page.addInitScript(() => {
      window.reviewEvents = [];
      window.addEventListener("dataset-review:scene", (event) => window.reviewEvents.push({
        phase:event.detail.phase, hasView:"view" in event.detail,
        publicOrbit:typeof event.detail.view?.setOrbit === "function",
        modelReady:Boolean(event.detail.view?.model),
      }));
    });
    await page.goto(url);
    await page.waitForFunction(() => window.reviewEvents.some((e) => e.phase === "loaded"));
    const events = await page.evaluate(() => window.reviewEvents);
    assert.deepEqual(events.find((e) => e.phase === "loading"), {phase:"loading",hasView:false,publicOrbit:false,modelReady:false});
    assert.deepEqual(events.find((e) => e.phase === "loaded"), {phase:"loaded",hasView:true,publicOrbit:true,modelReady:true});
    await page.route("**/api/scene?**", (route) => route.fulfill({status:422,contentType:"application/json",body:JSON.stringify({detail:"event contract test"})}));
    await page.locator(".scene").first().click();
    await page.waitForFunction(() => window.reviewEvents.some((e) => e.phase === "error"));
    assert.deepEqual(await page.evaluate(() => window.reviewEvents.find((e) => e.phase === "error")), {phase:"error",hasView:false,publicOrbit:false,modelReady:false});
    console.log("Shared dataset review lifecycle: loaded public view; loading/error without view passed");
  } finally { await browser.close(); }
})().catch((error) => {console.error(error);process.exit(1)});
