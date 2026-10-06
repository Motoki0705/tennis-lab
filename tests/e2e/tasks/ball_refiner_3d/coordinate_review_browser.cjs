/* Real local dataset + trained checkpoints. Writes screenshots outside source. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const url = process.env.REFINER_REVIEW_URL || "http://127.0.0.1:8787";
  const out = process.env.REFINER_REVIEW_ARTIFACTS;
  assert.ok(out, "REFINER_REVIEW_ARTIFACTS must name an isolated output directory");
  await fs.mkdir(out, {recursive: true});
  const browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    args: ["--no-sandbox", "--enable-unsafe-swiftshader"]});
  const page = await browser.newPage({viewport: {width: 1600, height: 1100}, deviceScaleFactor: 1});
  const errors = [], captured = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("response", async response => {
    if (/\/api\/(saved|preview|infer)$/.test(response.url()) && response.ok()) {
      captured.push(await response.json());
    }
  });
  const waitFor = async source => page.waitForFunction(value => document.querySelector("#source").textContent.includes(value)
    && !document.querySelector("#source").classList.contains("busy"), source, {timeout: 90000});
  const shot = async name => { await page.waitForTimeout(200); await page.screenshot({path: path.join(out, name + ".png"), fullPage: true}); };
  const digest = locator => locator.evaluate(canvas => canvas.toDataURL());
  try {
    await page.goto(url);
    await waitFor("保存済み評価");
    assert.equal(await page.locator("#model-2d").count(), 0);
    assert.equal(await page.locator("#event-graph").count(), 1);
    assert.match(await page.locator("#model-3d-info").textContent(), /REGRESSION|GAN|FLOW/);
    assert.equal(await page.locator("#view-2d canvas").count(), 1);
    assert.equal(await page.locator("#view-3d canvas").count(), 1);
    assert.ok((await digest(page.locator("#view-3d canvas"))).length > 15000);
    await shot("01-saved-desktop");
    const initial = captured.find(item => item.source === "saved");
    assert.ok(initial, "the actual trained saved predictions must be used");
    assert.equal(initial.scene.gt_3d.length, initial.scene.prediction_3d.length);
    assert.equal(initial.scene.gt_2d.length, 4);
    const baselineInput = initial.input_sha256;

    await page.locator("#next-event").click();
    assert.ok(Number(await page.locator("#scrub").inputValue()) > 0);
    const eventFrame = Number(await page.locator("#scrub").inputValue());
    assert.ok(initial.scene.events[eventFrame]);
    await page.locator("#camera").selectOption("2");
    assert.match(await page.locator("#camera-label").textContent(), /3/);
    await page.locator("#play").click();
    await page.waitForTimeout(150);
    await page.locator("#play").click();
    assert.ok(Number(await page.locator("#scrub").inputValue()) >= eventFrame);
    await page.locator("#scrub").evaluate(element => {element.value="150";element.dispatchEvent(new Event("input",{bubbles:true}));});
    await page.locator("#plot-mode").selectOption("speed");
    await shot("02-event-speed");

    const worldCanvas = page.locator("#view-3d canvas");
    const beforeOrbit = await digest(worldCanvas), box = await worldCanvas.boundingBox();
    await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
    await page.mouse.down();
    await page.mouse.move(box.x+box.width/2+70,box.y+box.height/2+40,{steps:8});
    await page.mouse.up(); await page.waitForTimeout(300);
    assert.notEqual(await digest(worldCanvas), beforeOrbit, "3D orbit must change the image");

    await page.locator("#layout").selectOption("separate");
    assert.equal(await page.locator("#view-2d canvas").count(), 2);
    assert.equal(await page.locator("#view-3d canvas").count(), 3);
    await shot("03-side-by-side");
    await page.locator("#layout").selectOption("overlay");
    await page.locator("#augmentation-mode").selectOption("none");
    await waitFor("GT・拡張後入力");
    const clean = captured.at(-1);
    assert.equal(clean.scene.audit.frame_missing_rate_2d,0);
    assert.equal(clean.scene.audit.noise_p95_px,0);
    assert.equal(clean.scene.event_probability,null,"augmentation changes must invalidate predictions");
    assert.equal(clean.scene.prediction_3d,null);
    await page.locator("#plot-mode").selectOption("error");
    await shot("04-clean-preview");

    await page.locator("#augmentation-mode").selectOption("both");
    await page.locator("#event-probability").fill("75");
    await page.locator("#noise-jitter").fill("3");
    await page.locator("#noise-outlier").fill("10");
    await page.locator("#noise-p95").fill("300");
    await page.locator("#augmentation-seed").fill("1234");
    await waitFor("GT・拡張後入力");
    await page.waitForFunction(() => document.querySelector("#audit-noise").textContent.includes("px"));
    await page.locator("#infer").click();
    await waitFor("CPU 推論");
    const live = captured.at(-1);
    assert.equal(live.source,"live");
    assert.equal(live.request.augmentation.noise_p95_px,300);
    assert.equal(live.request.augmentation.event_probability,0.75);
    assert.notEqual(live.input_sha256,baselineInput);
    assert.equal(live.scene.prediction_3d.length,live.scene.frames);
    assert.ok(live.scene.prediction_3d.every(point=>point.every(Number.isFinite)));
    assert.equal(live.scene.event_probability.length, live.scene.frames);
    assert.ok(live.scene.event_probability.every(p=>Number.isFinite(p) && p>=0 && p<=1));
    live.scene.events.forEach((event,i)=>{if(event) assert.equal(live.scene.event_target[i],1);});
    await shot("05-custom-inference");

    const configDownload = page.waitForEvent("download");
    await page.locator("#save-config").click();
    const configFile = path.join(out,"saved-review.json");
    await (await configDownload).saveAs(configFile);
    const savedConfig = JSON.parse(await fs.readFile(configFile,"utf8"));
    assert.equal(savedConfig.input_sha256,live.input_sha256);
    const imageDownload = page.waitForEvent("download");
    await page.locator("#save-image").click();
    const png = path.join(out,"exported-comparison.png");
    await (await imageDownload).saveAs(png);
    const bytes = await fs.readFile(png);
    assert.equal(bytes.subarray(1,4).toString(),"PNG");
    assert.ok(bytes.length>10000);
    await page.locator("#resample").click();
    await waitFor("GT・拡張後入力");
    assert.notEqual(captured.at(-1).input_sha256,live.input_sha256);
    await page.locator("#config-file").setInputFiles(configFile);
    await waitFor("CPU 推論");
    assert.equal(captured.at(-1).input_sha256,live.input_sha256);
    assert.deepEqual(captured.at(-1).scene.prediction_3d,live.scene.prediction_3d);

    await page.locator("#evaluation-preset").click();
    await waitFor("保存済み評価");
    const flowOption = await page.locator("#model-3d option").evaluateAll(options => options.find(o => /FLOW/.test(o.textContent) && /best.ckpt$/.test(o.value))?.value);
    assert.ok(flowOption,"trained Flow checkpoint must be suggested");
    await page.locator("#model-3d").selectOption(flowOption);
    await waitFor("保存済み評価");
    await page.locator("#resample").click();
    await waitFor("GT・拡張後入力");
    await page.locator("#infer").click();
    await waitFor("CPU 推論");
    assert.equal(captured.at(-1).models["3"].method,"flow");
    await shot("06-flow-inference");

    const pure = await page.evaluate(async () => {
      const {valuesFor} = await import("/static/plots.mjs");
      return {
        error: valuesFor([[3,4],[6,8]],[[0,0],[0,0]],null,"error",60),
        speed: valuesFor([[0,0],[3,4],[6,8]],null,[false,true,false],"speed",60),
      };
    });
    assert.deepEqual(pure.error,[5,10]);
    assert.deepEqual(pure.speed,[null,null,null],"a speed trace must not bridge a missing sample");

    // A late inference reply cannot replace a newer rally/augmentation preview.
    let release, started;
    const held = new Promise(resolve=>release=resolve), began = new Promise(resolve=>started=resolve);
    await page.route("**/api/infer",async route=>{started();await held;await route.fulfill({status:200,json:live});});
    await page.locator("#infer").click(); await began;
    await page.locator("#resample").click();
    await waitFor("GT・拡張後入力");
    const current = await page.locator("#scene-title").textContent();
    release(); await page.waitForTimeout(200);
    assert.equal(await page.locator("#scene-title").textContent(),current);
    assert.equal(await page.locator("#source").textContent(),"GT・拡張後入力");
    await page.unroute("**/api/infer");

    for (const width of [390,800,1600]) {
      await page.setViewportSize({width,height:width===390?844:1100});
      await page.waitForTimeout(200);
      const overflow = await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
      assert.equal(overflow,false,"page must not overflow horizontally at "+width);
      if (width===390) await shot("07-mobile");
    }
    assert.deepEqual(errors,[]);
    await fs.writeFile(path.join(out,"browser-results.json"),JSON.stringify({status:"passed",errors,
      checks:["trained suggestions","saved predictions","event/frame sync","Gaussian target and softmax probabilities","3D orbit","side-by-side",
        "augmentation switches","custom CPU inference","Flow inference","JSON/PNG export","state restore","stale response isolation","responsive layout"],
      initial_input:baselineInput,custom_input:live.input_sha256,custom_metrics:live.scene.metrics_3d},null,2));
    console.log("Coordinate review browser checks passed: "+out);
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
