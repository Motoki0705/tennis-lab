// Read-only smoke on a running viewer backed by actual local player data.
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.PLAYER_REVIEW_URL || 'http://127.0.0.1:8895';
(async () => {
 const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || chromium.executablePath(), args:['--no-sandbox']});
 try {
  const page = await browser.newPage({viewport:{width:1600,height:1200}});
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto(base, {waitUntil:'networkidle'});
  await page.waitForFunction(() => document.querySelector('#frame-label').textContent.startsWith('frame '));
  assert.equal(await page.locator('#error').isVisible(), false);
  const image = await page.locator('#viewer').evaluate(canvas => canvas.toDataURL());
  await page.locator('#boxes').uncheck();
  assert.notEqual(await page.locator('#viewer').evaluate(canvas => canvas.toDataURL()), image);
  await page.locator('#boxes').check();
  assert.equal(await page.locator('#viewer').evaluate(canvas => canvas.toDataURL()), image);
  await page.locator('#viewer').hover(); await page.mouse.wheel(0,-500);
  assert.notEqual(await page.locator('#viewer').evaluate(canvas => canvas.toDataURL()), image);
  await page.locator('#reset-view').click();
  assert.equal(await page.locator('#viewer').evaluate(canvas => canvas.toDataURL()), image);
  const oldFrame = await page.locator('#frame-label').innerText();
  await page.locator('#next').click();
  await page.waitForFunction(old => document.querySelector('#frame-label').textContent !== old, oldFrame);
  await page.locator('#previous').click();
  await page.waitForFunction(old => document.querySelector('#frame-label').textContent === old, oldFrame);
  await page.locator('#play').click();
  await page.waitForFunction(old => document.querySelector('#frame-label').textContent !== old, oldFrame);
  await page.locator('#play').click();
  assert.equal(await page.locator('#play').innerText(), '▶ 再生');
  await page.locator('#flag').selectOption('unresolved');
  await page.waitForFunction(() => document.querySelectorAll('.unknown').length > 0 && document.querySelector('#eligibility').textContent.includes('除外'));
  assert.match(await page.locator('#players').innerText(), /座標なし/);
  assert.match(await page.locator('#eligibility').innerText(), /frame全体/);
  const catalogue = await (await page.request.get(base + '/api/catalog')).json();
  const dataset = catalogue.datasets.find(d => d.available);
  const clips = await (await page.request.get(base + '/api/clips?' + new URLSearchParams({dataset:dataset.id}))).json();
  const withGap = clips.clips.find(c => c.unstored_frames > 0);
  if (withGap) {
   const timeline = await (await page.request.get(base + '/api/clip?' + new URLSearchParams({dataset:dataset.id,clip:withGap.id}))).json();
   const existing = timeline.frames[0].frame_index;
   await page.goto(base + '/#' + new URLSearchParams({dataset:dataset.id,clip:withGap.id,frame:existing}), {waitUntil:'networkidle'});
   await page.waitForFunction(frame => document.querySelector('#jump').value === String(frame), existing);
   await page.locator('#jump').fill(String(timeline.missing_ranges[0][0]));
   await page.locator('#jump').dispatchEvent('change');
   assert.match(await page.locator('#error').innerText(), /未保存/);
   assert.equal(await page.locator('#jump').inputValue(), String(existing));
  }
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.locator('#search').fill('__NO_SUCH_PLAYER_REVIEW_CLIP__');
  await page.waitForFunction(() => document.querySelector('#clip-count').textContent === '0 clips');
  assert.match(await page.locator('#sample-title').innerText(), /一致するクリップがありません/);
  assert.equal(await page.locator('#players').locator('.player').count(), 0);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ok:true,viewport:[1600,1200],mobile:[390,844],checks:['bbox/original','zoom/reset','seek/next/previous','play/pause','unresolved state','unstored frame refusal','empty filter state','mobile width','no browser errors']}, null, 2));
 } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode=1;});
