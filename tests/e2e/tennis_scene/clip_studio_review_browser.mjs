// ONLY use the disposable clip_studio_review_test project. Never run on real data.
// PLAYWRIGHT_MODULE=.../playwright-core/index.mjs CLIP_STUDIO_REVIEW_TEST_URL=http://127.0.0.1:8904 node <this file>
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const base = process.env.CLIP_STUDIO_REVIEW_TEST_URL;
assert.ok(base, 'Use a disposable review test server');
const browser = await chromium.launch({headless:true,
  ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
const page = await browser.newPage({viewport:{width:1600,height:1100}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  const original = await (await page.request.get(`${base}/api/project`)).json();
  assert.equal(original.dataset_id,'clip_studio_review_test','Never run on a production project');
  assert.equal(original.read_only,true);
  assert.deepEqual(original.sources.map(s => s.offset_sec),[0,-1]);
  assert.equal(original.clips[0].name,'clip_000');
  await page.goto(base);
  await page.locator('#compare').click();
  const seek = async time => {
    await page.locator('#seek-time').fill(String(time));await page.locator('#jump').click();
    await page.waitForFunction(t => document.querySelector('#review-time')?.dataset.time === String(t),time);
    await page.waitForFunction(() => Array.from(document.querySelectorAll('.frame-label')).every(e => !e.textContent.includes('取得中')));
  };
  await seek(0);
  assert.ok((await page.locator('#frame-map tr').nth(1).textContent()).includes('範囲外・映像なし'));
  assert.ok((await page.locator('.frame-label').nth(1).textContent()).includes('映像はありません'));
  await page.locator('.clip-row').click();await seek(2);
  assert.ok((await page.locator('#clip-review').textContent()).includes('[2.000, 4.000)'));
  assert.ok((await page.locator('#clip-review .membership').textContent()).includes('区間内'));
  assert.ok((await page.locator('#clip-review .clip-state').textContent()).includes('未出力'));
  assert.equal(await page.locator('#frame-map tr').nth(0).locator('td').nth(3).textContent(),'20');
  assert.equal(await page.locator('#frame-map tr').nth(1).locator('td').nth(3).textContent(),'10');
  assert.ok((await page.locator('.frame-label').nth(1).textContent()).includes('frame 10'));
  await seek(4);
  assert.ok((await page.locator('#clip-review .membership').textContent()).includes('区間外'));
  assert.ok((await page.locator('#clip-membership').textContent()).includes('保存clip: なし'));
  const pausedHeight = (await page.locator('#time-review').boundingBox()).height;
  await page.locator('#play').click();
  await page.waitForFunction(() => document.querySelector('#review-time').textContent.includes('再生中'));
  assert.deepEqual(await page.locator('#frame-map tr td:nth-child(4)').allTextContents(),['—','—']);
  assert.ok(Math.abs((await page.locator('#time-review').boundingBox()).height - pausedHeight) < 1);
  await page.locator('#play').click();await seek(4);
  assert.equal(await page.locator('#export-all').isVisible(),false);
  assert.equal(await page.locator('#sync-toggle').isVisible(),false);
  assert.equal(await page.locator('#clip-editor').isVisible(),false);
  // The endpoint must enforce readonly even when the UI is bypassed.
  for (const [path,body] of [
    ['edit',{revision:0,action:'create',start_sec:5,end_sec:6}],
    ['jobs',{revision:0,kind:'export'}],['jobs',{revision:0,kind:'sync'}],
  ]) assert.equal((await page.request.post(`${base}/api/${path}`,{data:body})).status(),403);
  await page.locator('#reload').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('再読込しました'));
  await seek(2);
  const restored = await (await page.request.get(`${base}/api/project`)).json();
  assert.deepEqual(restored,original);
  await page.setViewportSize({width:780,height:1000});
  assert.ok(await page.locator('#frame-map').isVisible());
  assert.ok(await page.locator('#seek').isVisible());
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth),false);
  assert.deepEqual(errors,[]);
  console.log('PASS: readonly API/UI, raw frame correspondence, negative offset missing video, half-open clip boundary, unexported state, reload, responsive layout');
} finally {await browser.close();}
