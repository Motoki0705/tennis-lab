// Run against a DISPOSABLE, empty project: this test creates/edits/deletes clips.
// PLAYWRIGHT_MODULE=/absolute/node_modules/playwright/index.mjs
// CLIP_STUDIO_TEST_URL=http://127.0.0.1:8767 node tests/e2e/tennis_scene/clip_studio_browser.mjs
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({headless:true,
  ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const errors = [];
page.on('pageerror',error => errors.push(error.message));
const base = process.env.CLIP_STUDIO_TEST_URL;
assert.ok(base, 'CLIP_STUDIO_TEST_URL must identify a disposable test project');
try {
  await page.goto(base);
  await page.waitForSelector('.viewer img:not([hidden])');
  const initial = await (await page.request.get(`${base}/api/project`)).json();
  assert.equal(initial.clips.length, 0, 'Use an empty disposable project');
  assert.ok(initial.sources.length >= 2);
  assert.ok(initial.extent[1] > 40);
  assert.equal(await page.locator('.viewer:visible').count(), 1);
  const picture = page.locator('.viewer:visible .picture');
  const pictureBox = await picture.boundingBox();
  assert.ok(pictureBox);
  const pointer = {x:pictureBox.x + pictureBox.width * 0.75, y:pictureBox.y + pictureBox.height * 0.2};
  await page.keyboard.down('Control');
  await page.mouse.move(pointer.x,pointer.y); await page.mouse.wheel(0,-350);
  await page.keyboard.up('Control');
  const imageBox = await picture.locator('img:not([hidden])').boundingBox();
  assert.ok(imageBox && imageBox.width > pictureBox.width);
  assert.ok(Math.abs((pointer.x-imageBox.x)/imageBox.width-0.75)<0.001);
  assert.ok(Math.abs((pointer.y-imageBox.y)/imageBox.height-0.2)<0.001);
  const zoomedReadout = await page.locator('.viewer:visible .zoom-readout').textContent();
  assert.notEqual(zoomedReadout,'100%');
  assert.equal(await picture.locator('video,img').evaluateAll(media => media[0].style.transform === media[1].style.transform),true);
  await page.locator('#compare').click();
  assert.equal(await page.locator('.viewer:visible').count(), initial.sources.length);
  const zoomReadouts = await page.locator('.zoom-readout').allTextContents();
  assert.equal(zoomReadouts[0],zoomedReadout);
  assert.ok(zoomReadouts.slice(1).every(readout => readout === '100%'));
  await page.locator('#focus').click();
  await page.getByRole('button',{name:'cam0 の拡大表示を等倍に戻す'}).click();
  assert.equal(await page.locator('.viewer:visible .zoom-readout').textContent(),'100%');
  const syncPanel = page.locator('#sync-panel');
  assert.equal(await syncPanel.isHidden(), true);
  const fullWidth = (await page.locator('#viewers').boundingBox()).width;
  await page.locator('#sync-toggle').click();
  const [viewersBox, syncBox] = await Promise.all([
    page.locator('#viewers').boundingBox(), syncPanel.boundingBox(),
  ]);
  assert.ok(viewersBox && syncBox);
  assert.ok(viewersBox.width < fullWidth);
  assert.ok(syncBox.x >= viewersBox.x + viewersBox.width);
  assert.ok(Math.abs(syncBox.y - viewersBox.y) < 2);
  await page.locator('#sync-close').click();
  assert.equal(await syncPanel.isHidden(), true);
  assert.equal((await page.locator('#viewers').boundingBox()).width, fullWidth);
  await page.locator('#camera').selectOption('1');
  assert.equal(await page.locator('.viewer.selected:visible').count(), 1);
  await page.locator('#camera').selectOption('0');
  const seek = async time => { await page.locator('#seek-time').fill(String(time)); await page.locator('#jump').click(); };
  await seek(5);
  await page.locator('#step').selectOption('5');
  await page.locator('#back').click(); // button focus must not suppress keyboard commands
  await page.keyboard.press('ArrowRight');
  assert.equal(Number(await page.locator('#seek-time').inputValue()), 5);
  await page.locator('#seek').focus();
  await page.keyboard.press('ArrowRight'); // range focus must also use configured step
  assert.equal(Number(await page.locator('#seek-time').inputValue()), 10);
  await page.locator('#step').selectOption('frame');
  await page.locator('#forward').click();
  const frameTime = 10 + 1 / initial.sources[0].fps;
  assert.ok(Math.abs(Number(await page.locator('#seek-time').inputValue())-frameTime)<0.001);
  await page.keyboard.press('i');
  await seek(13); await page.keyboard.press('o'); await page.keyboard.press('c');
  await page.waitForFunction(() => document.querySelector('#count').textContent === '1 clips');
  // Fractional frame boundary must not trigger native stepMismatch on submit.
  await page.locator('#clip-name').fill('rally_test');
  await page.locator('#clip-editor button[type=submit]').click();
  await page.waitForFunction(() => document.querySelector('.clip-row')?.textContent.includes('rally_test'));
  await page.locator('#delete').click();
  await page.waitForFunction(() => document.querySelector('#count').textContent === '0 clips');
  await page.locator('#undo').click();
  await page.waitForFunction(() => document.querySelector('#count').textContent === '1 clips');
  await page.reload(); await page.waitForSelector('.clip-row');
  assert.ok((await page.locator('.clip-row').textContent()).includes('rally_test'));
  // Progressive scrub updates before the drag ends, rather than trailing debounce.
  let frames = 0; page.on('response', response => { if (response.url().includes('/api/frame/0')) frames++; });
  await page.evaluate(async () => {
    const slider = document.querySelector('#seek');
    for (let i=0;i<30;i++) { slider.value=String(15+i/5); slider.dispatchEvent(new Event('input')); await new Promise(r=>setTimeout(r,25)); }
  });
  assert.ok(frames >= 2, `Expected progressive previews while scrubbing, got ${frames}`);
  await seek(20); await page.locator('#rate').selectOption('2');
  await page.locator('#play').click();
  await page.waitForFunction(() => document.querySelector('#play').textContent.includes('停止'));
  await page.waitForTimeout(1200);
  await page.locator('#play').click();
  const after = Number(await page.locator('#seek-time').inputValue());
  assert.ok(after > 21.5 && after < 23.5, `2x playback advanced to ${after}`);
  assert.ok(await page.locator('video').evaluateAll(videos => videos.filter((_,i)=>i!==0).every(v=>v.paused)));
  await page.locator('#sync-toggle').click();
  const offset = page.getByRole('spinbutton',{name:'cam1 時差（秒）'});
  await offset.fill('0.25');
  await page.locator('.offset-row').nth(1).getByRole('button',{name:'時差を適用'}).click();
  await page.waitForFunction(() => document.querySelector('#saved').textContent === '保存済み');
  const saved = await (await page.request.get(`${base}/api/project`)).json();
  assert.equal(saved.sources[1].offset_sec,0.25);
  assert.ok(Math.abs(saved.clips[0].start_sec-frameTime)<1e-6);
  await page.setViewportSize({width:780,height:1000});
  assert.ok(await page.locator('#seek').isVisible());
  const [narrowViewersBox, narrowSyncBox] = await Promise.all([
    page.locator('#viewers').boundingBox(), syncPanel.boundingBox(),
  ]);
  assert.ok(narrowViewersBox && narrowSyncBox);
  assert.ok(narrowSyncBox.y >= narrowViewersBox.y + narrowViewersBox.height);
  assert.deepEqual(errors,[]);
  if(process.env.CLIP_STUDIO_SCREENSHOT) await page.screenshot({path:process.env.CLIP_STUDIO_SCREENSHOT,fullPage:true});
  console.log('PASS: focus/comparison, pointer-anchored per-camera zoom, nearby collapsible sync panel, keyboard steps, fractional-frame edits, undo, reload, progressive scrubbing, 2x playback, sync offsets, responsive layout');
} finally { await browser.close(); }
