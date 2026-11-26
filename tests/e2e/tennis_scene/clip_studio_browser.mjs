// Run against a DISPOSABLE, empty project: this test creates/edits/deletes clips.
// PLAYWRIGHT_MODULE=/absolute/node_modules/playwright/index.mjs
// CLIP_STUDIO_TEST_URL=http://127.0.0.1:8767 node tests/e2e/tennis_scene/clip_studio_browser.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({headless:true,
  ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
page.setDefaultTimeout(5000);
const errors = [];
page.on('pageerror',error => errors.push(error.message));
const frameRequests = [];
page.on('request', request => { const match = request.url().match(/\/api\/frame\/(\d+)/); if (match) frameRequests.push(Number(match[1])); });
const streamedCameras = () => page.locator('video').evaluateAll(videos => videos.flatMap((video, index) => video.getAttribute('src') ? [index] : []));
const openSettings = async () => { if (await page.locator('#settings-menu').isHidden()) await page.locator('#settings').click(); };
const base = process.env.CLIP_STUDIO_TEST_URL;
assert.ok(base, 'CLIP_STUDIO_TEST_URL must identify a disposable test project');
const snapshot = async () => (await page.request.get(`${base}/api/project`)).json();
const row = name => page.locator(`.clip-row[data-clip="${name}"]`);
const waitForSave = () => page.waitForFunction(() => document.querySelector('#saved').textContent === '保存済み');
async function saveEdit(action) {
  const [response] = await Promise.all([page.waitForResponse('**/api/edit'), action()]);
  assert.equal(response.status(), 200);
  await waitForSave();
}
async function answerConfirmation(click, accept) {
  const answered = page.waitForEvent('dialog').then(async dialog => {
    const message = dialog.message();
    if (accept) await dialog.accept(); else await dialog.dismiss();
    return message;
  });
  const [, message] = await Promise.all([click(), answered]);
  return message;
}
async function confirmChanges(click, count, accept = true) {
  const message = await answerConfirmation(click, accept);
  assert.match(message, new RegExp(`${count} クリップのカウントが変わります`));
}
async function confirmBreaks(click, names, accept = true) {
  const message = await answerConfirmation(click, accept);
  assert.ok(message.includes(`${names.length} クリップで連鎖切れになります（${names.join('、')}）`), message);
}
try {
  await page.goto(base);
  await page.waitForSelector('.viewer img:not([hidden])');
  const initial = await snapshot();
  assert.equal(initial.dataset_id, 'clip_studio_edit_test', 'Never run editing E2E on a production project');
  assert.equal(initial.clips.length, 0, 'Use an empty disposable project');
  assert.equal(initial.match, null, 'Use a fresh disposable project with no saved match');
  assert.ok(initial.sources.length >= 2);
  assert.ok(initial.extent[1] > 40);
  assert.equal(await page.locator('.viewer:visible').count(), 1);
  // Single-camera clip work streams one camera and hides dataset/all-camera review panels.
  assert.deepEqual(await streamedCameras(), [0]);
  assert.ok(frameRequests.length > 0 && frameRequests.every(camera => camera === 0));
  assert.equal(await page.locator('#dataset-map').isVisible(), false);
  assert.equal(await page.locator('#time-review').isVisible(), false);
  assert.equal(await page.locator('#rate').isVisible(), false, 'rate lives in the settings menu');
  const [stageBox, gearBox] = await Promise.all([page.locator('#stage').boundingBox(), page.locator('#settings').boundingBox()]);
  assert.ok(gearBox.x + gearBox.width > stageBox.x + stageBox.width - 4 && gearBox.y < stageBox.y + 60, 'gear sits at the top right');
  await page.locator('#settings').click();
  assert.equal(await page.locator('#settings-menu').isVisible(), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#settings-menu').isVisible(), false);
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
  assert.deepEqual(await streamedCameras(), initial.sources.map((_, index) => index), 'comparison loads all cameras on demand');
  await page.waitForFunction(count => document.querySelectorAll('.viewer img:not([hidden])').length === count, initial.sources.length);
  assert.equal(await page.locator('#time-review').isVisible(), true);
  const zoomReadouts = await page.locator('.zoom-readout').allTextContents();
  assert.equal(zoomReadouts[0],zoomedReadout);
  assert.ok(zoomReadouts.slice(1).every(readout => readout === '100%'));
  await page.locator('#focus').click();
  assert.deepEqual(await streamedCameras(), [0]);
  await page.getByRole('button',{name:'cam0 の拡大表示を等倍に戻す'}).click();
  assert.equal(await page.locator('.viewer:visible .zoom-readout').textContent(),'100%');
  const syncPanel = page.locator('#sync-panel');
  assert.equal(await syncPanel.isHidden(), true);
  const fullWidth = (await page.locator('#viewers').boundingBox()).width;
  await page.locator('#sync-toggle').click();
  assert.equal(await page.locator('#compare').getAttribute('aria-pressed'), 'true', 'sync adjusts in comparison');
  assert.equal(await page.locator('#fullscreen').isDisabled(), true);
  assert.deepEqual(await streamedCameras(), initial.sources.map((_, index) => index));
  const [viewersBox, syncBox] = await Promise.all([
    page.locator('#viewers').boundingBox(), syncPanel.boundingBox(),
  ]);
  assert.ok(viewersBox && syncBox);
  assert.ok(viewersBox.width < fullWidth);
  assert.ok(syncBox.x >= viewersBox.x + viewersBox.width);
  assert.ok(Math.abs(syncBox.y - viewersBox.y) < 2);
  await page.locator('#camera').selectOption('1');
  await page.locator('#sync-close').click(); // 同期終了
  assert.equal(await syncPanel.isHidden(), true);
  assert.equal((await page.locator('#viewers').boundingBox()).width, fullWidth);
  assert.equal(await page.locator('.viewer.selected:visible').count(), 1);
  assert.deepEqual(await streamedCameras(), [1], 'finishing sync releases every other camera');
  await page.waitForFunction(() => !document.querySelector('.viewer:not([hidden]) .frame-label').textContent.includes('取得中'));
  frameRequests.length = 0;
  await page.locator('#camera').selectOption('0');
  assert.deepEqual(await streamedCameras(), [0]);
  const seek = async time => { await page.locator('#seek-time').fill(String(time)); await page.locator('#jump').click(); };
  await seek(5);
  await openSettings(); await page.locator('#step').selectOption('5');
  await page.locator('#back').click(); // button focus must not suppress keyboard commands
  await page.keyboard.press('ArrowRight');
  assert.equal(Number(await page.locator('#seek-time').inputValue()), 5);
  await page.locator('#seek').focus();
  await page.keyboard.press('ArrowRight'); // range focus must also use configured step
  assert.equal(Number(await page.locator('#seek-time').inputValue()), 10);
  await openSettings(); await page.locator('#step').selectOption('frame');
  await page.locator('#forward').click();
  const frameTime = 10 + 1 / initial.sources[0].fps;
  assert.ok(Math.abs(Number(await page.locator('#seek-time').inputValue())-frameTime)<0.001);
  await page.keyboard.press('i');
  await seek(13); await page.keyboard.press('o');
  assert.equal(await page.locator('#clip-track .mark-range').count(), 1, 'I/O range is drawn on the segment bar');
  assert.equal(await page.locator('#clip-track .mark-point').count(), 2);
  // Fullscreen keeps only the playback bar and the I/O + clip segment bar.
  await page.locator('#fullscreen').click();
  await page.waitForFunction(() => document.fullscreenElement?.id === 'stage');
  for (const id of ['#seek','#clip-track','#play','#settings','#fullscreen']) assert.ok(await page.locator(id).isVisible(), `${id} in fullscreen`);
  for (const id of ['.marks','.timeline-tools','#compare','#camera','#sync-toggle']) assert.equal(await page.locator(id).isVisible(), false, `${id} hidden in fullscreen`);
  const viewport = page.viewportSize();
  const [fsPicture, fsTransport, fsGear] = await Promise.all(['.viewer:visible .picture','.transport','#settings'].map(id => page.locator(id).boundingBox()));
  assert.ok(fsPicture.height > viewport.height * 0.75, `fullscreen picture height ${fsPicture.height}`);
  assert.ok(fsTransport.y + fsTransport.height >= viewport.height - 1, 'transport is docked at the bottom');
  assert.ok(fsGear.x > viewport.width * 0.85 && fsGear.y < 60, 'gear is at the top right in fullscreen');
  await page.locator('#settings').click();
  assert.ok(await page.locator('#rate').isVisible(), 'settings menu opens in fullscreen');
  await page.locator('#seek').click({position:{x:5,y:5}}); // outside click closes the menu and seeks
  assert.equal(await page.locator('#settings-menu').isVisible(), false);
  await page.locator('#seek').blur();
  await page.keyboard.press('f');
  await page.waitForFunction(() => document.fullscreenElement === null);
  assert.deepEqual(await streamedCameras(), [0]);
  assert.ok(frameRequests.every(camera => camera === 0), `unfocused cameras fetched frames: ${frameRequests}`);
  await page.keyboard.press('c');
  await page.waitForFunction(() => document.querySelector('#count').textContent === '1 clips');
  // Fractional frame boundary must not trigger native stepMismatch on submit.
  await page.locator('#clip-name').fill('rally_test');
  await page.locator('#clip-editor button[type=submit]').click();
  await page.waitForFunction(() => document.querySelector('.clip-row')?.textContent.includes('rally_test'));
  await page.locator('#export-one').click();
  await page.waitForFunction(() => document.querySelector('#job').textContent.includes('クリップを書き出しました'), null, {timeout: 10000});
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
  await seek(20); await openSettings(); await page.locator('#rate').selectOption('2');
  assert.equal(await page.locator('#settings').textContent(), '⚙ 2×');
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

  // Score saved clips while watching the source, then correct the confirmed chain.
  await page.locator('#sync-close').click();
  for (const [start, end] of [[15, 17], [22, 25]]) {
    await seek(start); await page.locator('#mark-in').click();
    await seek(end); await page.locator('#mark-out').click();
    await saveEdit(() => page.locator('#create').click());
  }
  let state = await snapshot();
  const [first, second, third] = state.clips.map(clip => clip.name);
  assert.deepEqual(state.clips.map(clip => clip.score), [null, null, null]);
  assert.match(await row(second).textContent(), /未入力/);
  await row(first).click();
  await page.locator('#open-match').click();
  await page.locator('#match-a').fill('青ペア');
  await page.locator('#match-b').fill('赤ペア');
  await saveEdit(() => page.locator('#match-form button[type=submit]').click());
  await page.locator('#score-points-a').fill('2');
  await page.locator('#score-points-b').fill('1');
  await page.keyboard.press('b'); // The focused score input suppresses shortcuts.
  assert.equal((await snapshot()).clips[0].score, null);
  await page.locator('#play').click();
  await page.waitForFunction(() => document.querySelector('#play').textContent.includes('停止'));
  await saveEdit(() => confirmChanges(() => page.keyboard.press('a'), 1));
  assert.match(await page.locator('#play').textContent(), /停止/, 'score save preserves playback');
  assert.match(await page.locator('#score-after').textContent(), /終了後.*3–1/);
  await row(second).click();
  assert.match(await row(second).textContent(), /開始前.*3–1.*仮入力/);
  await page.locator('#score-note').fill('ネット際');
  await page.locator('#score-server').selectOption('B');
  await saveEdit(() => confirmChanges(() => page.locator('#winner-b').click(), 1));
  state = await snapshot();
  assert.equal(state.clips[1].score.winner, 'B');
  assert.deepEqual(state.clips[2].score.before.points, [3, 2]);

  await page.reload(); await row(first).click();
  assert.match(await page.locator('#score-current').textContent(), /確定.*青ペア/);
  await confirmChanges(() => page.locator('#winner-b').click(), 2, false);
  assert.equal((await snapshot()).revision, state.revision, 'cancelled preview leaves the revision unchanged');
  await saveEdit(() => confirmChanges(() => page.locator('#winner-b').click(), 2));
  state = await snapshot();
  assert.deepEqual(state.clips.map(clip => clip.score.before.points), [[2, 1], [2, 2], [2, 3]]);
  assert.equal(state.clips[1].score.note, 'ネット際');
  assert.equal(state.clips[1].score.server, 'B');
  assert.equal(state.clips[1].score.winner, 'B');
  assert.match(await row(third).textContent(), /2–3/);

  const clearScore = () => page.locator('#score-clear').click();
  await confirmBreaks(clearScore, [second, third], false);
  assert.equal((await snapshot()).revision, state.revision);
  await saveEdit(() => confirmBreaks(clearScore, [second, third]));
  await row(second).click();
  assert.match(await row(second).textContent(), /確定.*得点 B.*連鎖切れ/);
  assert.doesNotMatch(await row(second).textContent(), /開始前/);
  assert.match(await page.locator('#score-warning').textContent(), /連鎖切れ.*開始前カウントを手入力/);
  assert.ok(await page.locator('#score-after').isHidden());
  assert.equal(await page.locator('#score-points-a').inputValue(), '');
  const broken = await snapshot();
  assert.deepEqual(broken.clips[1].score, state.clips[1].score);

  await page.locator('#match-settings summary').click();
  const [blockedCsv] = await Promise.all([
    page.waitForResponse('**/api/score-csv?*'), page.locator('#score-csv-download').click(),
  ]);
  assert.equal(blockedCsv.status(), 400);
  assert.equal(blockedCsv.headers()['content-disposition'], undefined);
  await page.waitForFunction(names => {
    const message = document.querySelector('#status.error')?.textContent;
    return message?.includes('連鎖切れ') && names.every(name => message.includes(name));
  }, [second, third]);
  const rejected = await page.request.post(`${base}/api/edit`, {data: {
    action: 'import_scores', revision: broken.revision, csv_text: await blockedCsv.text(),
  }});
  assert.equal(rejected.status(), 400);
  assert.deepEqual(await snapshot(), broken);
  await saveEdit(() => page.locator('#undo').click());
  assert.ok(await page.locator('#score-after').isVisible());

  const [download] = await Promise.all([
    page.waitForEvent('download'), page.locator('#score-csv-download').click(),
  ]);
  const csv = await readFile(await download.path(), 'utf8');
  assert.ok(csv.includes(`${second},confirmed,,,0,0,2,2,B,B,ネット際`));
  await page.locator('#score-csv-file').setInputFiles({
    name: 'score.csv', mimeType: 'text/csv',
    buffer: Buffer.from(csv.split('\r\n')[0] + `\r\n${second},confirmed,,,0,0,1,1,A,A,CSV取込\r\n`),
  });
  await saveEdit(() => confirmBreaks(() => page.locator('#score-csv-form button[type=submit]').click(), [third]));
  const imported = await snapshot();
  assert.deepEqual(imported.clips[0].score, state.clips[0].score);
  assert.deepEqual(imported.clips[2].score, state.clips[2].score);
  assert.equal(imported.clips[1].score.note, 'CSV取込');
  assert.equal(imported.clips[1].score.server_source, 'manual');
  await saveEdit(() => page.locator('#undo').click());
  assert.deepEqual((await snapshot()).clips, state.clips);

  await row(second).click();
  await page.locator('#clip-name').fill('moved_rally');
  await page.locator('#clip-start').fill('29');
  await page.locator('#clip-end').fill('31');
  await saveEdit(() => confirmBreaks(() => page.locator('#clip-editor button[type=submit]').click(), [third, 'moved_rally']));
  await page.reload(); await row('moved_rally').click();
  const moved = await snapshot();
  assert.deepEqual(moved.clips.map(clip => clip.score), state.clips.map(clip => clip.score));
  assert.match(await row(third).textContent(), /連鎖切れ/);

  await saveEdit(clearScore);
  await row('moved_rally').locator('.clip-score.unscored').waitFor();
  assert.match(await row('moved_rally').textContent(), /未入力/);
  assert.deepEqual((await snapshot()).clips[2].score, moved.clips[2].score);

  await page.locator('#sync-toggle').click();
  await page.setViewportSize({width:780,height:1000});
  assert.ok(await page.locator('#seek').isVisible());
  const [narrowViewersBox, narrowSyncBox] = await Promise.all([
    page.locator('#viewers').boundingBox(), syncPanel.boundingBox(),
  ]);
  assert.ok(narrowViewersBox && narrowSyncBox);
  assert.ok(narrowSyncBox.y >= narrowViewersBox.y + narrowViewersBox.height);
  assert.deepEqual(errors,[]);
  if(process.env.CLIP_STUDIO_SCREENSHOT) await page.screenshot({path:process.env.CLIP_STUDIO_SCREENSHOT,fullPage:true});
  console.log('PASS: playback, clip edits, sync and responsive layout; score chain preview/cancel/apply, broken-chain warning and CSV rejection, reload, rename and reorder, CSV download/import/undo, and clearing');
} finally { await browser.close(); }
