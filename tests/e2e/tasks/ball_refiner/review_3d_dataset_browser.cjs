/* Read-only browser checks and real-dataset screenshot capture. No fixture fallback. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.REVIEW_URL || 'http://127.0.0.1:8893';
const output = process.env.CAPTURE_DIR;
const datasetRoot = process.env.DATASET_ROOT;
if (!output || !datasetRoot) throw new Error('CAPTURE_DIR and real DATASET_ROOT are required');
const dataset = 'i936-dev-h-anchored-r11-s936';
const rally = 'val-00000';
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const viewport = {width: 1680, height: 1280};
fs.mkdirSync(output, {recursive: true});
const frames = [];
const checks = [];

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport, deviceScaleFactor: 1});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    async function ready(index) {
      await page.waitForFunction(i => document.getElementById('frame').value === String(i) && document.getElementById('time').textContent === `${(i * 1001 / 60000).toFixed(4)} s`, index);
      assert.equal(await page.locator('#error').isVisible(), false);
      assert.equal(await page.locator('#components tr').count(), 125);
      assert.equal(await page.locator('#cameras canvas').count(), 3);
    }
    async function change(index) {
      await page.locator('#frame').fill(String(index));
      await page.locator('#frame').dispatchEvent('change');
      await ready(index);
    }
    async function capture(name, frame, description) {
      await ready(frame);
      const response = await page.request.get(`${base}/api/frame?${new URLSearchParams({dataset, rally, frame})}`);
      assert.equal(response.status(), 200);
      const data = await response.json();
      const file = `${name}.png`;
      await page.screenshot({path: path.join(output, file)});
      const manifestFile = path.join(datasetRoot, dataset, 'manifest.json');
      const record = JSON.parse(fs.readFileSync(manifestFile)).rallies.find(r => r.rally_id === rally);
      frames.push({file, kind: 'new', description, dataset, sample: rally, split: 'val', frame, seconds: data.seconds, camera: 'all', viewport, device_scale_factor: 1, url: page.url(), view: {yaw: -0.7, pitch: 0.72, zoom_3d: 1, fit: false}, controls: {subset: await page.locator('#subset').inputValue(), minimum_weight: await page.locator('#threshold').inputValue(), zoom_2d: await page.locator('#zoom').isChecked()}, source_manifest_path: manifestFile, source_manifest_sha256: hash(manifestFile), npz_sha256: record.npz_sha256, screenshot_sha256: hash(path.join(output, file)), observed: {occlusion: data.cameras.map(c => c.occluded), out_of_frame: data.cameras.map(c => c.out_of_frame), presence: data.cameras.map(c => c.presence), prior_only_probability: data.prior_only_probability, convergence: data.convergence}});
    }
    await page.goto(`${base}/?${new URLSearchParams({dataset, split: 'val', rally, frame: 50, zoom: 1})}`);
    await ready(50);
    const catalog = await (await page.request.get(`${base}/api/catalog`)).json();
    assert.equal(catalog.length, 9);
    assert.equal(catalog.find(d => d.id === 'synthetic-3d-i936-smoke-r2-v2').orphan_npz.length, 1);
    assert.equal(catalog.filter(d => !d.available).length, 2);
    checks.push('9 manifest catalog; failed runs and orphan NPZ stay unavailable');
    assert.match(await page.locator('#frame-status').textContent(), /収束 未評価/);
    await capture('01-observed', 50, '遮蔽なし・画面内の同時刻2D候補/3D真値');
    await page.locator('#jump-gap').click();
    await ready(85);
    assert.match(await page.locator('#frame-status').textContent(), /遮蔽 3\/3/);
    assert.match(await page.locator('#gap-note').textContent(), /遮蔽は不存在ではなく/);
    await capture('02-all-camera-gap', 85, '全camera gapでも保存GMM/存在確率が残る');
    checks.push('gap jump, rational time synchronization, 125 full components');
    await change(140);
    assert.match(await page.locator('#frame-status').textContent(), /画面外 2\/3/);
    await capture('03-out-of-frame', 140, '画面外maskと2D位置候補を別々に読む');
    await change(85);
    await page.locator('#subset').selectOption('0');
    await page.locator('#threshold').selectOption('0');
    await ready(85);
    await page.waitForFunction(() => document.getElementById('draw-summary').textContent.startsWith('1/125成分'));
    await capture('04-prior-component', 85, '保存された空camera subsetだけを描画。frame全体がprior-onlyという意味ではない');
    checks.push('empty-camera subset filter retains original tiny probability and omitted mass');
    // A slow old response must not repaint another frame after a newer selection.
    await page.route('**/api/frame?**', async route => {if (new URL(route.request().url()).searchParams.get('frame') === '20') await new Promise(resolve => setTimeout(resolve, 350)); await route.continue();});
    await page.locator('#frame').fill('20'); await page.locator('#frame').dispatchEvent('change');
    await change(21);
    await page.waitForTimeout(500);
    await ready(21);
    checks.push('slow stale frame response cannot overwrite the newer selection');
    await page.unroute('**/api/frame?**');
    await page.locator('#subset').selectOption('all'); await page.locator('#threshold').selectOption('0.001'); await ready(21);
    await page.locator('#zoom').uncheck(); await ready(21);
    await page.locator('#play').click();
    await page.waitForFunction(() => Number(document.getElementById('frame').value) > 21 && document.getElementById('time').textContent.endsWith(' s'));
    await page.locator('#play').click();
    assert.equal(await page.locator('#play').textContent(), '再生');
    checks.push('source view / zoom, threshold, replay / stop, deep-link controls');
    // All three splits and both main review datasets must use saved frame payloads.
    for (const family of [dataset, 'i936-pilot-h-anchored-s936']) {
      for (const split of ['train', 'val', 'test']) {
        await page.goto(`${base}/?${new URLSearchParams({dataset: family, split, rally: `${split}-00000`, frame: 0})}`);
        await ready(0);
        assert.equal(await page.locator('#dataset').inputValue(), family);
        assert.equal(await page.locator('#rally').inputValue(), `${split}-00000`);
      }
    }
    await page.goto(`${base}/?${new URLSearchParams({dataset: 'synthetic-3d-i936-dev-r5', split: 'val', frame: 0})}`);
    await page.waitForFunction(() => !document.getElementById('error').hidden);
    assert.match(await page.locator('#error').textContent(), /登録済みrallyはありません/);
    await page.goto(`${base}/?${new URLSearchParams({dataset, split: 'val', rally: 'val-99999', frame: 0})}`);
    await page.waitForFunction(() => !document.getElementById('error').hidden);
    assert.match(await page.locator('#error').textContent(), /指定rally/);
    checks.push('train/val/test in dev/pilot; missing split and unknown deep-link fail explicitly');
    assert.deepEqual(errors, []);
    const sourceFiles = ['src/tasks/ball_refiner/scripts/review_3d_dataset.py', 'src/tasks/ball_refiner/refiner_3d/review/data.py', 'src/tasks/ball_refiner/refiner_3d/review/web.py', 'src/tasks/ball_refiner/refiner_3d/review/static/index.html', 'src/tasks/ball_refiner/refiner_3d/review/static/app.js', 'src/tasks/ball_refiner/refiner_3d/review/static/style.css'];
    const metadata = {task: 'ball_refiner_3d', captured_utc: new Date().toISOString(), browser: await browser.version(), before: '既存のブラウザUIなし。全画像は新画面。', code_file_hashes: Object.fromEntries(sourceFiles.map(f => [f, hash(f)])), screenshots: frames};
    fs.writeFileSync(path.join(output, 'capture.json'), JSON.stringify(metadata, null, 2) + '\n');
    fs.writeFileSync(path.join(output, 'browser_checks.json'), JSON.stringify({checks, page_errors: errors}, null, 2) + '\n');
    console.log(JSON.stringify({checks, screenshots: frames.map(f => f.file), page_errors: errors}, null, 2));
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode = 1;});
