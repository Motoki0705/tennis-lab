// Exercise the real saved clip_000 production diagnosis and annotation store.
// Requires an already running read-only server; never generates review data.
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const [base, output] = process.argv.slice(2);
  assert(base && output, 'usage: dataset_review_browser.cjs URL OUTPUT_DIRECTORY');
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1250 }, deviceScaleFactor: 1 });
    const errors = [], frames = [], failed = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) failed.push([response.url(), response.status()]); });
    const wait = (caseId, frame) => page.waitForFunction(([c, f]) => document.body.dataset.case === String(c) && document.body.dataset.frame === String(f) && document.body.dataset.loading === 'false', [caseId, frame]);
    async function inspect(caseId, frame) {
      await page.locator('#frame').fill(String(frame));
      await page.locator('#frame').dispatchEvent('change');
      await wait(caseId, frame);
      const titles = await page.locator('.view-head').allTextContents();
      assert.equal(titles.length, 3);
      assert(titles.every(title => title.includes(`f${frame}`)));
      assert.equal(await page.locator('.full').count(), 3);
      assert.equal(await page.locator('.crop').count(), 3);
      const response = await page.request.get(`${base}/api/frame?case=${caseId}&frame=${frame}`);
      const data = await response.json();
      assert.equal(data.frame, frame);
      frames.push({ case: caseId, frame, seconds: data.seconds, evidence: data.evidence, titles });
      return data;
    }
    await page.goto(base);
    await wait(0, 0);
    const catalog = await (await page.request.get(`${base}/api/catalog`)).json();
    assert.equal(catalog.cases[0].clip, 'video_000/clip_000');
    assert.equal(catalog.cases[1].clip, 'video_000/clip_000');
    assert.equal(catalog.cases[0].source_sha256, catalog.cases[1].source_sha256);
    const f4 = await inspect(0, 4);
    assert.deepEqual(f4.evidence.scores.cameras, ['cam0', 'cam1']);
    assert.equal(f4.evidence.scores.costs[0], f4.evidence.scores.costs[1]);
    assert.deepEqual(f4.evidence.scores.supports.slice(0, 2), [1, 1]);
    assert((await page.locator('#decision').innerText()).includes('ambiguous_margin'));
    await page.screenshot({ path: path.join(output, 'new_production_frame4.png') });
    await page.locator('[data-hypothesis="1"]').click();
    assert((await page.locator('#geometry-legend').innerText()).includes('青: H2'));
    await page.locator('[data-hypothesis="0"]').click();
    await page.locator('.full').first().click({ position: { x: 170, y: 110 } });
    const f631 = await inspect(0, 631);
    assert.deepEqual(f631.evidence.scores.cameras, ['cam0', 'cam1', 'cam2']);
    assert.deepEqual(f631.evidence.scores.supports.slice(0, 2), [1, 0]);
    // Reset manually selected crop so capture follows the point again.
    await page.reload(); await wait(0, 0); await inspect(0, 631);
    await page.screenshot({ path: path.join(output, 'new_production_frame631.png') });
    await inspect(0, 1);
    assert((await page.locator('#evidence').innerText()).includes('sample対象外'));
    await page.locator('#next').click(); await wait(0, 2);
    await page.locator('#back').click(); await wait(0, 1);
    await page.locator('#play').click();
    await page.waitForFunction(() => Number(document.body.dataset.frame) >= 3);
    await page.locator('#play').click();
    await page.selectOption('#case', '1'); await wait(1, 0);
    assert((await page.locator('#decision').innerText()).includes('保存実験で採用'));
    assert((await page.locator('#notice').innerText()).includes('独立したhuman GTではありません'));
    await page.screenshot({ path: path.join(output, 'after_annotation_frame0.png') });
    const annotation = await inspect(1, 4);
    assert.equal(annotation.evidence.state, 'not_saved');
    assert.equal(annotation.evidence.scores, null);
    assert.equal(annotation.observing_cameras.length, 3);
    await page.screenshot({ path: path.join(output, 'new_annotation_frame4.png') });
    assert.deepEqual(errors, []); assert.deepEqual(failed, []);
    await fs.writeFile(path.join(output, 'browser-verification.json'), JSON.stringify({ viewport: { width: 1600, height: 1250 }, errors, failed, catalog, frames, result: 'passed' }, null, 2));
    console.log('Passed: synchronized source frames, saved costs/support, unknown states, selection, crop, playback, comparison provenance.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
