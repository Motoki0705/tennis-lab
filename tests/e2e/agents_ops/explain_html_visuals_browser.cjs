// node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
// Uses one browser setup and a real RevisionStore snapshot. No external API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {execFileSync} = require('node:child_process');
const {performance} = require('node:perf_hooks');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '../../..');
const output = process.env.EXPLAIN_VISUAL_OUTPUT || fs.mkdtempSync(path.join(os.tmpdir(), 'explain-visuals-'));

(async () => {
  const setup = performance.now();
  const fixture = JSON.parse(execFileSync(path.join(root, '.venv/bin/python'), ['-m', 'tests.support.agents_ops.explain_html_visuals_browser', output], {cwd: root, encoding: 'utf8'}));
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox']});
  const setupMs = performance.now() - setup;
  try {
    const errors = [], network = [];
    const page = await browser.newPage({viewport: {width: 1100, height: 900}, reducedMotion: 'reduce'});
    await page.addInitScript(() => document.addEventListener('DOMContentLoaded', () => {
      for (const version of ['1', '2']) {
        const rogue = document.createElement('div');
        rogue.setAttribute('data-managed-graph', version);
        rogue.setAttribute('data-definition', '{}');
        document.body.prepend(rogue);
      }
    }, {once: true}));
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) network.push(r.url()); });
    let started = performance.now();
    await page.goto(pathToFileURL(fixture.index).href);
    const graph = page.locator('[data-figure-id="error-plot"]');
    await page.waitForFunction(() => window.ExplainVisualsV2?.version === '2');
    assert.equal(await page.locator('head').evaluate(head => head.firstElementChild?.getAttribute('data-visual-policy')), '1');
    assert.ok(await page.locator('head').evaluate(head => {
      const nodes = [...head.querySelectorAll('meta,script,style,link')];
      return nodes[0].getAttribute('data-visual-policy') === '1';
    }));
    assert.equal(await graph.locator('[data-slider-value]').textContent(), '0 px; Depth = 16 m');
    const slider = graph.locator('input[type="range"]');
    await slider.focus();
    await slider.press('Home');
    assert.equal(await graph.locator('[data-slider-value]').textContent(), '-10 px; Depth = 24 m');
    await slider.press('End');
    assert.equal(await graph.locator('[data-slider-value]').textContent(), '10 px; Depth = 12 m');
    assert.equal(await graph.locator('circle[data-x="10"]').getAttribute('data-y'), '12');
    assert.deepEqual(await graph.locator('[data-values] tr').allTextContents(), ['Depth-1024', 'Depth016', 'Depth1012']);
    const laterGraph = page.locator('[data-figure-id="later-error"]');
    await laterGraph.locator('input[type="range"]').focus();
    await laterGraph.locator('input[type="range"]').press('Home');
    assert.equal(await laterGraph.locator('[data-slider-value]').textContent(), '-10 px; Depth = 24 m');
    assert.equal(await graph.locator('[data-slider-value]').textContent(), '10 px; Depth = 12 m');
    const comparison = page.locator('[data-figure-id="comparison"]');
    assert.equal(await comparison.locator('circle[data-x="7"]').getAttribute('data-y'), '11');
    assert.match(await comparison.textContent(), /Frame \(frame\)/);
    assert.equal(await graph.locator('[role="group"]').getAttribute('aria-label'), 'Depth decreases from 24 to 12 m as disparity error increases.');
    assert.ok(await page.locator('img').evaluateAll(images => images.every(img => img.complete && img.naturalWidth > 0 && img.alt)));
    await graph.screenshot({path: path.join(output, 'observation-error.png')});
    await page.setViewportSize({width: 390, height: 844});
    assert.ok(await graph.locator('svg').evaluate(el => el.getBoundingClientRect().width <= 390));
    assert.ok(await page.locator('body').evaluate(el => el.scrollWidth <= window.innerWidth));
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    const interactiveMs = performance.now() - started;

    started = performance.now();
    for (const candidate of fixture.rejected_cases) {
      if (candidate.exported) {
        const unsafe = await browser.newPage();
        await unsafe.goto(pathToFileURL(candidate.index).href);
        assert.equal(await unsafe.evaluate(() => window.PWNED), undefined, candidate.name);
        await unsafe.close();
      }
      assert.equal(candidate.exported, false, `${candidate.name} must never reach export`);
      assert.equal(candidate.error, 'invalid_document', candidate.name);
    }
    assert.equal(fs.existsSync(path.join(output, 'unexpected')), false);
    assert.ok(!fs.readFileSync(fixture.index, 'utf8').includes('globalThis.PWNED'));
    assert.equal(fixture.rejected, 'invalid_document');
    assert.equal(fixture.current, fixture.revision);
    await page.reload();
    assert.equal(await page.evaluate(() => window.PWNED), undefined);
    assert.equal(await graph.locator('[data-slider-value]').textContent(), '0 px; Depth = 16 m');
    // Disabling JS keeps meaningful initial charts, data, sources and alt text.
    const staticPage = await browser.newPage({javaScriptEnabled: false});
    await staticPage.goto(pathToFileURL(fixture.index).href);
    assert.equal(await staticPage.locator('[data-figure-id="error-plot"] circle').count(), 3);
    assert.match(await staticPage.locator('body').textContent(), /Hypothetical example/);
    assert.equal(await staticPage.locator('[data-asset-id="concept"]').getAttribute('alt'), 'A small blue rectangle.');
    await staticPage.close();
    const corrupted = await browser.newPage();
    await corrupted.addInitScript(() => document.addEventListener('DOMContentLoaded', () => {
      document.querySelector('[data-figure-id="error-plot"] [data-definition]').setAttribute('data-definition', '{}');
    }, {once: true}));
    await corrupted.goto(pathToFileURL(fixture.index).href);
    assert.equal(await corrupted.locator('[data-figure-id="error-plot"] [data-render-error]').getAttribute('data-render-error'), 'invalid_graph');
    await corrupted.locator('[data-figure-id="later-error"] input').focus();
    await corrupted.locator('[data-figure-id="later-error"] input').press('Home');
    assert.equal(await corrupted.locator('[data-figure-id="later-error"] [data-slider-value]').textContent(), '-10 px; Depth = 24 m');
    await corrupted.close();
    const legacy = await browser.newPage();
    await legacy.goto(pathToFileURL(fixture.legacy_index).href);
    assert.equal(await legacy.evaluate(() => window.ExplainVisualsV1?.version), '1');
    await legacy.locator('input').focus();
    await legacy.locator('input').press('Home');
    assert.equal(await legacy.locator('[data-slider-value]').textContent(), '-10 px; Depth = 24 m');
    await legacy.close();
    const rejectedMs = performance.now() - started;
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    console.log(JSON.stringify({passed: 2, setupMs, interactiveMs, rejectedMs, output}));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
