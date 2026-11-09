// Pytest owns the real HTTP server and store. One Chromium, five scenarios.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { performance } = require('node:perf_hooks');
const { execFileSync } = require('node:child_process');
const { chromium, request: requestApi } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const config = JSON.parse(fs.readFileSync(process.argv[2] || 0, 'utf8'));
const output = config.output;
fs.mkdirSync(output, { recursive: true });
const errors = [],
  consoleErrors = [],
  consoleEntries = [],
  requests = [],
  externalFinished = [],
  failures = [];
const origin = new URL(config.url).origin;
let browser, apiRequest;
const frame = async (page) => (await page.locator('#document').elementHandle()).contentFrame();
const revision = (page) => page.locator('#revision').getAttribute('data-revision');
const state = (page, value) =>
  page.waitForFunction((value) => document.getElementById('status').dataset.state === value, value);
const changed = (page, old) =>
  page.waitForFunction(
    (old) =>
      document.getElementById('revision').dataset.revision &&
      document.getElementById('revision').dataset.revision !== old,
    old,
  );
const control = async (body) => {
  const result = await apiRequest.post(config.url + '/__test__/control', {
    headers: { 'X-Explain-Token': config.token },
    data: body,
  });
  assert.equal(result.status(), 200);
  return result.json();
};
const release = (request) => control({ action: 'release', request_id: request.request_id });
async function open(documentId, options = {}) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 850 },
    reducedMotion: 'reduce',
    ...options,
  });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text().replaceAll(config.token, '[token]'));
      consoleEntries.push({
        text: message.text().replaceAll(config.token, '[token]'),
        location: message.location(),
        url: page.url().split('#')[0],
      });
    }
  });
  page.on('request', (request) =>
    requests.push({
      url: request.url().replaceAll(config.token, '[token]'),
      method: request.method(),
      token: Boolean(request.headers()['x-explain-token']),
    }),
  );
  page.on('requestfinished', (request) => {
    if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== origin)
      externalFinished.push(request.url());
  });
  page.on('requestfailed', (request) =>
    failures.push({
      url: request.url().replaceAll(config.token, '[token]'),
      error: request.failure()?.errorText,
    }),
  );
  await page.addInitScript(() => {
    if (window !== window.top) {
      // Edge can suppress animation callbacks in the hidden staging frame.
      // Drop callbacks during its connect event so readiness must be independent
      // of animation scheduling. Later visible reading-state events use real RAF.
      const animate = window.requestAnimationFrame.bind(window);
      let connected = false;
      window.requestAnimationFrame = (callback) => (connected ? animate(callback) : 0);
      addEventListener('message', (event) => {
        if (event.data?.channel === 'explain-reader-v1' && event.data.type === 'connect')
          setTimeout(() => {
            connected = true;
          }, 0);
      });
      return;
    }
    window.readerEvents = [];
    addEventListener('message', (event) => {
      if (event.data?.channel === 'explain-reader-v1') window.readerEvents.push(event.data);
    });
  });
  await page.goto(`${config.url}/#document_id=${documentId}&token=${config.token}`);
  await state(page, 'ready');
  return { page, context };
}
async function submit(page, instruction, click = '#send') {
  await page.locator('#instruction').fill(instruction);
  const sent = page.waitForRequest(
    (request) => request.url().endsWith('/requests') && request.method() === 'POST',
  );
  await page.locator(click).click();
  return (await sent).postDataJSON();
}
async function completed(page, old) {
  await changed(page, old);
  await state(page, 'succeeded');
}
async function reading() {
  const { page, context } = await open(
    'reading',
    config.demo
      ? {}
      : { recordVideo: { dir: path.join(output, 'video'), size: { width: 1280, height: 850 } } },
  );
  const first = await revision(page);
  assert.equal(first, config.initial.reading);
  assert.equal(new URL(page.url()).hash, '');
  let body = await frame(page);
  await body.locator('details summary').click();
  await body.locator('input').focus();
  await body.locator('input').press('End');
  for (let step = 0; step < 3; step++) await body.locator('input').press('ArrowLeft');
  await body.evaluate(() => {
    document.getElementById('limits').scrollIntoView();
  });
  await page.waitForFunction(() =>
    window.readerEvents.some(
      (event) =>
        event.type === 'state' &&
        event.state.y > 1000 &&
        event.state.blocks.some(
          (block) =>
            block.content_id === 'limits' && block.inputs[0]?.value === '7',
        ),
    ),
  );
  const before = await body.locator('#limits').evaluate((el) => el.getBoundingClientRect().top);
  const beforeScroll = await body.evaluate(() => scrollY);
  const edit = await submit(page, 'hold add an explanation to the whole page');
  assert.deepEqual(edit.target, { kind: 'document' });
  assert.equal(edit.base_revision, first);
  await state(page, 'running');
  assert.equal(await revision(page), first);
  assert.equal(await body.locator('.answer').count(), 0);
  await body.locator('details summary').click();
  await body.locator('details summary').click();
  await page.screenshot({ path: path.join(output, '01-reading-while-editing.png') });
  await release(edit);
  await completed(page, first);
  body = await frame(page);
  assert.equal(await body.locator('.answer').count(), 1);
  assert.equal(
    await body.locator('details[data-content-id="assumptions"]').getAttribute('open'),
    '',
    'an unchanged details content block must retain its own open state',
  );
  assert.equal(await body.locator('input').inputValue(), '7');
  assert.equal(await body.locator('output').textContent(), '7');
  assert.ok(
    Math.abs(
      (await body.locator('#limits').evaluate((el) => el.getBoundingClientRect().top)) - before,
    ) < 4,
    'the unchanged inner section must stay in place inside a changed article',
  );
  assert.ok(
    (await body.evaluate(() => scrollY)) > beforeScroll,
    'the added paragraph above the reading anchor must change its absolute scroll position',
  );
  assert.equal(await page.locator('.answer').count(), 0, 'answers belong in the document');
  assert.equal(await body.locator('[data-reader-changed="true"]#geometry').count(), 1);
  await page.locator('#changes [data-content-id="geometry"]').click();
  await body.waitForFunction(
    () => Math.abs(document.getElementById('geometry').getBoundingClientRect().top) < 4,
  );
  assert.ok(
    Math.abs(await body.locator('#geometry').evaluate((el) => el.getBoundingClientRect().top)) < 4,
  );
  await page.screenshot({ path: path.join(output, '02-change-navigation.png') });
  const second = await revision(page);
  await page.locator('#history-panel summary').click();
  await page.locator('#undo').click();
  await completed(page, second);
  body = await frame(page);
  assert.equal(await body.locator('.answer').count(), 0);
  const restored = await revision(page);
  assert.notEqual(restored, first);
  await page.screenshot({ path: path.join(output, '03-undo.png') });
  if (config.demo) {
    await context.close();
    return;
  }
  await body.locator('#quote').evaluate((el) => {
    const range = document.createRange();
    range.setStart(el.firstChild, 3);
    range.setEnd(el.firstChild, 22);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
  });
  await page.waitForFunction(() => document.getElementById('target').value === 'selection');
  const selection = await submit(page, 'Explain the selected rays with a concrete example');
  assert.equal(selection.base_revision, restored);
  assert.equal(selection.target.kind, 'selection');
  assert.equal(selection.target.content_id, 'geometry');
  assert.equal(selection.target.exact, 'Intersect two rays.');
  assert.equal(selection.target.start, 20);
  assert.equal(selection.target.end, 39);
  await completed(page, restored);
  body = await frame(page);
  assert.equal(await body.locator('.answer').count(), 1);
  await page.locator('#history li').last().getByText('読む', { exact: true }).click();
  await page.waitForFunction(
    (first) => document.getElementById('revision').dataset.revision === first,
    first,
  );
  assert.equal(await page.locator('#send').isDisabled(), true);
  await page.locator('#refresh').click();
  await changed(page, first);
  await page
    .locator('details')
    .filter({ has: page.locator('#export') })
    .locator('summary')
    .click();
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadEvent;
  const zipPath = path.join(output, 'explanation.zip');
  await download.saveAs(zipPath);
  execFileSync(path.resolve(__dirname, '../../../.venv/bin/python'), [
    '-c',
    'import sys,zipfile; archive=zipfile.ZipFile(sys.argv[1]); assert archive.testzip() is None; assert all(item.date_time >= (1980,1,1,0,0,0) for item in archive.infolist())',
    zipPath,
  ]);
  const zip = fs.readFileSync(zipPath),
    names = [];
  let offset = 0;
  const unpacked = path.join(output, 'export');
  fs.mkdirSync(unpacked, { recursive: true });
  while (zip.readUInt32LE(offset) === 0x04034b50) {
    const length = zip.readUInt32LE(offset + 18),
      nameLength = zip.readUInt16LE(offset + 26),
      extra = zip.readUInt16LE(offset + 28);
    const name = zip.subarray(offset + 30, offset + 30 + nameLength).toString();
    const start = offset + 30 + nameLength + extra,
      bytes = zip.subarray(start, start + length);
    names.push(name);
    const file = path.join(unpacked, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
    offset = start + length;
    assert.ok(!bytes.includes(Buffer.from(config.token)));
  }
  assert.ok(names.includes('assets/court.svg') && names.includes('assets/plot_renderer.mjs'));
  const staticPage = await context.newPage();
  await staticPage.goto(pathToFileURL(path.join(unpacked, 'index.html')).href);
  assert.equal(await staticPage.locator('#edit-form,[data-reader-document]').count(), 0);
  assert.ok(
    await staticPage.locator('img').evaluate((el) => el.complete && el.naturalWidth > 0 && el.alt),
  );
  await staticPage.locator('input').focus();
  await staticPage.locator('input').press('Home');
  for (let step = 0; step < 7; step++) await staticPage.locator('input').press('ArrowRight');
  assert.equal(await staticPage.locator('output').textContent(), '-3');
  await context.close();
}
async function recovery() {
  const { page, context } = await open('recovery');
  let old = await revision(page);
  let posts = 0;
  let acknowledge;
  const accepted = new Promise((resolve) => {
    acknowledge = resolve;
  });
  await page.route('**/api/documents/recovery/requests', async (route) => {
    posts++;
    await route.fetch();
    await route.abort('failed');
    acknowledge();
  });
  const lost = await submit(page, 'hold lost acknowledgement');
  await accepted;
  await context.setOffline(true);
  assert.equal(await revision(page), old);
  assert.ok(
    (await page.evaluate(() => sessionStorage.getItem('explain-html:recovery:pending'))).includes(
      lost.request_id,
    ),
  );
  assert.ok(!(await page.evaluate(() => JSON.stringify(sessionStorage))).includes(config.token));
  await release(lost);
  await page.unroute('**/api/documents/recovery/requests');
  await context.setOffline(false);
  await completed(page, old);
  assert.equal(await page.locator('#notice').textContent(), '');
  assert.equal(posts, 1);
  assert.equal(
    (await control({})).calls.filter((call) => call.request_id === lost.request_id).length,
    1,
  );
  old = await revision(page);
  const held = await submit(page, 'hold reconnect the same tab');
  await state(page, 'running');
  await page.goto(`${config.url}/#document_id=recovery&token=${config.token}`);
  await state(page, 'running');
  await release(held);
  await completed(page, old);
  assert.equal(
    (await control({})).calls.filter((call) => call.request_id === held.request_id).length,
    1,
  );
  // Lose the response before the real restore can publish. The first history
  // lookup must return without this request, then a late commit must still be
  // applied without another POST, including after reconnecting the same tab.
  old = await revision(page);
  const answersBeforeUndo = await (await frame(page)).locator('.answer').count();
  let undoPosts = 0,
    undoReply,
    acknowledgeUndo;
  const undoAccepted = new Promise((resolve) => {
    acknowledgeUndo = resolve;
  });
  await control({ action: 'hold_undo' });
  await page.route('**/api/documents/recovery/undo', async (route) => {
    undoPosts++;
    const request = route.request().postDataJSON();
    undoReply = route.fetch();
    // Assertion cleanup can close this context before the held fetch settles.
    // Keep the original promise for the success path's explicit await below.
    undoReply.catch(() => {});
    await control({ action: 'wait_entered', request_id: request.request_id });
    await route.abort('failed');
    acknowledgeUndo(request);
  });
  const firstHistory = page.waitForResponse(
    (response) => response.url().endsWith('/revisions') && response.status() === 200,
  );
  await page.locator('#history-panel summary').click();
  await page.locator('#undo').click();
  const undo = await undoAccepted;
  const unpublished = (await (await firstHistory).json()).revisions;
  assert.ok(!unpublished.some((row) => row.request_id === undo.request_id));
  await page.waitForFunction(() =>
    document.getElementById('notice').textContent.startsWith('復元'),
  );
  assert.equal(
    JSON.parse(
      await page.evaluate(() => sessionStorage.getItem('explain-html:recovery:pending')),
    )?.request_id,
    undo.request_id,
    'an unpublished undo remains pending after a successful history lookup',
  );
  assert.equal(await revision(page), old);
  assert.equal(await page.locator('#send').isDisabled(), true);
  assert.equal(await page.locator('#undo').isDisabled(), true);
  assert.equal(await page.locator('#reconnect').isVisible(), true);
  await page.locator('#reconnect').click();
  assert.equal(await revision(page), old);
  await page.goto(`${config.url}/#document_id=recovery&token=${config.token}`);
  await state(page, 'reconciling');
  await release(undo);
  assert.equal((await undoReply).status(), 200);
  await completed(page, old);
  assert.equal(await (await frame(page)).locator('.answer').count(), answersBeforeUndo - 1);
  assert.equal(await page.locator('#notice').textContent(), '');
  const history = await apiRequest.get(`${config.url}/api/documents/recovery/revisions`, {
    headers: { 'X-Explain-Token': config.token },
  });
  const published = (await history.json()).revisions.find(
    (row) => row.request_id === undo.request_id,
  );
  assert.equal(published.revision_id, await revision(page));
  assert.equal(undoPosts, 1);
  assert.equal(
    (await control({})).restores.filter((call) => call.request_id === undo.request_id).length,
    1,
  );
  assert.equal(
    await page.evaluate(() => sessionStorage.getItem('explain-html:recovery:pending')),
    null,
  );
  await page.unroute('**/api/documents/recovery/undo');
  old = await revision(page);
  const cancelled = await submit(page, 'hold cancellation');
  await state(page, 'running');
  await page.locator('#cancel').click();
  await state(page, 'cancelled');
  await release(cancelled);
  assert.equal(await revision(page), old);
  const failed = await submit(page, 'hold fail');
  await state(page, 'running');
  await release(failed);
  await state(page, 'failed');
  assert.equal(await revision(page), old);
  const retried = await submit(page, 'hold retry explicitly', '#retry');
  assert.notEqual(retried.request_id, failed.request_id);
  assert.equal(retried.base_revision, old);
  await state(page, 'running');
  await release(retried);
  await completed(page, old);
  old = await revision(page);
  await submit(page, 'image please');
  await state(page, 'failed');
  assert.match(await page.locator('#notice').textContent(), /image_generation_unavailable/);
  assert.equal(await revision(page), old);
  await submit(page, 'interrupt');
  await state(page, 'interrupted');
  assert.equal(await revision(page), old);
  const other = await open('recovery');
  await submit(other.page, 'external add');
  await completed(other.page, old);
  const latest = await revision(other.page);
  await submit(page, 'stale add');
  await state(page, 'rejected');
  assert.equal(await revision(page), old);
  await page.locator('#refresh').click();
  await changed(page, old);
  assert.equal(await revision(page), latest);
  await other.context.close();
  await context.close();
}
async function security() {
  const { page, context } = await open('security', { viewport: { width: 390, height: 844 } });
  const body = await frame(page),
    old = await revision(page);
  assert.equal(await page.locator('#document').getAttribute('sandbox'), 'allow-scripts');
  assert.equal(await body.evaluate(() => self.origin), 'null');
  assert.equal(
    await body.evaluate(() => {
      try {
        return parent.document.body.innerHTML;
      } catch {
        return 'denied';
      }
    }),
    'denied',
  );
  assert.equal(
    await body.evaluate(() => {
      try {
        return localStorage.length;
      } catch {
        return 'denied';
      }
    }),
    'denied',
  );
  const callCount = (await control({})).calls.length;
  const view = await page.evaluate(
    () => window.readerEvents.findLast((event) => event.type === 'state').view,
  );
  await body.evaluate((view) => {
    const info = document.querySelector('[data-reader-document]');
    const message = {
      channel: 'explain-reader-v1',
      document_id: info.dataset.readerDocument,
      revision: info.dataset.readerRevision,
      view,
    };
    parent.postMessage(
      { ...message, type: 'submit', instruction: 'MALICIOUS', token: 'read token' },
      '*',
    );
    parent.postMessage(
      { ...message, type: 'selection', target: { kind: 'content', content_id: 'not-in-document' } },
      '*',
    );
    const script = document.createElement('script');
    script.textContent = 'parent.postMessage({stolen:true},"*");window.PWNED=1';
    document.body.append(script);
  }, view);
  await page.waitForFunction(() =>
    window.readerEvents.some((event) => event.target?.content_id === 'not-in-document'),
  );
  await page.evaluate((old) => {
    postMessage(
      {
        channel: 'explain-reader-v1',
        type: 'selection',
        document_id: 'security',
        revision: old,
        target: { kind: 'content', content_id: 'geometry' },
      },
      '*',
    );
  }, old);
  assert.equal(await page.locator('#target').inputValue(), 'document');
  assert.equal(await body.evaluate(() => window.PWNED), undefined);
  assert.equal((await control({})).calls.length, callCount);
  assert.equal(
    await body.evaluate(async () => {
      try {
        await fetch('/api/documents/security');
        return 'allowed';
      } catch {
        return 'denied';
      }
    }),
    'denied',
  );
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.ok(await body.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.ok(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches));
  assert.ok(await body.locator('img').evaluateAll((images) => images.every((el) => el.alt)));
  await body.evaluate(() => {
    const start = document.querySelector('#quote').firstChild,
      end = document.querySelector('#limits h2').firstChild;
    const range = document.createRange();
    range.setStart(start, 3);
    range.setEnd(end, 4);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
  });
  await page.waitForFunction(() => document.getElementById('target').value === 'missing');
  assert.equal(await page.locator('#send').isDisabled(), true);
  await body.evaluate(() => getSelection().removeAllRanges());
  await body.locator('h2').first().focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => document.getElementById('target').value === 'content:geometry');
  await page.locator('#instruction').focus();
  await page.keyboard.type('Keyboard-only explanation');
  await page.keyboard.press('Tab');
  assert.equal(await page.locator('#send').evaluate((el) => el === document.activeElement), true);
  const sent = page.waitForRequest(
    (request) => request.method() === 'POST' && request.url().endsWith('/requests'),
  );
  await page.keyboard.press('Enter');
  assert.equal((await sent).postDataJSON().target.content_id, 'geometry');
  await completed(page, old);
  await page.screenshot({ path: path.join(output, '04-mobile-keyboard.png') });
  assert.ok(!(await page.content()).includes(config.token));
  assert.equal(await page.evaluate(() => localStorage.length), 0);
  await context.close();
}
async function removed() {
  const { page, context } = await open('removed');
  const old = await revision(page);
  const body = await frame(page);
  await body.locator('#limits h2').focus();
  await page.keyboard.press('Enter');
  await body.locator('#limits').evaluate((el) => el.scrollIntoView());
  await page.waitForFunction(
    () =>
      document.getElementById('target').value === 'content:limits' &&
      window.readerEvents.some(
        (event) => event.type === 'state' && event.state.anchor?.content_id === 'limits',
      ),
  );
  await submit(page, 'remove this section');
  await completed(page, old);
  assert.equal(await (await frame(page)).locator('#limits').count(), 0);
  assert.equal(await page.locator('#target').inputValue(), 'missing');
  assert.equal(await page.locator('#send').isDisabled(), true);
  assert.match(await page.locator('#notice').textContent(), /削除|再構成/);
  assert.equal(await page.locator('#changes button').count(), 0);
  await page.locator('#target').selectOption('document');
  assert.equal(await page.locator('#send').isDisabled(), false);
  await context.close();
}
async function managedRenderer() {
  for (const version of ['2', '1']) {
    const { page, context } = await open(`managed-v${version}`);
    const first = await revision(page);
    let body = await frame(page);
    const graph = body.locator('[data-figure-id="error-plot"]');
    assert.equal(await page.locator('#edit-form').isVisible(), true);
    assert.equal(await page.locator('[data-asset-id]').count(), 0);
    assert.equal(await page.locator('#document').getAttribute('sandbox'), 'allow-scripts');
    assert.equal(await body.evaluate(() => self.origin), 'null');
    assert.equal(
      await body.evaluate((version) => window[`ExplainVisualsV${version}`]?.version, version),
      version,
    );
    const slider = graph.locator('input[type="range"]');
    await slider.focus();
    await slider.press(version === '2' ? 'End' : 'Home');
    const expected = version === '2' ? '10' : '-10';
    const depth = version === '2' ? '12' : '24';
    assert.equal(await slider.inputValue(), expected);
    assert.match(
      await graph.locator('[data-slider-value]').textContent(),
      new RegExp(`${expected} px; Depth = ${depth} m`),
    );
    assert.equal(await graph.locator(`circle[data-x="${expected}"]`).getAttribute('data-y'), depth);
    await page.waitForFunction(
      (expected) =>
        window.readerEvents.some(
          (event) =>
            event.type === 'state' &&
            event.state.blocks.some(
              (block) => block.content_id === 'limits' && block.inputs[0]?.value === expected,
            ),
        ),
      expected,
    );
    await page.locator('#target').selectOption('content:geometry');
    const edit = await submit(page, `hold add explanation beside the real v${version} graph`);
    await state(page, 'running');
    assert.equal(await revision(page), first);
    await release(edit);
    await completed(page, first);
    body = await frame(page);
    const updatedGraph = body.locator('[data-figure-id="error-plot"]');
    assert.equal(await updatedGraph.locator('input').inputValue(), expected);
    assert.match(
      await updatedGraph.locator('[data-slider-value]').textContent(),
      new RegExp(`${expected} px; Depth = ${depth} m`),
    );
    assert.equal(
      await body.evaluate((version) => window[`ExplainVisualsV${version}`]?.version, version),
      version,
    );
    // The real managed hash and fixed bridge may execute. Arbitrary new bytes
    // must still be rejected in the opaque iframe after the reader transformation.
    await body.evaluate(() => {
      const script = document.createElement('script');
      script.textContent = 'window.MANAGED_ESCAPE=true';
      document.body.append(script);
    });
    assert.equal(await body.evaluate(() => window.MANAGED_ESCAPE), undefined);
    assert.equal(
      await body.evaluate(() => {
        try {
          return parent.document.title;
        } catch {
          return 'denied';
        }
      }),
      'denied',
    );
    await page.screenshot({ path: path.join(output, `managed-v${version}.png`) });
    if (version === '2') {
      const current = await revision(page);
      await submit(page, 'image: add a new generated illustration');
      await state(page, 'failed');
      assert.equal(await revision(page), current);
      assert.match(await page.locator('#notice').textContent(), /image_generation_unavailable/);
      const help = page.locator('details').filter({ hasText: 'AI画像の新規生成' });
      await help.locator('summary').click();
      assert.match(await help.innerText(), /初版/);
      assert.match(await help.innerText(), /SVG・グラフ/);
      assert.equal(await page.locator('#send').isEnabled(), true);
    }
    await context.close();
  }
}
(async () => {
  const start = performance.now();
  browser = config.cdp
    ? await chromium.connectOverCDP(config.cdp)
    : await chromium.launch({
        headless: true,
        executablePath: process.env.CHROMIUM_PATH,
        args: ['--no-sandbox'],
      });
  apiRequest = await requestApi.newContext();
  const setupMs = performance.now() - start,
    scenarios = [];
  try {
    for (const operation of config.demo
      ? [reading]
      : [reading, recovery, security, removed, managedRenderer]) {
      const started = performance.now();
      await operation();
      scenarios.push({ name: operation.name, ms: performance.now() - started });
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(externalFinished, []);
    // The control CSP also rejects inline attributes during inert indexing. It
    // remains strict; styles in the actual content iframe must still work.
    const inertStyleWarnings = consoleEntries.filter(
      (entry) =>
        /Applying inline style/.test(entry.text) && entry.location.url.endsWith('/ui/app.js'),
    );
    assert.deepEqual(
      consoleEntries.filter(
        (entry) =>
          /Applying inline style/.test(entry.text) && !entry.location.url.endsWith('/ui/app.js'),
      ),
      [],
    );
    assert.deepEqual(
      consoleErrors.filter(
        (error) =>
          !/net::ERR_FAILED|status of 409|Content Security Policy|Fetch API cannot load/.test(
            error,
          ),
      ),
      [],
    );
    assert.ok(
      requests
        .filter(
          (request) => request.url.includes('/content/') || request.url.includes('/api/exports/'),
        )
        .every((request) => !request.token),
    );
    assert.ok(requests.every((request) => !request.url.includes(config.token)));
    fs.writeFileSync(
      path.join(output, 'browser-log.json'),
      JSON.stringify(
        {
          setupMs,
          scenarios,
          errors,
          consoleErrors,
          consoleEntries,
          inertStyleWarnings,
          failures,
          externalFinished,
          requests,
        },
        null,
        2,
      ),
    );
    console.log(
      JSON.stringify({
        passed: scenarios.length,
        setupMs,
        scenarios,
        consoleErrors,
        failures,
        externalFinished,
        output,
      }),
    );
  } catch (error) {
    for (const context of browser.contexts())
      for (const page of context.pages()) {
        await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
        console.error(
          'Page state:',
          await page
            .locator('body')
            .innerText()
            .catch(() => ''),
          'Console:',
          consoleErrors,
        );
      }
    throw error;
  } finally {
    await apiRequest.dispose();
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
