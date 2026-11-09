// Real browser qualification of the exact private bundle; no service credentials.
const path = require('node:path');
const { pathToFileURL, fileURLToPath } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE ||
  path.resolve(__dirname, '../../../../scripts/ci_metrics/node_modules/playwright'));

async function check(index) {
  const issues = [];
  const issue = (message) => issues.push({ rule: 'browser_rendering', message });
  const root = path.dirname(path.resolve(index)) + path.sep;
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH });
  try {
    const context = await browser.newContext({ reducedMotion: 'reduce' });
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.protocol === 'file:' && fileURLToPath(url).startsWith(root)) await route.continue();
      else { issue('A resource request left the staged bundle.'); await route.abort(); }
    });
    const page = await context.newPage();
    page.on('pageerror', () => issue('A script failed while rendering the document.'));
    for (const width of [1280, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.goto(pathToFileURL(index).href, { waitUntil: 'load', timeout: 15000 });
      const defects = await page.evaluate(() => {
        const problems = [];
        if (!document.body || !document.body.innerText.trim() || !document.body.getClientRects().length ||
            getComputedStyle(document.body).visibility !== 'visible' || Number(getComputedStyle(document.body).opacity) === 0) {
          problems.push('The document has no visible readable body.');
        }
        if (document.documentElement.scrollWidth > window.innerWidth + 2) problems.push('The page overflows the viewport horizontally.');
        for (const image of document.images) {
          if (!image.complete || !image.naturalWidth) problems.push('An image failed to load.');
          if (!image.alt.trim()) problems.push('An image is missing alternative text.');
        }
        for (const figure of document.querySelectorAll('figure[data-figure-id]')) {
          const box = figure.getBoundingClientRect();
          if ((!box.width || !box.height) && !figure.closest('details:not([open])')) problems.push('A figure has no visible dimensions.');
        }
        for (const slider of document.querySelectorAll('[data-managed-slider]')) {
          const graph = slider.closest('[data-managed-graph]');
          const output = graph.querySelector('[data-slider-value]');
          const before = output.textContent;
          const step = Number(slider.step);
          slider.value = Number(slider.value) < Number(slider.max)
            ? Number(slider.value) + step : Number(slider.value) - step;
          slider.dispatchEvent(new Event('input', { bubbles: true }));
          if (output.textContent === before) problems.push('A managed slider did not update its displayed value.');
        }
        if (document.querySelector('[data-render-error]')) problems.push('A managed graph reported a rendering error.');
        return problems;
      });
      defects.forEach(issue);
    }
    await context.close();
    return { status: issues.length ? 'failed' : 'passed', issues };
  } finally {
    await browser.close();
  }
}

check(process.argv[2]).then((result) => process.stdout.write(JSON.stringify(result))).catch(() => {
  process.stdout.write(JSON.stringify({ status: 'failed', issues: [
    { rule: 'browser_rendering', message: 'Chromium could not complete the rendering check.' },
  ] }));
  process.exitCode = 1;
});
