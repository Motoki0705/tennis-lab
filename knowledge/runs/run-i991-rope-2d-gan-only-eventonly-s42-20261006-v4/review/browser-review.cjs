const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require('/tmp/coordinate-review-browser/node_modules/playwright');
const out = '/home/kamimura/projects/tennis-lab/outputs/ball_refiner/review/gan-only-20261006';
(async () => {
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
  const page = await browser.newPage({viewport:{width:1600,height:1160}});
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  const ready = () => page.waitForFunction(() => !document.querySelector('#source').classList.contains('busy') && !document.querySelector('#save-config').disabled, {timeout:60000});
  await page.goto('http://127.0.0.1:8786'); await ready();
  await page.click('#refresh'); await ready();
  await page.check('#show-last'); await ready();
  for (const dim of [2,3]) {
    await page.selectOption(`#model-${dim}d`, `outputs:train/rope-${dim}d-gan-only-eventonly/20261006-v4-s42/logs/version_0/checkpoints/last.ckpt`);
    await ready();
  }
  const reply = page.waitForResponse(r => r.url().endsWith('/api/infer'), {timeout:60000});
  await page.click('#infer');
  const response = await reply; assert.equal(response.status(), 200);
  const result = await response.json(); await ready();
  assert.equal(result.source, 'live'); assert(result.scene.prediction_2d && result.scene.prediction_3d);
  for (const dim of [2,3]) assert(result.request[`checkpoint_${dim}d`].endsWith('/last.ckpt'));
  for (const key of ['noise_p95_px','jitter_sigma_px','outlier_probability','isolated_probability']) assert.equal(result.request.augmentation[key],0);
  await page.selectOption('#plot-mode', 'y'); await page.click('#next-event');
  await page.screenshot({path:out+'/last-2d-3d-coordinates.png',fullPage:true});
  await page.selectOption('#plot-mode','speed'); await page.selectOption('#layout','separate');
  await page.screenshot({path:out+'/last-2d-3d-speed.png',fullPage:true});
  assert.deepEqual(errors, []);
  fs.writeFileSync(out+'/last-browser-result.json', JSON.stringify(result));
  const summary = {rally:result.request.rally, source:result.source, request:result.request, input_sha256:result.input_sha256, metrics2d:result.scene.metrics_2d[0], metrics3d:result.scene.metrics_3d, errors};
  fs.writeFileSync(out+'/last-browser-summary.json',JSON.stringify(summary,null,2));
  console.log(JSON.stringify(summary)); await browser.close();
})().catch(error => {console.error(error); process.exit(1);});
