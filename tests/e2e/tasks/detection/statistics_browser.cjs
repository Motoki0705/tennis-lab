// Live CPU integration. Requires the pose-approved review UI, no model/GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.STATISTICS_URL || 'http://127.0.0.1:8776';
const output = process.env.SCREENSHOT_DIR || '/tmp/ball-statistics-browser';
fs.mkdirSync(output,{recursive:true});
(async()=>{
 let ready=false;
 for(let attempt=0;attempt<60;attempt++){
  try{const response=await fetch(url,{signal:AbortSignal.timeout(3000)});if(response.ok){ready=true;break;}}catch{}
  await new Promise(resolve=>setTimeout(resolve,1000));
 }
 assert.ok(ready,'Review server did not become ready');
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1500,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.locator('#statistics-open').waitFor({timeout:90000});
  await page.click('#statistics-open');await page.locator('#statistics-dialog').waitFor();
  const resultResponse=page.waitForResponse(r=>r.url().endsWith('/result')&&r.status()===200,{timeout:600000});
  await page.locator('[data-id="run"]').click();
  const response=await resultResponse, report=await response.json();
  await page.waitForFunction(()=>document.querySelector('[data-id="status"]').textContent.startsWith('計算完了'));
  assert.ok(report.clip_count>0);
  assert.deepEqual(report.config.strides,[1,4,8,16,32]);
  const all=report.groups.all;
  assert.ok(all.clip.pooled.counts.frames>=all.selected.pooled.counts.frames);
  assert.ok(all.windows.pooled.counts['windows/stride_1/count']>=all.windows.pooled.counts['windows/stride_32/count']);
  await page.selectOption('[data-id="scope"]','selected');
  await page.selectOption('[data-id="group"]','source/chat_annotation');
  assert.ok(await page.locator('[data-id="clips"] tbody tr').count()>0);
  const clip=report.clips.find(c=>c.source==='chat_annotation');
  await page.getByRole('button',{name:clip.clip_id,exact:true}).click();
  await page.locator('[data-id="detail"] h3').waitFor();
  assert.equal(await page.locator('[data-id="detail"] h3').textContent(),clip.clip_id);
  const frameButton=page.getByRole('button',{name:'画像で確認',exact:true});
  await frameButton.click();
  await page.waitForFunction(()=>!document.getElementById('statistics-dialog').open);
  await page.waitForFunction(id=>document.getElementById('scene-title').textContent===id,clip.clip_id);
  await page.click('#statistics-open');
  await page.selectOption('[data-id="group"]','all');
  const metricsDetails=page.locator('details').filter({has:page.locator('[data-id="metrics"]')});
  await metricsDetails.locator('summary').click();
  await page.selectOption('[data-id="aggregation"]','between_clips');
  await page.locator('[data-id="metric"]').fill('median/pose/left_wrist');
  assert.ok(await page.locator('[data-id="metrics"] tbody tr').count()>0);
  for(const width of [1500,800,390]){
   await page.setViewportSize({width,height:1100});
   await page.waitForTimeout(100);
   assert.ok(await page.locator('#statistics-dialog').evaluate(el=>el.scrollWidth<=el.clientWidth+1),`dialog overflow ${width}`);
   await page.screenshot({path:`${output}/statistics-${width}.png`,fullPage:true});
  }
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:'PASS',clips:report.clip_count,frames:all.clip.pooled.counts.frames,
    selected:all.selected.pooled.counts.frames,pose_clips:report.pose_available_clips,identity:report.identity}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
