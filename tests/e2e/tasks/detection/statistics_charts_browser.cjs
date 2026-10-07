// Exact chart checks against a saved real API response, with no extra statistics run.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const report=JSON.parse(fs.readFileSync(process.env.STATISTICS_REPORT||'/tmp/ball-charts-report.json','utf8'));
const clip=JSON.parse(fs.readFileSync(process.env.STATISTICS_CLIP||'/tmp/ball-charts-clip.json','utf8'));
const picture=fs.readFileSync(process.env.DETECTION_IMAGE||'/tmp/ball-charts-image.jpg');
const root=path.resolve(__dirname,'../../../..');
const staticRoot=path.join(root,'src/tasks/base/visualization/detection/static');
const chartRoot=path.join(root,'src/tasks/ball_detection/visualization/static/statistics');
const output=process.env.SCREENSHOT_DIR||'/tmp/ball-statistics-charts';fs.mkdirSync(output,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1540,height:1100},deviceScaleFactor:2,acceptDownloads:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const dataset='pose-approved/ball-mix-v2-player-pose-v1',scene=`${dataset}::${clip.clip.clip_id}`;
  await page.route('http://statistics-charts.test/**',async route=>{
   const u=new URL(route.request().url());
   if(u.pathname==='/api/catalog')return route.fulfill({json:{task:'ball_detection',title:'Ball Detection',mode:'review',cuda_available:false,play_intervals_available:false,statistics_ui:'/statistics-static/panel.mjs',datasets:[{id:dataset,label:'Pose承認済み',available:true,count:report.clip_count,mode:'temporal'}],checkpoints:[],warnings:[]}});
   if(u.pathname==='/api/scenes')return route.fulfill({json:{items:[{id:scene,label:clip.clip.clip_id,frames:clip.clip.frame_count}],total:1}});
   if(u.pathname==='/api/preview'){const frame=Number(u.searchParams.get('start'));return route.fulfill({json:{scene,frames:clip.clip.frame_count,width:1280,height:720,items:[{index:frame,name:`frame_${frame}.jpg`,gt:{points:[],rasters:[]}}],warnings:[]}});}
   if(u.pathname==='/api/image')return route.fulfill({body:picture,contentType:'image/jpeg'});
   if(u.pathname==='/api/statistics/config')return route.fulfill({json:report.config});
   if(u.pathname==='/api/statistics/jobs')return route.fulfill({status:202,json:{id:'chart-test',state:'complete'}});
   if(u.pathname.endsWith('/result'))return route.fulfill({json:report});
   if(u.pathname.endsWith('/clip'))return route.fulfill({json:clip});
   const isChart=u.pathname.startsWith('/statistics-static/'),name=u.pathname==='/'?'index.html':u.pathname.split('/').at(-1);
   const directory=isChart?chartRoot:staticRoot;
   return route.fulfill({body:fs.readFileSync(path.join(directory,name)),contentType:name.endsWith('.css')?'text/css':name.endsWith('.html')?'text/html':'text/javascript'});
  });
  await page.goto('http://statistics-charts.test/');await page.locator('#statistics-open').waitFor();await page.click('#statistics-open');
  await page.locator('[data-id="run"]').click();
  await page.waitForFunction(()=>document.querySelectorAll('.statistics-chart-grid svg').length===6);
  const card=key=>page.locator(`[data-chart="${key}"]`);
  const values=await card('composition').locator('rect[data-kind]').evaluateAll(elements=>elements.slice(0,5).map(el=>({value:Number(el.dataset.value),n:Number(el.dataset.denominator)})));
  assert.equal(values.reduce((sum,r)=>sum+r.value,0),values[0].n,'stacked categories are exhaustive instances');
  assert.ok(!await card('composition').textContent().then(t=>t.includes('未確認・球注釈なしのフレームは含みません')));
  await page.selectOption('select[aria-label="分布の指標"]','gap');
  await page.selectOption('select[aria-label="分布の対象"]','clip');
  const expected=report.groups.all.clip.between_clips['median/gaps/coordinate_gap/bounded/seconds'].median;
  assert.equal(Number(await card('distribution').locator('circle[data-median]').first().getAttribute('data-median')),expected);
  await page.selectOption('select[aria-label="位置と動き"]','speed');
  const m=report.groups.all.clip.pooled.views;
  let observedCell=null,missingCell=null;
  m['motion/observed/edge_count'].forEach((row,y)=>row.forEach((n,x)=>{if(n>0&&!observedCell)observedCell=[x,y];if(n===0&&!missingCell)missingCell=[x,y];}));
  const [cx,cy]=observedCell;
  assert.equal(Number(await card('spatial').locator(`[data-cell="${cx},${cy}"]`).getAttribute('data-value')),m['motion/observed/speed_sum'][cy][cx]/m['motion/observed/edge_count'][cy][cx]);
  if(missingCell)assert.equal(await card('spatial').locator(`[data-cell="${missingCell.join(',')}"]`).getAttribute('data-value'),'missing');
  await card('spatial').locator(`[data-cell="${cx},${cy}"]`).focus();
  assert.match(await card('spatial').locator('.statistics-tooltip').textContent(),/有効な移動ペア/);
  await page.selectOption('select[aria-label="位置と動き"]','movement');
  assert.ok(await card('spatial').locator('g[data-vector]').count()>0);
  await page.selectOption('select[aria-label="被覆率の分母"]','clip');
  assert.equal(Number(await card('stride').locator('circle[data-stride="16"]').getAttribute('data-coverage')),report.groups.all.windows.pooled.rates['windows/stride_16/coverage'].value);
  await page.selectOption('select[aria-label="被覆率の縦軸"]','zoom');
  assert.match(await card('stride').textContent(),/縦軸は差を見るため拡大/);
  await page.selectOption('select[aria-label="poseの指標"]','spike');
  const wrist=card('pose').locator('rect[data-series="左手首"]');await wrist.focus();
  assert.match(await card('pose').locator('.statistics-tooltip').textContent(),/左手首/);
  const save=page.waitForEvent('download');await card('composition').getByRole('button',{name:/SVG保存/}).click();
  const download=await save,svgFile=path.join(output,'composition.svg');await download.saveAs(svgFile);
  const svg=fs.readFileSync(svgFile,'utf8');assert.match(svg,/<svg[^>]+xmlns=/);assert.match(svg,/viewBox=/);assert.match(svg,/<style>/);assert.match(svg,/球注釈の構成/);assert.match(svg,/<metadata>/);assert.match(svg,/pose_manifest_sha256/);
  const exportPage=await browser.newPage();await exportPage.goto(`file://${svgFile}`);assert.equal(await exportPage.locator('parsererror').count(),0);await exportPage.close();
  const point=card('clip-scatter').locator('[data-clip]');
  const chosen=await point.evaluateAll((elements,id)=>elements.some(el=>el.dataset.clip===id),clip.clip.clip_id);
  assert.ok(chosen,'reported clip has both coordinates for the scatter');
  const target=card('clip-scatter').locator(`[data-clip="${clip.clip.clip_id}"]`);await target.focus();await target.press('Enter');
  await page.locator('[data-id="detail"] h3').waitFor();
  assert.equal(await page.locator('[data-id="detail"] h3').textContent(),clip.clip.clip_id);
  assert.ok(await card('trajectory').locator('circle[data-frame]').count()>0);
  const frame=Number(await card('trajectory').locator('circle[data-frame]').first().getAttribute('data-frame'));
  await card('trajectory').locator('circle[data-frame]').first().focus();await card('trajectory').locator('circle[data-frame]').first().press('Enter');
  await page.waitForFunction(frame=>document.getElementById('frame-name').textContent===`frame_${frame}.jpg`,frame);
  assert.equal(await page.locator('#statistics-dialog').evaluate(el=>el.open),false);
  await page.click('#statistics-open');
  await page.selectOption('[data-id="group"]','split/train');
  await page.selectOption('[data-id="scope"]','selected');
  assert.match(await card('composition').textContent(),/train split · 採用範囲/);
  // Degenerate synthetic inputs: null remains unavailable; a measured zero is a point.
  const degenerates=await page.evaluate(async()=>{
    const {ranges,spatialMap,windowProfile}=await import('/statistics-static/plots.mjs');
    const range=ranges([{label:'none',summary:{n:0,p5:null,median:null,p95:null,mean:null,eligible:2}},{label:'zero',summary:{n:2,p5:0,median:0,p95:0,mean:0,eligible:2}}],{title:'境界値',unit:'秒',key:'test'});
    const spatial=spatialMap([[null,0]],{title:'境界値',unit:'正規化/秒',key:'test'});
    const profile=windowProfile(Array(32).fill(0),Array(32).fill(0),0);
    return {points:range.querySelectorAll('circle[data-median]').length,zero:range.querySelector('circle[data-median]').dataset.median,missing:spatial.querySelector('[data-cell="0,0"]').dataset.value,value:spatial.querySelector('[data-cell="1,0"]').dataset.value,profile:profile.textContent,invalid:/\b(NaN|Infinity)\b/.test(range.innerHTML+spatial.innerHTML+profile.innerHTML)};
  });
  assert.equal(degenerates.points,1);assert.equal(degenerates.zero,'0');assert.equal(degenerates.missing,'missing');assert.equal(degenerates.value,'0');assert.match(degenerates.profile,/採用窓がありません/);assert.equal(degenerates.invalid,false);
  for(const width of [1540,900,390]){
    await page.setViewportSize({width,height:1100});
    await page.locator('#statistics-dialog').evaluate(el=>el.scrollTop=0);await page.waitForTimeout(100);
    assert.ok(await page.locator('#statistics-dialog').evaluate(el=>el.scrollWidth<=el.clientWidth+1),`dialog overflow ${width}`);
    await page.screenshot({path:path.join(output,`charts-${width}.png`),fullPage:true});
  }
  await page.setViewportSize({width:1540,height:1100});
  for(const name of ['composition','spatial','distribution','stride','pose','clip-scatter']){
    await card(name).scrollIntoViewIfNeeded();await card(name).screenshot({path:path.join(output,`${name}.png`)});
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: six chart families, exact denominators/quantiles, scope/source filters, missing versus zero, tooltips/keyboard, SVG export, clip/frame navigation, responsive containment');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
