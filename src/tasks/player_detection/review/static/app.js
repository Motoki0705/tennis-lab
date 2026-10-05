const $ = (id) => document.getElementById(id);
const COLORS = { observed: '#10d794', inferred: '#ffbe43', unresolved: '#ad78d9' };
const NAMES = { observed: '観測bbox', inferred: '推定 / 補完bbox', unresolved: '位置未解決' };
const state = { dataset: null, catalogue: [], clips: [], clip: null, frames: [], frame: null, image: null, index: 0, epoch: 0, playing: false, timer: null, zoom: 1, pan: [0, 0] };
const number = (n) => n.toLocaleString('en-US');
const escape = (text) => String(text).replace(/[&<>"']/g, (ch) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));

async function api(path, args = {}) {
  const response = await fetch('/api/' + path + '?' + new URLSearchParams(args));
  if (!response.ok) { const body = await response.json(); throw new Error(body.detail || response.statusText); }
  return response.json();
}
function report(error) { $('error').textContent = error.message; $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
function stop() { state.playing = false; clearTimeout(state.timer); $('play').textContent = '▶ 再生'; }
function identity(frame) { return { dataset: state.dataset.id, clip: state.clip.id, frame }; }
function syncURL() { if (state.frame) history.replaceState(null, '', '#' + new URLSearchParams({...identity(state.frame.frame_index), flag: $('flag').value})); }

function overview() {
  const d = state.dataset;
  $('overview').innerHTML = [
    [number(d.frames), '保存フレーム', 'player記録があるRGB', ''],
    [number(d.clips), 'クリップ / source ' + d.source_videos, 'source_id単位でsplit', ''],
    [number(d.annotation_statuses.partial || 0), 'partial 注釈クリップ', '空間精度は未検証', ''],
    [number(d.counts.unresolved), '位置未解決の選手記録', '座標なし ≠ 選手なし', 'warn'],
  ].map(([value, label, note, cls]) => '<div class="metric ' + cls + '"><strong>' + value + '</strong><div><b>' + label + '</b>' + note + '</div></div>').join('');
  $('build').textContent = 'build ' + d.created_at.slice(0, 10) + ' · annotation snapshot';
  $('dataset-facts').innerHTML = '<b>' + escape(d.schema) + '</b><br>' + Object.entries(d.splits).map(([key, value]) => key + ': ' + number(value.frames) + ' frames / ' + value.sources + ' sources').join('<br>') + '<br>split source overlap: ' + (d.split_source_overlap.length ? escape(d.split_source_overlap.join(', ')) : '0');
}

async function refreshClips(preferredClip, preferredFrame) {
  stop(); clearError();
  const epoch = ++state.epoch;
  const result = await api('clips', {dataset: state.dataset.id, split: $('split').value, source: $('source').value, flag: $('flag').value, search: $('search').value});
  if (epoch !== state.epoch) return;
  state.clips = result.clips;
  $('clip-count').textContent = result.total + ' clips';
  $('clips').innerHTML = result.clips.map((c) => '<button class="clip" data-clip="' + escape(c.id) + '"><span class="tag">' + escape(c.split) + '</span> <span class="muted">#' + String(c.index).padStart(2, '0') + ' · ' + number(c.stored_frames) + ' frames</span><div class="clip-title">' + escape(c.source_title) + '</div><small>' + escape(c.source_id) + '</small><div class="clip-tags"><span class="tag orange">inferred ' + number(c.counts.inferred) + '</span><span class="tag purple">unresolved ' + number(c.counts.unresolved) + '</span></div></button>').join('');
  for (const button of $('clips').querySelectorAll('button')) button.onclick = () => chooseClip(button.dataset.clip).catch(report);
  if (!result.clips.length) { clearFrame('条件に一致するクリップがありません'); return; }
  const requested = preferredClip || (state.clip && result.clips.some((c) => c.id === state.clip.id) ? state.clip.id : result.clips[0].id);
  if (!result.clips.some((c) => c.id === requested)) throw new Error('指定クリップは現在のdataset / filterにありません: ' + requested);
  await chooseClip(requested, preferredFrame);
}

function clearFrame(message) {
  ++state.epoch; stop(); state.clip = null; state.frame = null; state.image = null; state.frames = [];
  $('viewer').getContext('2d').clearRect(0, 0, $('viewer').width, $('viewer').height);
  $('image-empty').textContent = message; $('image-empty').hidden = false;
  $('players').replaceChildren(); $('frame-facts').replaceChildren(); $('eligibility').textContent = message;
  $('sample-title').textContent = message; $('sample-id').textContent = ''; $('sample-meta').textContent = ''; $('split-badge').textContent = '';
  $('frame-label').textContent = ''; $('image-size').textContent = ''; $('gaps').textContent = '';
  $('issue-list').replaceChildren(); $('issue-count').textContent = '';
  $('seek').max = 0; $('seek').value = 0;
  $('timeline').getContext('2d').clearRect(0, 0, $('timeline').width, $('timeline').height); $('timeline-label').textContent = '';
  for (const id of ['previous', 'next', 'play', 'jump', 'seek']) $(id).disabled = true;
}

async function chooseClip(id, requestedFrame) {
  stop(); clearError(); clearFrame('保存済みRGBを読み込んでいます');
  const epoch = ++state.epoch;
  const clip = await api('clip', {dataset: state.dataset.id, clip: id});
  if (epoch !== state.epoch) return;
  state.clip = clip;
  const flag = $('flag').value;
  state.frames = flag === 'all' ? clip.frames : clip.frames.filter((f) => f.flags.includes(flag));
  state.index = 0; state.zoom = 1; state.pan = [0, 0];
  if (requestedFrame !== undefined) {
    const index = state.frames.findIndex((f) => f.frame_index === Number(requestedFrame));
    if (index < 0) { clearFrame('指定frameは保存されていないか、現在の状態filter対象外です'); throw new Error('frame ' + requestedFrame + 'を表示できません。未保存frameは負例とみなしません。'); }
    state.index = index;
  }
  $('sample-meta').textContent = 'CLIP #' + String(clip.index).padStart(2,'0') + ' / SOURCE ' + clip.source_id;
  $('sample-title').textContent = clip.source_title;
  $('sample-id').textContent = clip.id;
  $('split-badge').textContent = clip.split;
  $('issue-count').textContent = '(' + clip.annotation_issues.length + ')';
  $('issue-list').innerHTML = clip.annotation_issues.map((issue) => '<li>' + escape(issue) + '</li>').join('');
  $('gaps').textContent = clip.unstored_frames ? '未保存: ' + clip.unstored_frames + ' / ' + clip.clip_frame_count + ' frames（' + clip.missing_ranges.slice(0, 6).map(([a,b]) => a === b ? a : a + '–' + b).join(', ') + (clip.missing_ranges.length > 6 ? ', …' : '') + '）。player列挙のないframeはこのstoreにありません。完全な負例ではありません。' : '保存: ' + clip.stored_frames + ' / ' + clip.clip_frame_count + ' clip frames · track IDはこのclip内だけで有効';
  for (const button of $('clips').querySelectorAll('button')) button.classList.toggle('selected', button.dataset.clip === id);
  const selected = $('clips').querySelector('.selected');
  if (selected) selected.scrollIntoView({block:'nearest'});
  $('seek').max = Math.max(0, state.frames.length - 1);
  if (!state.frames.length) { clearFrame('このクリップに条件一致frameはありません'); return; }
  await showFrame();
}

function loadImage(url) { return new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => reject(new Error('保存画像を読み込めません')); img.src = url; }); }

async function showFrame() {
  if (!state.clip || !state.frames.length) return;
  const epoch = ++state.epoch;
  const frameIndex = state.frames[state.index].frame_index;
  const args = identity(frameIndex);
  const [frame, img] = await Promise.all([api('frame', args), loadImage('/api/image?' + new URLSearchParams(args))]);
  if (epoch !== state.epoch) return;
  state.frame = frame; state.image = img;
  for (const id of ['previous', 'next', 'play', 'jump', 'seek']) $(id).disabled = false;
  $('image-empty').hidden = true;
  $('viewer').width = frame.width; $('viewer').height = frame.height;
  $('jump').max = state.clip.clip_frame_count - 1; $('jump').value = frame.frame_index;
  $('seek').value = state.index;
  $('frame-label').textContent = 'frame ' + frame.frame_index + ' · ' + frame.seconds.toFixed(2) + ' s';
  $('image-size').textContent = frame.width + ' × ' + frame.height + ' px';
  $('previous').disabled = state.index === 0; $('next').disabled = state.index === state.frames.length - 1;
  renderFacts(); draw(); renderPlayers(); drawTimeline(); syncURL();
}

function renderFacts() {
  const f = state.frame;
  $('annotation-status').textContent = f.annotation_status === 'partial' ? 'partial · 空間精度は未検証' : f.annotation_status + ' · 注釈注意事項を確認';
  const counts = {observed: 0, inferred: 0, unresolved: 0};
  for (const a of f.annotations) counts[a.bbox_source]++;
  const facts = [['Source frame', f.source_frame_index], ['Clip frame', f.frame_index], ['reviewed', f.reviewed ? 'yes' : 'no'], ['target区間', f.is_target ? 'yes' : 'context'], ['observed / inferred', counts.observed + ' / ' + counts.inferred], ['unresolved', counts.unresolved]];
  $('frame-facts').innerHTML = '<div class="facts">' + facts.map(([key,value]) => '<div class="fact"><span>' + key + '</span><strong>' + value + '</strong></div>').join('') + '</div>';
  const reasons = {eligible: '既定の学習選別: 対象', unresolved_player: '既定の学習選別: 除外', unreviewed: '既定の学習選別: 除外', no_visible_box: '既定の学習選別: 除外'};
  const explanation = {eligible:'stride / epoch sampling前の条件を満たします。bbox精度の合格判定ではありません。', unresolved_player:'位置未解決の選手を含むため、frame全体が除外されます。未解決を負例にしません。', unreviewed:'require_reviewed=trueのため、未reviewのframeを除外します。', no_visible_box:'画像内へclip後、最短辺' + state.dataset.selection.min_visible_box_px + ' px以上のboxがありません。'};
  $('eligibility').classList.toggle('excluded', f.selection_reason !== 'eligible');
  $('eligibility').innerHTML = '<strong>' + reasons[f.selection_reason] + '</strong><p>' + explanation[f.selection_reason] + '</p>';
}

function bbox(ctx, a, width) {
  if (!a.bbox_xyxy || !$('boxes').checked) return;
  const [x1,y1,x2,y2] = a.bbox_xyxy;
  ctx.strokeStyle = COLORS[a.bbox_source]; ctx.lineWidth = width;
  ctx.setLineDash(a.bbox_source === 'inferred' ? [width*3, width*2] : []);
  ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
  if (!$('labels').checked) return;
  const fontSize = Math.max(18, state.frame.width / 70);
  ctx.font = '600 ' + fontSize + 'px sans-serif';
  const text = a.track_id + ' · ' + a.bbox_source;
  const tw = ctx.measureText(text).width + 12;
  const tx = Math.max(2, Math.min(x1, state.frame.width-tw-2));
  const ty = Math.max(fontSize+8, Math.min(y1-4, state.frame.height-4));
  ctx.fillStyle = '#102438d9'; ctx.fillRect(tx,ty-fontSize-6,tw,fontSize+9);
  ctx.fillStyle = COLORS[a.bbox_source]; ctx.fillText(text,tx+6,ty-3);
}

function draw() {
  if (!state.image) return;
  const ctx = $('viewer').getContext('2d');
  const {width, height} = state.frame;
  ctx.clearRect(0,0,width,height);
  ctx.save(); ctx.translate(width/2 + state.pan[0], height/2 + state.pan[1]); ctx.scale(state.zoom,state.zoom); ctx.translate(-width/2,-height/2);
  ctx.drawImage(state.image,0,0,width,height);
  for (const a of state.frame.annotations) bbox(ctx,a,Math.max(2.5,width/480)/state.zoom);
  ctx.restore();
}

function renderPlayers() {
  $('players').replaceChildren();
  if (!state.frame) return;
  for (const a of state.frame.annotations) {
    const card = document.createElement('div'); card.className = 'player';
    const info = document.createElement('div'); info.className = 'info';
    const tags = [a.occluded ? '遮蔽あり' : '遮蔽なし', a.truncated ? '切れあり' : '切れなし'];
    const use = a.default_training_box ? '既定選別の教師box候補（sampling前）' : a.visible_box_eligible ? '画像内boxは有効 / frame全体は除外' : '画像内の有効教師boxなし';
    info.innerHTML = '<strong>' + escape(a.track_id) + '</strong><span class="source ' + a.bbox_source + '">' + NAMES[a.bbox_source] + '</span>' + tags.join(' · ') + '<br>' + (a.bbox_xyxy ? a.bbox_xyxy.map((x) => x.toFixed(1)).join(', ') + ' px' : 'bbox_xyxy: null') + '<br><span class="muted">' + use + '</span>';
    if (!a.bbox_xyxy) {
      const unknown = document.createElement('div'); unknown.className = 'unknown'; unknown.innerHTML = '<b>?</b>座標なし<br>選手なしではない'; card.append(unknown);
    } else {
      const canvas = document.createElement('canvas'); canvas.width = 280; canvas.height = 200;
      const [x1,y1,x2,y2] = a.bbox_xyxy;
      const bw = x2-x1, bh = y2-y1;
      const left = Math.max(0,x1-bw*.2), top = Math.max(0,y1-bh*.15);
      const right = Math.min(state.frame.width,x2+bw*.2), bottom = Math.min(state.frame.height,y2+bh*.15);
      const cw = right-left, ch = bottom-top;
      const ctx = canvas.getContext('2d');
      if (cw > 0 && ch > 0) {
        const scale = Math.min(canvas.width/cw,canvas.height/ch);
        const dx = (canvas.width-cw*scale)/2, dy = (canvas.height-ch*scale)/2;
        ctx.drawImage(state.image,left,top,cw,ch,dx,dy,cw*scale,ch*scale);
        if ($('boxes').checked) { ctx.strokeStyle = COLORS[a.bbox_source]; ctx.lineWidth = 2; ctx.setLineDash(a.bbox_source === 'inferred' ? [6,4] : []); ctx.strokeRect(dx+(x1-left)*scale,dy+(y1-top)*scale,bw*scale,bh*scale); }
      } else { ctx.fillStyle = '#b6c4d5'; ctx.font = '16px sans-serif'; ctx.fillText('画像内の領域なし', 60, 100); }
      card.append(canvas);
    }
    card.append(info); $('players').append(card);
  }
}

function drawTimeline() {
  const canvas = $('timeline'); canvas.width = Math.max(400,canvas.clientWidth*2);
  const ctx = canvas.getContext('2d'); const count = state.clip.clip_frame_count;
  ctx.fillStyle = '#e3e9ef'; ctx.fillRect(0,0,canvas.width,16);
  for (const frame of state.clip.frames) { ctx.fillStyle = frame.flags.includes('unresolved') ? '#a779cd' : frame.flags.includes('inferred') ? '#e7b24c' : '#65b596'; ctx.fillRect(frame.frame_index/count*canvas.width,0,Math.max(1,canvas.width/count),16); }
  ctx.fillStyle = '#243e5e'; ctx.fillRect(state.frame.frame_index/count*canvas.width-1,0,3,16);
  $('timeline-label').textContent = number(state.frames.length) + ' 表示対象 / ' + number(state.clip.stored_frames) + ' 保存frames' + ($('flag').value !== 'all' ? ' · 状態filter中' : '');
}

async function chooseDataset(id, clip, frame) {
  stop(); clearError(); clearFrame('保存済みRGBを読み込んでいます');
  const epoch = ++state.epoch;
  const dataset = state.catalogue.find((d) => d.id === id);
  if (!dataset || !dataset.available) throw new Error(dataset ? dataset.reason : 'Unknown dataset: ' + id);
  state.dataset = dataset; state.clip = null; $('dataset').value = id;
  $('source').innerHTML = '<option value="all">すべてのsource</option>'; $('split').value = 'all'; $('search').value = '';
  overview();
  const all = await api('clips', {dataset: id});
  if (epoch !== state.epoch) return;
  const sources = new Map(all.clips.map((c) => [c.source_id,c.source_title]));
  $('source').innerHTML += [...sources].map(([key,title]) => '<option value="' + escape(key) + '">' + escape(key) + ' · ' + escape(title) + '</option>').join('');
  await refreshClips(clip,frame);
}

$('dataset').onchange = () => { $('flag').value = 'all'; chooseDataset($('dataset').value).catch(report); };
for (const id of ['split', 'source', 'flag']) $(id).onchange = () => refreshClips().catch(report);
let searchTimer;
$('search').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => refreshClips().catch(report),180); };
$('seek').oninput = () => { stop(); state.index = Number($('seek').value); showFrame().catch(report); };
$('previous').onclick = () => { stop(); state.index = Math.max(0,state.index-1); showFrame().catch(report); };
$('next').onclick = () => { stop(); state.index = Math.min(state.frames.length-1,state.index+1); showFrame().catch(report); };
$('jump').onchange = () => { stop(); const index = state.frames.findIndex((f) => f.frame_index === Number($('jump').value)); if (index < 0) { report(new Error('このframeは未保存、または現在の状態filter対象外です。負例への置換はしません。')); $('jump').value = state.frame.frame_index; return; } clearError(); state.index = index; showFrame().catch(report); };
$('play').onclick = async () => {
  if (state.playing) { stop(); return; }
  state.playing = true; $('play').textContent = 'Ⅱ 停止 · 5 fps';
  async function tick() {
    if (!state.playing) return;
    if (state.index >= state.frames.length-1) { stop(); return; }
    state.index++; try { await showFrame(); } catch (error) { report(error); stop(); }
    if (state.playing) state.timer = setTimeout(tick,200);
  }
  await tick();
};
for (const id of ['boxes','labels']) $(id).onchange = () => { draw(); renderPlayers(); };
$('reset-view').onclick = () => { state.zoom = 1; state.pan = [0,0]; draw(); };
$('viewer').onwheel = (event) => { event.preventDefault(); state.zoom = Math.max(1,Math.min(8,state.zoom*Math.exp(-event.deltaY*.001))); if (state.zoom === 1) state.pan = [0,0]; draw(); };
let drag;
$('viewer').onpointerdown = (event) => { if (!state.image) return; drag = [event.clientX,event.clientY,...state.pan]; $('viewer').setPointerCapture(event.pointerId); };
$('viewer').onpointermove = (event) => { if (!drag) return; const rect = $('viewer').getBoundingClientRect(); state.pan = [drag[2]+(event.clientX-drag[0])*state.frame.width/rect.width,drag[3]+(event.clientY-drag[1])*state.frame.height/rect.height]; draw(); };
$('viewer').onpointerup = () => { drag = null; };
document.onkeydown = (event) => { if (['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)) return; if (event.key === 'ArrowRight') $('next').click(); if (event.key === 'ArrowLeft') $('previous').click(); };

async function start() {
  const catalogue = await api('catalog'); state.catalogue = catalogue.datasets;
  $('availability').textContent = catalogue.datasets.filter((d) => !d.available).map((d) => d.id + ': ' + d.reason).join(' / ');
  $('dataset').innerHTML = catalogue.datasets.map((d) => '<option value="' + escape(d.id) + '" ' + (d.available ? '' : 'disabled') + '>' + escape(d.id) + (d.available ? '' : ' · unavailable') + '</option>').join('');
  const available = catalogue.datasets.filter((d) => d.available);
  if (!available.length) { clearFrame('対応するローカルplayer frame storeがありません'); report(new Error(catalogue.datasets.length ? catalogue.datasets.map((d) => d.id + ': ' + d.reason).join('\n') : 'data/player_detection に既存データを配置してください。データ生成・自動処理はこの画面では行いません。')); return; }
  await openHash();
}

async function openHash() {
  const available = state.catalogue.filter((d) => d.available);
  const hash = new URLSearchParams(location.hash.slice(1));
  $('flag').value = 'all';
  if (hash.has('flag')) $('flag').value = hash.get('flag');
  if (!$('flag').value) throw new Error('Unknown state filter in URL');
  await chooseDataset(hash.get('dataset') || available[0].id, hash.get('clip') || undefined, hash.has('frame') ? hash.get('frame') : undefined);
}
window.addEventListener('hashchange', () => openHash().catch(report));
start().catch(report);
