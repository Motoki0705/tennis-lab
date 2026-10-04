"use strict";
const $ = id => document.getElementById(id);
const state = {catalog: [], sequence: null, frame: null, requested: 0, yaw: -0.7, pitch: 0.72, zoom: 1, fit: false, timer: null, revision: 0, navigation: 0};
const color = {green: "#6ae4b7", blue: "#63b7ff", amber: "#f4b65b", red: "#d86970", pink: "#d9a6fc", grid: "#314153", muted: "#9bafc4"};
const percent = p => p < 0.0001 && p > 0 ? p.toExponential(2) : `${(100 * p).toFixed(2)}%`;
const sum = xs => xs.reduce((a, b) => a + b, 0);
const norm = xs => Math.hypot(...xs);
function el(tag, text, className) {const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e;}
function error(message) {$('error').hidden = false; $('error').textContent = message; $('time').textContent = '—'; stop();}
async function api(route, query = {}) {const response = await fetch(`/api/${route}?${new URLSearchParams(query)}`); const data = await response.json(); if (!response.ok) throw new Error(data.detail || response.statusText); return data;}
function query() {return {dataset: $('dataset').value, rally: $('rally').value};}
function role(row) {if (row.status === 'failed') return '失敗記録'; if (row.status !== 'complete') return '停止・部分生成'; if (row.degradation.startsWith('anchored')) return row.mode === 'pilot' ? '学習規模の確認用' : '開発用'; return row.mode === 'dev' ? '旧入力の比較用' : '過去の動作確認';}
function stop() {clearTimeout(state.timer); state.timer = null; $('play').textContent = '再生';}
function canvas(id) {
  const c = typeof id === 'string' ? $(id) : id; const ratio = window.devicePixelRatio || 1;
  const width = c.clientWidth, height = c.clientHeight; c.width = Math.round(width * ratio); c.height = Math.round(height * ratio);
  const ctx = c.getContext('2d'); ctx.scale(ratio, ratio); return {ctx, width, height};
}
function line(ctx, points, stroke, width = 1, dash = []) {ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath(); let begun = false; for (const p of points) {if (!p) {begun = false; continue;} if (!begun) {ctx.moveTo(...p); begun = true;} else ctx.lineTo(...p);} ctx.stroke(); ctx.setLineDash([]);}
function cross(ctx, p, stroke, radius = 5) {line(ctx, [[p[0] - radius, p[1]], [p[0] + radius, p[1]]], stroke, 2); line(ctx, [[p[0], p[1] - radius], [p[0], p[1] + radius]], stroke, 2);}
function ellipse(ctx, mean, cov, transform, stroke, alpha, filled = false) {
  const a = cov[0][0], b = cov[0][1], d = cov[1][1], delta = Math.hypot(a - d, 2 * b);
  const v1 = Math.sqrt(Math.max(0, (a + d + delta) / 2)), v2 = Math.sqrt(Math.max(0, (a + d - delta) / 2));
  const angle = Math.atan2(2 * b, a - d) / 2, ca = Math.cos(angle), sa = Math.sin(angle);
  const points = Array.from({length: 65}, (_, j) => {const t = 2 * Math.PI * j / 64, x = 2 * v1 * Math.cos(t), y = 2 * v2 * Math.sin(t); return transform([mean[0] + x * ca - y * sa, mean[1] + x * sa + y * ca]);});
  ctx.globalAlpha = alpha; line(ctx, points, stroke, 1.4); if (filled) {ctx.fillStyle = stroke; ctx.globalAlpha = alpha * 0.12; ctx.fill();} ctx.globalAlpha = 1;
}
function selection() {
  const n = $('subset').value, threshold = Number($('threshold').value);
  return state.frame.weights.map((w, i) => i).filter(i => (n === 'all' || sum(state.frame.subsets[i].map(Number)) === Number(n)) && state.frame.weights[i] >= threshold);
}
function renderTimeline() {
  if (!state.sequence) return; const {ctx, width, height} = canvas('timeline'), s = state.sequence, t = s.timestamps_seconds.length, label = 54, row = 14;
  ctx.font = '10px system-ui'; ctx.fillStyle = '#111b27'; ctx.fillRect(label, 0, width - label, height);
  for (let v = 0; v < s.occlusion.length; v++) {ctx.fillStyle = color.muted; ctx.fillText(`cam${v}`, 5, 13 + row * v); for (let i = 0; i < t; i++) {const x = label + i / t * (width - label), dx = (width - label) / t + 0.6; if (s.occlusion[v][i]) {ctx.fillStyle = color.amber; ctx.fillRect(x, 4 + row * v, dx, 5);} if (s.out_of_frame[v][i]) {ctx.fillStyle = color.red; ctx.fillRect(x, 9 + row * v, dx, 5);}}}
  const ey = 4 + s.occlusion.length * row; ctx.fillStyle = color.muted; ctx.fillText('event', 5, ey + 9);
  for (let i = 0; i < t; i++) if (s.event_region[i]) {ctx.fillStyle = color.pink; ctx.fillRect(label + i / t * (width - label), ey, (width - label) / t + 0.6, 8);}
  const x = label + state.requested / Math.max(1, t - 1) * (width - label); line(ctx, [[x, 0], [x, height]], color.green, 2);
}
function sceneProjection(width, height) {
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  let extent = 38; if (state.fit && state.frame) {extent = Math.max(extent, ...selection().map(i => norm(state.frame.means_m[i]) * 2 + 4));}
  const scale = Math.min(width / extent, height / (extent * 0.64)) * state.zoom;
  return p => {const u = p[0] * cy - p[1] * sy, v = p[0] * sy + p[1] * cy; return [width / 2 + u * scale, height * 0.57 + (v * sp - p[2] * cp) * scale];};
}
function renderScene() {
  if (!state.frame) return; const f = state.frame, {ctx, width, height} = canvas('scene'), project = sceneProjection(width, height);
  for (let y = -15; y <= 15; y += 5) line(ctx, [project([-10, y, 0]), project([10, y, 0])], '#233447');
  for (let x = -10; x <= 10; x += 5) line(ctx, [project([x, -15, 0]), project([x, 15, 0])], '#233447');
  const court = [[-5.485, -11.885, 0], [5.485, -11.885, 0], [5.485, 11.885, 0], [-5.485, 11.885, 0], [-5.485, -11.885, 0]];
  line(ctx, court.map(project), '#58768d', 1.5);
  for (const x of [-4.115, 4.115]) line(ctx, [project([x, -11.885, 0]), project([x, 11.885, 0])], '#58768d');
  for (const y of [-6.4, 0, 6.4]) line(ctx, [project([-4.115, y, 0]), project([4.115, y, 0])], '#58768d');
  line(ctx, [project([0, -6.4, 0]), project([0, 6.4, 0])], '#58768d');
  line(ctx, [project([-5.485, 0, 0.914]), project([5.485, 0, 0.914])], '#58768d', 1, [3, 3]);
  ctx.font = '11px system-ui'; for (const [p, label, stroke] of [[[8, 0, 0], 'X [m]', color.blue], [[0, 14, 0], 'Y [m]', color.muted], [[0, 0, 8], 'Z [m]', color.pink]]) {line(ctx, [project([0, 0, 0]), project(p)], stroke); const s = project(p); ctx.fillStyle = stroke; ctx.fillText(label, s[0] + 5, s[1]);}
  line(ctx, state.sequence.truth_m.map(project), '#327761', 1.6);
  let offscreen = 0;
  for (const i of selection().sort((a, b) => f.weights[a] - f.weights[b])) {
    const mu = f.means_m[i], L = f.scale_tril_m[i], w = f.weights[i], prior = !f.subsets[i].some(Boolean), stroke = prior ? color.pink : color.blue, center = project(mu);
    if (center[0] < 0 || center[0] > width || center[1] < 0 || center[1] > height) offscreen++;
    ctx.globalAlpha = prior ? 0.7 : Math.max(0.14, Math.min(0.8, Math.sqrt(w) * 1.3));
    for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) {
      const points = Array.from({length: 49}, (_, j) => {const t = j / 48 * 2 * Math.PI, vec = [0, 0, 0]; vec[a] = 2 * Math.cos(t); vec[b] = 2 * Math.sin(t); return project(mu.map((x, d) => x + sum(L[d].map((value, col) => value * vec[col]))));});
      line(ctx, points, stroke, 0.8);
    }
    ctx.globalAlpha = 1; ctx.fillStyle = stroke; ctx.beginPath(); ctx.arc(...center, 1.7 + 3 * Math.sqrt(w), 0, 2 * Math.PI); ctx.fill();
  }
  const mean = f.means_m[0].map((_, d) => sum(f.weights.map((w, i) => w * f.means_m[i][d])));
  cross(ctx, project(mean), '#fff', 4); cross(ctx, project(f.truth_m), color.green, 7);
  ctx.font = '11px ui-monospace,monospace'; ctx.fillStyle = color.green; const tp = project(f.truth_m); ctx.fillText(`GT (${f.truth_m.map(x => x.toFixed(2)).join(', ')})`, tp[0] + 10, tp[1] - 6);
  const ids = selection(), mass = sum(ids.map(i => f.weights[i])); $('draw-summary').textContent = `${ids.length}/${f.weights.length}成分 · 表示mass ${percent(mass)} · 省略mass ${percent(Math.max(0, 1 - mass))}${offscreen ? ` · ${offscreen}中心がviewport外` : ''}`;
  $('mean-error').textContent = `${norm(mean.map((x, d) => x - f.truth_m[d])).toFixed(2)} m`;
}
function renderStatus() {
  const f = state.frame, hidden = f.cameras.filter(c => c.occluded).length, outside = f.cameras.filter(c => c.out_of_frame).length, views = f.cameras.length;
  const labels = [['遮蔽', `${hidden}/${views}`, hidden > 0], ['画面外', `${outside}/${views}`, outside > 0], ['収束', {unassessed: '未評価', nonconverged: '未収束', converged: '評価済・収束', not_saved: '診断未保存'}[f.convergence], f.convergence !== 'converged']];
  $('frame-status').replaceChildren(...labels.map(([a, b, warn]) => el('span', `${a} ${b}`, `pill${warn ? ' warn' : ''}`)), el('span', f.event_region ? 'event ±5' : f.free_flight ? 'free flight' : '境界 / その他', 'pill'));
  $('prior-mass').textContent = percent(f.prior_only_probability);
  $('gap-note').textContent = hidden === views ? '全cameraのevidence gap。保存2D分布は残ります。遮蔽は不存在ではなく、prior-only massとも別の状態です。' : '存在確率はcamera内amodal存在。画面外・遮蔽maskは別々に保存されています。';
  $('subset-mass').replaceChildren(...f.subset_mass.map(s => {const row = el('div', undefined, 'mass-row'); row.append(el('span', s.cameras.some(Boolean) ? s.cameras.map((v, i) => v ? `c${i}` : '').filter(Boolean).join('+') : '∅ prior'), el('b', percent(s.mass))); return row;}));
  $('components').replaceChildren(...f.weights.map((w, i) => i).sort((a, b) => f.weights[b] - f.weights[a]).map(i => {const row = el('tr', undefined, f.subsets[i].some(Boolean) ? '' : 'prior-row'); for (const value of [i, f.subsets[i].map((v, j) => v ? j : '').filter(x => x !== '').join('+') || '∅', percent(f.weights[i]), f.methods[i]]) row.append(el('td', String(value))); return row;}));
}
function renderCameras() {
  if (!state.frame) return;
  const views = state.frame.cameras;
  if ($('cameras').children.length !== views.length) {$('cameras').replaceChildren(...views.map(v => {const card = el('div', undefined, 'camera'); card.append(el('div', undefined, 'camera-header'), el('canvas'), el('div', undefined, 'camera-footer')); return card;}));}
  for (const [index, view] of views.entries()) {
    const card = $('cameras').children[index], title = card.querySelector('.camera-header');
    title.replaceChildren(el('b', `cam${view.id}`), el('span', `presence ${percent(view.presence)}`), el('span', view.occluded ? '遮蔽 / gap' : '遮蔽なし', `pill${view.occluded ? ' warn' : ''}`), el('span', view.out_of_frame ? '画面外 / 背面' : '画面内', `pill${view.out_of_frame ? ' warn' : ''}`));
    const {ctx, width, height} = canvas(card.querySelector('canvas')), size = view.size_wh, zoom = $('zoom').checked;
    let bounds = [0, size[0] - 1, 0, size[1] - 1];
    if (zoom) {
      const xs = view.means_px.map(p => p[0]), ys = view.means_px.map(p => p[1]); if (view.truth_px) {xs.push(view.truth_px[0]); ys.push(view.truth_px[1]);}
      const sx = Math.max(25, ...view.covariance_px2.map(c => Math.sqrt(c[0][0]) * 2)), sy = Math.max(25, ...view.covariance_px2.map(c => Math.sqrt(c[1][1]) * 2));
      bounds = [Math.min(...xs) - sx, Math.max(...xs) + sx, Math.min(...ys) - sy, Math.max(...ys) + sy];
    }
    const padding = 24, scale = Math.min((width - 2 * padding) / (bounds[1] - bounds[0]), (height - 2 * padding) / (bounds[3] - bounds[2])), ox = (width - scale * (bounds[1] - bounds[0])) / 2, oy = (height - scale * (bounds[3] - bounds[2])) / 2;
    const transform = p => [ox + (p[0] - bounds[0]) * scale, oy + (p[1] - bounds[2]) * scale];
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, width, height); ctx.clip();
    const corners = [[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1], [0, 0]].map(transform); line(ctx, corners, '#46617a', 1.2);
    line(ctx, view.court_px.map(p => p && transform(p)), '#314e51', 1);
    for (const [i, mean] of view.means_px.entries()) {
      ellipse(ctx, mean, view.covariance_px2[i], transform, color.blue, Math.max(0.3, Math.sqrt(view.weights[i])), true);
      const p = transform(mean); ctx.fillStyle = color.blue; ctx.beginPath(); ctx.arc(...p, 2.5, 0, 2 * Math.PI); ctx.fill(); ctx.font = '10px system-ui'; ctx.fillText(`k${i} ${percent(view.weights[i])}`, p[0] + 5, p[1] + 12);
    }
    if (view.truth_px) {const p = transform(view.truth_px); cross(ctx, p, color.green, 6); ctx.fillStyle = color.green; ctx.font = '11px system-ui'; ctx.fillText('GT投影', p[0] + 8, p[1] - 5); if (p[0] < 0 || p[0] > width || p[1] < 0 || p[1] > height) {ctx.fillStyle = color.red; ctx.fillText('GT投影はviewport外 → 分布周辺を拡大', 14, height - 12);}}
    else {ctx.fillStyle = color.red; ctx.fillText('GT投影: 背面 / 投影未定義', 14, height - 12);}
    ctx.restore(); ctx.fillStyle = color.muted; ctx.font = '10px ui-monospace,monospace'; ctx.fillText(`x ${bounds[0].toFixed(0)}…${bounds[1].toFixed(0)} px`, 8, 13); ctx.fillText(`y ${bounds[2].toFixed(0)}…${bounds[3].toFixed(0)} px`, 8, height - 5);
    card.querySelector('.camera-footer').textContent = `${size.join('×')} source · ${view.weights.length}成分 · GT ${view.truth_px ? view.truth_px.map(x => x.toFixed(1)).join(', ') + ' px' : 'N/A（背面）'}`;
  }
}
function render() {renderTimeline(); if (state.frame) {renderScene(); renderStatus(); renderCameras();}}
async function setFrame(index) {
  const revision = ++state.revision;
  if (!state.sequence) return;
  if (!Number.isInteger(index) || index < 0 || index >= state.sequence.record.frames) {clearPanels(); error('指定Frameは保存範囲外です。'); return;}
  state.requested = Math.min(state.sequence.timestamps_seconds.length - 1, Math.max(0, Math.floor(index))); $('frame').value = state.requested; $('slider').value = state.requested;
  // Clear the previous frame while loading so a mixed-time view is never presented.
  clearPanels(); $('time').textContent = '読み込み中'; renderTimeline(); $('error').hidden = true;
  try {const f = await api('frame', {...query(), frame: state.requested}); if (revision !== state.revision) return; state.frame = f; $('time').textContent = `${f.seconds.toFixed(4)} s`; render();
    const url = new URL(location.href); for (const [key, value] of Object.entries({...query(), split: $('split').value, frame: state.requested, subset: $('subset').value, threshold: $('threshold').value, zoom: $('zoom').checked ? '1' : '0'})) url.searchParams.set(key, value); history.replaceState(null, '', url);
  } catch (e) {if (revision === state.revision) {clearPanels(); error(e.message);}}
}
function clearPanels() {state.frame = null; for (const id of ['scene', 'timeline']) {const {ctx, width, height} = canvas(id); ctx.clearRect(0, 0, width, height);} for (const card of $('cameras').children) {const {ctx, width, height} = canvas(card.querySelector('canvas')); ctx.clearRect(0, 0, width, height); card.querySelector('.camera-header').textContent = 'データ読み込み中'; card.querySelector('.camera-footer').textContent = '—';} $('components').replaceChildren(); $('frame-status').replaceChildren(); $('subset-mass').replaceChildren(); $('draw-summary').textContent = ''; $('prior-mass').textContent = '—'; $('mean-error').textContent = '—';}
async function loadRally(initialFrame = 0) {
  stop(); ++state.revision; const navigation = ++state.navigation; state.sequence = null; clearPanels(); if (!$('rally').value) {error('このsplitに登録済みrallyはありません。Dataset一覧の状態と計画数を確認してください。'); return;}
  try {const s = await api('sequence', query()); if (navigation !== state.navigation) return; state.sequence = s; $('slider').max = s.record.frames - 1; $('frame').max = s.record.frames - 1; $('identity').textContent = `${s.record.geometry_clip} · seed ${s.record.seed} · ${s.record.frames} frame`; $('provenance').textContent = `Manifest: ${s.manifest_path}\nSHA256: ${s.manifest_sha256}\nNPZ: ${s.rally}.npz\nSHA256: ${s.record.npz_sha256}\n2D: uv→Dμ / DΣDᵀ、D=diag(W−1,H−1)\n3D: [m], covariance [m²]。真値はsimulator、候補は保存三角測量。\nFPS: 60000/1001。eventはnative秒→最寄りframe。\n撮影時はURLのdataset/rally/frameとviewportを保存してください。`; await setFrame(initialFrame);} catch (e) {if (navigation === state.navigation) error(e.message);}
}
async function loadDataset(initialRally, initialFrame = 0) {
  stop(); ++state.revision; const navigation = ++state.navigation; state.sequence = null; clearPanels(); const row = state.catalog.find(x => x.id === $('dataset').value); $('dataset-state').textContent = `${role(row)} / ${row.status}`; $('dataset-state').className = `pill${row.status !== 'complete' ? ' warn' : ''}`; $('dataset-count').textContent = `登録 ${row.counts.train} / ${row.counts.val} / ${row.counts.test} · ${row.frames.toLocaleString()} frame`; $('degradation').textContent = row.degradation.startsWith('anchored') ? '2D入力の誤差: 保存モデル分布由来（較正fit frameを再使用）' : row.degradation.startsWith('provisional') ? '2D入力の誤差: 旧モデル分布由来（暫定）' : '2D入力の誤差: 仮定した劣化（実測ではない）';
  try {const rallies = await api('rallies', {dataset: row.id, split: $('split').value}); if (navigation !== state.navigation) return; $('rally').replaceChildren(...rallies.map(r => {const option = el('option', `${r.id} · ${r.frames} f`); option.value = r.id; return option;})); if (initialRally) {if (!rallies.some(r => r.id === initialRally)) throw new Error('指定rallyは選択splitに登録されていません。'); $('rally').value = initialRally;} await loadRally(initialFrame);} catch (e) {if (navigation === state.navigation) error(e.message);}
}
function jump(condition) {if (!state.sequence) return; const t = state.sequence.timestamps_seconds.length, matches = Array.from({length: t}, (_, i) => i).filter(i => condition(i) && (i === 0 || !condition(i - 1))); const next = matches.find(i => i > state.requested) ?? matches[0]; if (next !== undefined) {stop(); setFrame(next);} else {$('gap-note').textContent = 'このrallyに該当する状態はありません。';}}
$('dataset').onchange = () => loadDataset(); $('split').onchange = () => loadDataset(); $('rally').onchange = () => loadRally();
$('frame').onchange = e => {stop(); setFrame(Number(e.target.value));}; $('slider').oninput = e => {stop(); setFrame(Number(e.target.value));};
$('previous').onclick = () => {stop(); setFrame(Math.max(0, state.requested - 1));}; $('next').onclick = () => {stop(); if (state.sequence) setFrame(Math.min(state.sequence.record.frames - 1, state.requested + 1));};
$('jump-gap').onclick = () => jump(i => state.sequence.occlusion.every(v => v[i])); $('jump-out').onclick = () => jump(i => state.sequence.out_of_frame.some(v => v[i])); $('jump-event').onclick = () => jump(i => state.sequence.record.events.some(e => ['hit', 'bounce'].includes(e.kind) && e.frame === i));
$('play').onclick = async () => {if (state.timer !== null) {stop(); return;} if (!state.sequence) return; $('play').textContent = '停止'; const tick = async () => {await setFrame((state.requested + 1) % state.sequence.record.frames); if (state.timer !== null && state.frame) state.timer = setTimeout(tick, 120);}; state.timer = setTimeout(tick, 0);};
$('timeline').onclick = event => {if (!state.sequence) return; stop(); setFrame(Math.min(state.sequence.record.frames - 1, Math.max(0, Math.round((event.offsetX - 54) / ($('timeline').clientWidth - 54) * (state.sequence.record.frames - 1)))));};
for (const id of ['subset', 'threshold', 'zoom']) $(id).onchange = () => {if (state.frame) setFrame(state.requested);};
$('reset-view').onclick = () => {state.yaw = -0.7; state.pitch = 0.72; state.zoom = 1; state.fit = false; renderScene();}; $('fit-view').onclick = () => {state.fit = true; state.zoom = 1; renderScene();};
let drag = null; $('scene').onpointerdown = event => {drag = [event.clientX, event.clientY]; $('scene').setPointerCapture(event.pointerId);}; $('scene').onpointermove = event => {if (!drag) return; state.yaw += (event.clientX - drag[0]) * 0.007; state.pitch = Math.max(0.1, Math.min(1.45, state.pitch + (event.clientY - drag[1]) * 0.006)); drag = [event.clientX, event.clientY]; renderScene();}; $('scene').onpointerup = () => {drag = null;}; $('scene').onwheel = event => {event.preventDefault(); state.zoom = Math.min(4, Math.max(0.2, state.zoom * Math.exp(-event.deltaY * 0.001))); renderScene();};
window.addEventListener('resize', render);
async function initialize() {
  try {
    state.catalog = await api('catalog');
    $('dataset').replaceChildren(...state.catalog.map(r => {const option = el('option', `${r.id} · ${r.status}`); option.value = r.id; option.disabled = !r.available; return option;}));
    $('catalog').querySelector('tbody').replaceChildren(...state.catalog.map(r => {const row = el('tr'); for (const value of [`${r.id} / ${r.schema.split('.').at(-1)}`, `${r.status} / ${r.mode}`, `${r.counts.train} / ${r.counts.val} / ${r.counts.test} (計画 ${r.planned_counts.train}/${r.planned_counts.val}/${r.planned_counts.test})`, r.frames.toLocaleString(), `${r.orphan_npz.length} / ${r.missing.length}`]) row.append(el('td', value)); return row;}));
    const p = new URLSearchParams(location.search); if (p.has('dataset')) {const requested = state.catalog.find(r => r.id === p.get('dataset')); if (!requested || !requested.available) throw new Error('指定Datasetは存在しないか登録rallyを提供できません。'); $('dataset').value = requested.id;}
    else {$('dataset').value = state.catalog.find(r => r.available && r.degradation.startsWith('anchored') && r.mode === 'dev')?.id || state.catalog.find(r => r.available)?.id || '';}
    if (!$('dataset').value) throw new Error('表示可能な登録rallyがありません。');
    if (p.has('split')) {if (!['train', 'val', 'test'].includes(p.get('split'))) throw new Error('Unknown split'); $('split').value = p.get('split');}
    for (const id of ['subset', 'threshold']) if (p.has(id)) {if (!Array.from($(id).options).some(o => o.value === p.get(id))) throw new Error(`Unknown ${id}`); $(id).value = p.get(id);}
    $('zoom').checked = p.get('zoom') === '1'; await loadDataset(p.get('rally'), Number(p.get('frame') || 0));
  } catch (e) {error(e.message);}
}
initialize();
