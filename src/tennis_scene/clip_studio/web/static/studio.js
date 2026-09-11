import {Playback, clamp, stepSeconds, timecode} from './playback.js';
const $ = id => document.getElementById(id);
let project, selected = null, markIn = null, markOut = null, saving = false, view = null, proposal = null;
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
async function api(path, body) {
  const response = await fetch(`/api/${path}`, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
  return result;
}
function bind(id, handler, event = 'click') {
  $(id).addEventListener(event, e => {
    if (event === 'submit') e.preventDefault();
    Promise.resolve().then(() => handler(e)).catch(error => status(error.message, true));
  });
}
const playback = new Playback($('viewers'), (time, playing) => {
  $('time').textContent = timecode(time); $('seek-time').value = time.toFixed(3);
  $('seek').value = time; $('play').textContent = playing ? 'Ⅱ 停止' : '▶ 再生';
  if (playing && view && (time > view[1] || time < view[0])) setView(time - (view[1] - view[0]) * 0.2, view[1] - view[0]);
}, status, index => { $('camera').value = index; playback.select(index); });
function setView(start, span) {
  const [lo, hi] = project.extent;
  span = clamp(span, Math.min(0.25, hi - lo), hi - lo);
  start = clamp(start, lo, hi - span); view = [start, start + span];
  $('seek').min = view[0]; $('seek').max = view[1]; $('seek').value = playback.time;
  $('view-start').textContent = timecode(view[0]); $('view-end').textContent = timecode(view[1]);
  renderTrack();
}
function renderTrack() {
  $('clip-track').replaceChildren();
  project.clips.forEach(clip => {
    const left = Math.max(clip.start_sec, view[0]), right = Math.min(clip.end_sec, view[1]);
    if (right <= left) return;
    const item = document.createElement('button'); item.title = clip.name; item.setAttribute('aria-label', `選択 ${clip.name}`);
    item.style.left = `${100 * (left - view[0]) / (view[1] - view[0])}%`;
    item.style.width = `${100 * (right - left) / (view[1] - view[0])}%`;
    item.onclick = () => selectClip(clip.name); $('clip-track').append(item);
  });
}
function marks() {
  $('in-time').textContent = markIn === null ? '—' : timecode(markIn);
  $('out-time').textContent = markOut === null ? '—' : timecode(markOut);
  $('create').disabled = saving || markIn === null || markOut === null || markOut <= markIn;
}
function clearLoop() { playback.loop = null; $('loop').textContent = '区間リピート'; }
function selectClip(name) {
  selected = name; clearLoop(); renderClips();
  const clip = project.clips.find(c => c.name === name);
  if (clip) { playback.seek(clip.start_sec); setView(clip.start_sec - 1, clip.end_sec - clip.start_sec + 2); }
}
function renderClips() {
  $('count').textContent = `${project.clips.length} clips`; $('clips').replaceChildren();
  if (!project.clips.length) { const hint = document.createElement('p'); hint.className = 'hint'; hint.textContent = '開始 I → 終了 O → 追加 C でラリーを記録'; $('clips').append(hint); }
  [...project.clips].sort((a,b) => a.start_sec - b.start_sec).forEach(clip => {
    const button = document.createElement('button'); button.className = 'clip-row'; button.classList.toggle('active', clip.name === selected);
    const title = document.createElement('span'); title.textContent = clip.name;
    const times = document.createElement('small'); times.textContent = `${timecode(clip.start_sec)} / ${(clip.end_sec - clip.start_sec).toFixed(2)}s`;
    button.append(title, times); button.onclick = () => selectClip(clip.name); $('clips').append(button);
  });
  const clip = project.clips.find(c => c.name === selected);
  $('clip-editor').hidden = !clip;
  if (clip) { $('clip-name').value = clip.name; $('clip-start').value = clip.start_sec.toFixed(6); $('clip-end').value = clip.end_sec.toFixed(6); }
  renderTrack();
}
function renderOffsets() {
  $('offsets').replaceChildren();
  project.sources.forEach((source, index) => {
    const row = document.createElement('div'); row.className = 'offset-row';
    const title = document.createElement('strong'); title.textContent = source.camera_id;
    const input = document.createElement('input'); input.type = 'number'; input.step = '0.001'; input.value = source.offset_sec.toFixed(6);
    input.setAttribute('aria-label', `${source.camera_id} 時差（秒）`);
    const apply = document.createElement('button'); apply.textContent = '時差を適用';
    const update = value => {
      const offsets = project.sources.map(s => s.offset_sec); offsets[index] = value;
      mutate({action: 'offsets', offsets_sec: offsets}).catch(error => status(error.message, true));
    };
    apply.onclick = () => { if (input.value !== '' && Number.isFinite(input.valueAsNumber)) update(input.valueAsNumber); else status('有限の時差を指定してください。', true); };
    row.append(title, input, apply);
    [-1, 1].forEach(sign => { const button = document.createElement('button'); button.textContent = `${sign > 0 ? '+' : '−'}1フレーム`; button.onclick = () => update(source.offset_sec + sign / source.fps); row.append(button); });
    $('offsets').append(row);
  });
}
function renderProject(initial = false) {
  $('video').textContent = `${project.dataset_id} / ${project.video_id}`;
  $('saved').textContent = '保存済み'; $('saved').title = project.projects_path;
  $('undo').disabled = !project.can_undo; $('redo').disabled = !project.can_redo;
  if (initial) {
    ['camera','reference'].forEach(id => {
      $(id).replaceChildren(); project.sources.forEach((source,index) => { const option = new Option(source.camera_id, index); $(id).add(option); });
    });
  }
  playback.load(project);
  if (!view) setView(...[project.extent[0], project.extent[1] - project.extent[0]]);
  else setView(view[0], view[1] - view[0]);
  const [a,b] = project.common;
  $('common').textContent = b > a ? `全カメラ共通 ${timecode(a)} – ${timecode(b)}` : '全カメラ共通の区間なし';
  renderClips(); renderOffsets(); marks();
}
async function mutate(operation) {
  if (saving) throw new Error('保存中です。完了を待って操作してください。');
  saving = true; $('saved').textContent = '保存中…'; marks(); playback.pause(); clearLoop();
  try {
    project = await api('edit', {revision: project.revision, ...operation});
    if (operation.action === 'create') selected = project.clips.at(-1).name;
    if (operation.action === 'update') selected = operation.new_name;
    proposal = null; $('proposal').replaceChildren(); renderProject(); status('変更を保存しました');
  } catch (error) { $('saved').textContent = '保存失敗・操作は未適用'; throw error; }
  finally { saving = false; marks(); }
}
function step(direction, multiplier = 1) {
  clearLoop(); playback.seek(playback.time + direction * stepSeconds($('step').value, project.sources[playback.selected].fps, multiplier));
}
bind('focus', () => setMode(true)); bind('compare', () => setMode(false));
function setMode(focus) {
  playback.setFocus(focus);
  ['focus','compare'].forEach(id => { const active = (id === 'focus') === focus; $(id).classList.toggle('active', active); $(id).setAttribute('aria-pressed', active); });
}
bind('camera', () => playback.select(Number($('camera').value)), 'change');
bind('play', () => playback.play()); bind('rate', () => playback.setRate(Number($('rate').value)), 'change');
bind('back', () => step(-1)); bind('forward', () => step(1));
bind('seek', () => { clearLoop(); playback.seek(Number($('seek').value)); }, 'input');
bind('jump', () => { const time = $('seek-time').valueAsNumber; if (!Number.isFinite(time)) throw new Error('移動先の時刻を入力してください。'); clearLoop(); playback.seek(time); setView(playback.time - (view[1]-view[0])/2, view[1]-view[0]); });
bind('zoom-in', () => setView(playback.time - (view[1]-view[0])/4, (view[1]-view[0])/2));
bind('zoom-out', () => setView(playback.time - (view[1]-view[0]), (view[1]-view[0])*2));
bind('pan-left', () => setView(view[0] - (view[1]-view[0])/2, view[1]-view[0]));
bind('pan-right', () => setView(view[0] + (view[1]-view[0])/2, view[1]-view[0]));
bind('fit', () => setView(project.extent[0], project.extent[1]-project.extent[0]));
bind('mark-in', () => { markIn = playback.time; marks(); }); bind('mark-out', () => { markOut = playback.time; marks(); });
bind('clear', () => { markIn = markOut = null; marks(); });
bind('create', async () => {
  if (markIn === null || markOut === null || markOut <= markIn) throw new Error('開始より後に終了を指定してください。');
  await mutate({action:'create', start_sec:markIn, end_sec:markOut}); markIn = markOut = null; marks();
});
bind('undo', () => mutate({action:'undo'})); bind('redo', () => mutate({action:'redo'}));
bind('reload', async () => { if (saving) throw new Error('保存中です。'); project = await api('project'); clearLoop(); renderProject(); status('保存済みプロジェクトを再読込しました'); });
bind('clip-editor', () => mutate({action:'update', name:selected, new_name:$('clip-name').value, start_sec:$('clip-start').valueAsNumber, end_sec:$('clip-end').valueAsNumber}), 'submit');
bind('set-start', () => { $('clip-start').value = playback.time.toFixed(6); });
bind('set-end', () => { $('clip-end').value = playback.time.toFixed(6); });
bind('delete', () => mutate({action:'delete', name:selected}));
bind('loop', () => {
  if (playback.loop) { clearLoop(); playback.pause(); return; }
  const clip = project.clips.find(c => c.name === selected); playback.loop = clip;
  $('loop').textContent = 'リピート解除'; playback.seekAndPlay(clip.start_sec);
});
async function startJob(body) {
  if (saving) throw new Error('保存完了後に実行してください。');
  proposal = null; $('proposal').replaceChildren();
  await api('jobs', {revision:project.revision, ...body}); await pollJob();
}
bind('auto-sync', () => startJob({kind:'sync', reference:Number($('reference').value)}));
bind('export-one', () => startJob({kind:'export', clip_names:[selected]}));
bind('export-all', () => startJob({kind:'export'})); bind('cancel-job', () => api('jobs/cancel', {}));
let pollTimer;
async function pollJob() {
  clearTimeout(pollTimer);
  try {
    const job = await api('jobs'); const running = job.status === 'running';
    ['auto-sync','export-one','export-all'].forEach(id => { $(id).disabled = running; });
    $('cancel-job').hidden = !running;
    $('cancel-job').textContent = job.kind === 'sync' ? '音声同期をキャンセル' : '書き出しをキャンセル';
    $('job-progress').hidden = !running;
    if (job.frames_total) { $('job-progress').max = job.frames_total; $('job-progress').value = job.frames_completed; } else if (job.total) { $('job-progress').max = job.total; $('job-progress').value = job.completed; } else $('job-progress').removeAttribute('value');
    const labels = {idle:'',running:job.message,done:job.kind === 'sync' ? '同期候補の計算が完了しました' : `${job.completed}クリップを書き出しました。${job.skipped?.length || 0}クリップは同じ内容で出力済みのためスキップしました。`,failed:`失敗: ${job.message}`,cancelled:`キャンセルしました（完了済み ${job.completed || 0}クリップは保持）`};
    $('job').textContent = labels[job.status];
    if (job.status === 'done' && job.kind === 'sync' && proposal !== job.revision) {
      proposal = job.revision; $('proposal').replaceChildren();
      const text = document.createElement('p'); text.textContent = job.offsets_sec.map((offset,i) => `${project.sources[i].camera_id}: ${offset.toFixed(3)}秒 / 信頼度 ${job.confidences[i].toFixed(2)}`).join(' ・ ');
      const apply = document.createElement('button'); apply.textContent = 'この同期候補を適用'; apply.disabled = project.revision !== job.revision;
      apply.onclick = () => { if (project.revision !== job.revision) { status('同期計算後に編集されています。再計算してください。',true); return; } mutate({action:'offsets',offsets_sec:job.offsets_sec}).catch(error => status(error.message,true)); };
      $('proposal').append(text, apply);
    }
    if (running) pollTimer = setTimeout(pollJob, 700);
  } catch(error) { status(error.message,true); pollTimer = setTimeout(pollJob, 2000); }
}
document.addEventListener('keydown', e => {
  if (!project || e.isComposing || e.altKey) return;
  if (e.target.closest('textarea,select,input:not([type=range])') || e.target.isContentEditable) return;
  if (e.key === ' ' && e.target.closest('button')) return;
  if (e.ctrlKey || e.metaKey) {
    if (e.key.toLowerCase() === 'z') { e.preventDefault(); $(e.shiftKey ? 'redo' : 'undo').click(); } return;
  }
  const actions = {' ':'play',i:'mark-in',o:'mark-out',c:'create'};
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); step(e.key === 'ArrowLeft' ? -1 : 1,e.shiftKey ? 10 : 1); }
  else if (e.key === ',' || e.key === '.') { e.preventDefault(); clearLoop(); playback.seek(playback.time + (e.key === ',' ? -1 : 1)/project.sources[playback.selected].fps); }
  else if (actions[e.key.toLowerCase()]) { e.preventDefault(); if (!e.repeat) $(actions[e.key.toLowerCase()]).click(); }
});
try {
  const notice = await api('startup-notice');
  if (notice) {
    $('startup-notice').textContent = notice.message;
    $('startup-notice').classList.toggle('warning', notice.warning);
    $('startup-notice').hidden = false;
  }
  project = await api('project'); renderProject(true); status('単一カメラでラリーを切り出せます。同期確認は「カメラ比較」へ。'); await pollJob(); }
catch(error) { status(error.message,true); }
