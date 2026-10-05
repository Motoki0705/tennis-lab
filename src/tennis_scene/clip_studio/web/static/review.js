import {timecode} from './playback.js';
const $ = id => document.getElementById(id);
const labels = {indexed:'登録済み',unexported:'未出力',incomplete:'出力不足',conflict:'不一致',invalid:'確認失敗',unregistered:'未登録',index_unknown:'登録未確認'};
const node = (tag, text, className = '') => {
  const element = document.createElement(tag); element.textContent = text; element.className = className; return element;
};
async function get(path, signal) {
  const response = await fetch(`/api/${path}`, {signal});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
  return result;
}
export class DatasetReview {
  constructor(onStatus) { this.onStatus = onStatus; this.loadEpoch = 0; this.frameEpoch = 0; this.time = 0; }
  load(project, selected) {
    this.project = project; this.selected = selected; this.catalog = null;
    const epoch = ++this.loadEpoch;
    $('raw-summary').textContent = `${project.sources.length} camera / ${project.video_id}`;
    $('project-summary').textContent = `${project.clips.length} 保存clip / 秒で同期`;
    $('dataset-summary').textContent = '既存出力を照合中…';
    $('dataset-provenance').textContent = `${project.projects_path} · ${project.read_only ? '読取専用' : '編集モード'}`;
    $('source-paths').replaceChildren(...project.sources.map(s => node('p', `${s.camera_id}: ${s.source_path}`, 'source-path')));
    this.renderSelected(); this.decorateRows();
    get(`review?revision=${project.revision}`).then(catalog => {
      if (epoch !== this.loadEpoch) return;
      this.catalog = catalog;
      const registered = catalog.video_registered_total === null ? '登録未確認' : `${catalog.video_registered_total} 登録`;
      const pending = catalog.counts.unexported || 0;
      const issues = catalog.clips.filter(c => !['indexed','unexported'].includes(c.state)).length;
      $('dataset-summary').textContent = `${registered} / ${pending} 未出力${issues ? ` / ${issues} 要確認` : ''}`;
      $('dataset-provenance').textContent = `${project.projects_path} · SHA-256 ${catalog.projects_sha256 || '未保存'}\n${catalog.dataset_path} · dataset全体 ${catalog.dataset_registered_total ?? '不明'} 登録clip${catalog.index_error ? `\n登録確認エラー: ${catalog.index_error}` : ''}`;
      this.decorateRows(); this.renderSelected();
    }).catch(error => {
      if (epoch !== this.loadEpoch) return;
      $('dataset-summary').textContent = '出力状態を確認できません';
      this.onStatus(`dataset照合: ${error.message}`, true);
    });
  }
  select(name) { this.selected = name; this.renderSelected(); }
  decorateRows() {
    document.querySelectorAll('.clip-row').forEach(row => {
      row.querySelector('.clip-state')?.remove();
      const clip = this.catalog?.clips.find(c => c.name === row.dataset.clip);
      if (clip) row.append(node('span', labels[clip.state], `clip-state state-${clip.state}`));
    });
  }
  renderSelected() {
    const panel = $('clip-review'); panel.replaceChildren();
    const clip = this.project?.clips.find(c => c.name === this.selected);
    if (!clip) { panel.append(node('p','一覧からclipを選ぶと区間と出力状態を確認できます。','hint')); return; }
    panel.append(node('h3',`選択区間 · ${clip.name}`));
    panel.append(node('p',`[${clip.start_sec.toFixed(3)}, ${clip.end_sec.toFixed(3)}) 秒`, 'clip-bounds'));
    panel.append(node('p',`${(clip.end_sec-clip.start_sec).toFixed(3)}秒 · 終了時刻を含まない`, 'hint'));
    const inside = clip.start_sec <= this.time && this.time < clip.end_sec;
    panel.append(node('p',inside ? `現在位置は区間内 · clip内 ${(this.time-clip.start_sec).toFixed(3)}秒` : '現在位置は選択clipの区間外', inside ? 'membership inside' : 'membership outside'));
    const record = this.catalog?.clips.find(c => c.name === clip.name);
    if (!record) { panel.append(node('p','出力状態を照合中…','hint')); return; }
    panel.append(node('span',labels[record.state],`clip-state state-${record.state}`));
    panel.append(node('p',record.reason,'hint'));
    if (record.export) {
      const ex = record.export;
      panel.append(node('p',`${ex.width}×${ex.height} · ${ex.fps.toFixed(3)} fps\n${ex.num_frames} frames · 出力frame [0, ${ex.num_frames})`,'export-format'));
      const details = node('details',''); details.append(node('summary','切出時の元frame・変換'));
      ex.cameras.forEach(camera => {
        const fit = camera.letterbox ? `letterbox x=${camera.letterbox.pad_x}, y=${camera.letterbox.pad_y}px` : '解像度変換なし';
        details.append(node('p',`${camera.camera_id}: ${camera.source_frame_start}–${camera.source_frame_end}（末尾を含む） · ${fit}`,'hint'));
      });
      details.append(node('p',record.manifest_path,'source-path'));panel.append(details);
    }
  }
  updateMembership() {
    const clip = this.project.clips.find(c => c.name === this.selected);
    const text = $('clip-review').querySelector('.membership');
    if (!clip || !text) return;
    const inside = clip.start_sec <= this.time && this.time < clip.end_sec;
    text.textContent = inside ? `現在位置は区間内 · clip内 ${(this.time-clip.start_sec).toFixed(3)}秒` : '現在位置は選択clipの区間外';
    text.className = inside ? 'membership inside' : 'membership outside';
  }
  pendingRows(playing) {
    $('frame-map').replaceChildren(...this.project.sources.map(source => {
      const row = node('tr',''), identity = node('td','');
      identity.append(node('strong',source.camera_id),node('small',`${source.width}×${source.height} / ${source.fps.toFixed(3)} fps`));
      row.append(identity,node('td',`${source.offset_sec >= 0 ? '+' : ''}${source.offset_sec.toFixed(6)}`),node('td','—'),node('td','—'),node('td',playing ? '停止して照合' : '照合中…'));
      return row;
    }));
  }
  atTime(time, playing) {
    this.time = time;
    if (!this.project) return;
    this.updateMembership();
    if (playing && this.wasPlaying) { $('review-time').textContent = `${timecode(time)} · 再生中・停止して照合`; return; }
    this.wasPlaying = playing;
    clearTimeout(this.timer); this.controller?.abort();
    const epoch = ++this.frameEpoch;
    delete $('review-time').dataset.time;
    $('review-time').textContent = `${timecode(time)} · ${playing ? '再生中・停止して照合' : '照合中…'}`;
    this.pendingRows(playing);
    if (playing) { $('clip-membership').textContent = '複数動画の再生中は厳密な同時frameではありません。停止して同期を確認してください。'; return; }
    this.timer = setTimeout(async () => {
      this.controller = new AbortController();
      try {
        const result = await get(`correspondence?time=${time}&revision=${this.project.revision}`, this.controller.signal);
        if (epoch !== this.frameEpoch) return;
        $('review-time').textContent = `${timecode(time)} · 停止時の対応`;
        $('review-time').dataset.time = String(time);
        $('frame-map').replaceChildren();
        result.cameras.forEach((camera,index) => {
          const source = this.project.sources[index], row = node('tr',''); row.dataset.camera = camera.camera_id;
          row.className = camera.available ? '' : 'unavailable';
          const identity = node('td',''); identity.append(node('strong',camera.camera_id),node('small',`${source.width}×${source.height} / ${source.fps.toFixed(3)} fps`));
          row.append(identity,node('td',`${camera.offset_sec >= 0 ? '+' : ''}${camera.offset_sec.toFixed(6)}`),node('td',camera.local_time_sec.toFixed(3)),node('td',camera.frame_index === null ? '—' : String(camera.frame_index)),node('td',camera.available ? 'あり' : '範囲外・映像なし'));
          $('frame-map').append(row);
        });
        $('clip-membership').textContent = `元動画時刻 = 共通時刻 + 保存offset · 最近傍frame。現在位置を含む保存clip: ${result.containing_clips.join(', ') || 'なし（clip未選別の時間）'}`;
      } catch(error) {
        if (error.name !== 'AbortError' && epoch === this.frameEpoch) {
          $('review-time').textContent = '時刻の対応を確認できません';
          this.onStatus(`frame照合: ${error.message}`,true);
        }
      }
    }, 80);
  }
}
