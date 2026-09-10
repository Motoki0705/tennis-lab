// Native playback for smooth scanning; server-decoded frames for paused inspection.
export function clamp(value, start, end) { return Math.min(end, Math.max(start, value)); }
export function stepSeconds(value, fps, multiplier = 1) {
  return (value === 'frame' ? 1 / fps : Number(value)) * multiplier;
}
export function timecode(value) {
  const sign = value < 0 ? '-' : '';
  const ms = Math.round(Math.abs(value) * 1000);
  return sign + [Math.floor(ms / 3600000), Math.floor(ms / 60000) % 60,
    Math.floor(ms / 1000) % 60].map(n => String(n).padStart(2, '0')).join(':') + '.' + String(ms % 1000).padStart(3, '0');
}

export class Playback {
  constructor(container, onTime, onStatus, onSelect) {
    this.container = container;
    this.onTime = onTime;
    this.onStatus = onStatus;
    this.onSelect = onSelect;
    this.selected = 0;
    this.focus = true;
    this.playing = false;
    this.rate = 1;
    this.time = 0;
    this.loop = null;
    this.generation = 0;
    this.playEpoch = 0;
    this.previewPending = false;
    this.tiles = [];
  }
  load(project) {
    this.pause(false);
    this.project = project;
    ++this.generation;
    if (!this.tiles.length) {
      project.sources.forEach((source, index) => {
        const tile = document.createElement('article'); tile.className = 'viewer';
        const head = document.createElement('div'); head.className = 'viewer-head';
        const title = document.createElement('span'); title.textContent = source.camera_id;
        const button = document.createElement('button'); button.textContent = 'フォーカス';
        button.onclick = () => this.onSelect(index);
        head.append(title, button);
        const picture = document.createElement('div'); picture.className = 'picture';
        const video = document.createElement('video'); video.preload = 'metadata'; video.muted = true; video.playsInline = true;
        video.src = `/api/media/${index}`;
        const img = document.createElement('img'); img.alt = `${source.camera_id} の確認フレーム`;
        const label = document.createElement('span'); label.className = 'frame-label';
        picture.append(video, img, label); tile.append(head, picture); this.container.append(tile);
        video.addEventListener('error', () => {
          label.textContent = 'ブラウザがこの動画を再生できません。H.264 MP4を用意してください。停止フレームは確認できます。';
          label.classList.add('error');
          if (index === this.selected) { this.pause(false); this.onStatus(label.textContent, true); }
        });
        video.addEventListener('ended', () => {
          if (index === this.selected && this.playing) {
            if (this.loop) this.seekAndPlay(this.loop.start_sec);
            else { this.time = source.duration_sec - this.project.sources[index].offset_sec; this.pause(); }
          }
        });
        this.tiles.push({tile, video, img, label, url: null});
      });
    }
    this.time = clamp(this.time, ...project.extent);
    this.layout(); this.seek(this.time);
  }
  visible(index) { return !this.focus || index === this.selected; }
  layout() {
    this.container.classList.toggle('focus', this.focus);
    this.tiles.forEach(({tile}, index) => {
      tile.hidden = !this.visible(index); tile.classList.toggle('selected', index === this.selected);
    });
  }
  select(index) { this.pause(false); this.selected = index; ++this.generation; this.layout(); this.seek(this.time); }
  setFocus(focus) { this.pause(false); this.focus = focus; ++this.generation; this.layout(); this.seek(this.time); }
  seek(time) {
    this.pause(false);
    this.time = clamp(time, ...this.project.extent);
    this.onTime(this.time, false);
    this.tiles.forEach(({video, img, label}, index) => {
      video.hidden = true;
      if (this.visible(index)) { label.textContent = 'フレームを取得中…'; label.classList.remove('error'); }
    });
    this.schedulePreview();
  }
  schedulePreview() {
    if (this.previewPending || this.previewTimer || this.playing) return;
    this.previewTimer = setTimeout(() => { this.previewTimer = null; void this.preview(); }, 60);
  }
  async preview() {
    this.previewPending = true;
    const generation = this.generation, requestedTime = this.time, revision = this.project.revision;
    await Promise.all(this.tiles.map(async (tile, index) => {
      if (!this.visible(index)) return;
      try {
        const response = await fetch(`/api/frame/${index}?time=${requestedTime}&revision=${revision}`);
        if (generation !== this.generation || this.playing) return;
        if (response.status === 204) { tile.img.hidden = true; tile.label.textContent = 'この時刻の映像はありません'; return; }
        if (!response.ok) throw new Error((await response.json()).detail || 'プレビュー取得失敗');
        const blob = await response.blob();
        if (generation !== this.generation || this.playing) return;
        if (tile.url) URL.revokeObjectURL(tile.url);
        tile.url = URL.createObjectURL(blob); tile.img.src = tile.url;
        tile.img.hidden = false;
        tile.label.textContent = `frame ${response.headers.get('X-Frame-Index')} · ${timecode(requestedTime + this.project.sources[index].offset_sec)}`;
      } catch (error) {
        if (generation === this.generation && !this.playing) {
          tile.label.textContent = error.message; tile.label.classList.add('error'); this.onStatus(error.message, true);
        }
      }
    }));
    this.previewPending = false;
    if (requestedTime !== this.time || generation !== this.generation) this.schedulePreview();
  }
  pause(preview = true) {
    ++this.playEpoch;
    this.playing = false; cancelAnimationFrame(this.raf);
    this.tiles.forEach(({video}) => video.pause());
    if (this.project) {
      this.onTime(this.time, false);
      if (preview) this.seek(this.time);
    }
  }
  async play() {
    if (this.playing) { this.pause(); return; }
    const source = this.project.sources[this.selected];
    const local = this.time + source.offset_sec;
    if (local < 0 || local >= source.duration_sec) {
      this.onStatus('選択カメラに映像がある時刻へ移動してください。', true); return;
    }
    clearTimeout(this.previewTimer); this.previewTimer = null;
    ++this.generation;
    const epoch = ++this.playEpoch;
    const master = this.tiles[this.selected].video;
    try {
      master.currentTime = local; master.playbackRate = this.rate;
      await master.play();
      if (epoch !== this.playEpoch) return;
      this.playing = true;
      this.clock();
    } catch (error) {
      if (epoch !== this.playEpoch) return;
      this.pause(); this.onStatus(`再生できません: ${error.message}`, true);
    }
  }
  seekAndPlay(time) { this.seek(time); void this.play(); }
  clock() {
    if (!this.playing) return;
    const epoch = this.playEpoch;
    const source = this.project.sources[this.selected];
    const master = this.tiles[this.selected].video;
    this.time = master.currentTime - source.offset_sec;
    if (this.loop && this.time >= this.loop.end_sec) { this.seekAndPlay(this.loop.start_sec); return; }
    this.tiles.forEach(({video, img, label}, index) => {
      const camera = this.project.sources[index];
      const local = this.time + camera.offset_sec;
      if (!this.visible(index) || local < 0 || local >= camera.duration_sec) {
        video.pause(); video.hidden = true; img.hidden = true;
        label.textContent = 'この時刻の映像はありません'; return;
      }
      img.hidden = true; video.hidden = false;
      label.textContent = `${timecode(local)} · 再生中`;
      if (index !== this.selected) {
        if (Math.abs(video.currentTime - local) > 0.12 && !video.seeking) video.currentTime = local;
        video.playbackRate = this.rate;
        if (video.paused && !video.error) video.play().catch(error => {
          if (epoch !== this.playEpoch) return;
          this.pause(); this.onStatus(`比較カメラ ${camera.camera_id} を再生できません: ${error.message}`, true);
        });
      }
    });
    this.onTime(this.time, true);
    this.raf = requestAnimationFrame(() => this.clock());
  }
  setRate(rate) { this.rate = rate; this.tiles.forEach(({video}) => { video.playbackRate = rate; }); }
}
