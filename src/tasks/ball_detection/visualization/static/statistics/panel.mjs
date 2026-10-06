import {node, number, heatmap, trajectory, table, label, percent} from './charts.mjs';
const SCOPES = {clip: 'clip全体', play: 'プレイ候補', selected: '採用範囲', excluded: '非採用範囲'};
async function api(url, body) {
  const response = await fetch(url, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function button(text, action) {const b=node('button',text);b.type='button';b.onclick=action;return b;}
function options(select, entries) {select.replaceChildren(...entries.map(([key,label])=>new Option(label,key)));}
export class StatisticsPanel {
  constructor(dialog, navigate) {
    this.dialog=dialog; this.navigate=navigate; this.token=0; this.job=null; this.report=null; this.selectedClip=null;
    const css=document.createElement('link');css.rel='stylesheet';css.href='/statistics-static/style.css';document.head.append(css);
    dialog.classList.add('statistics-dialog');
    dialog.innerHTML=`<header><strong>ボールデータセット統計</strong><button data-id="close">閉じる</button></header>
      <p data-id="dataset"></p><form data-id="form" class="statistics-settings">
      <label>比較stride<input data-id="strides" required></label><label>採用範囲stride<input data-id="scope_stride" type="number" min="1" max="32" required></label>
      <label>位置分布の横分割<input data-id="grid_x" type="number" min="1" max="16" required></label><label>縦分割<input data-id="grid_y" type="number" min="1" max="16" required></label>
      <label>pose速度の比較閾値（身長/秒）<input data-id="pose_jump_speeds" required></label><label>主閾値（身長/秒）<input data-id="pose_jump_primary" type="number" min="0.01" step="any" required></label>
      <label>関節の最小スコア<input data-id="pose_min_score" type="number" step="any" required></label>
      <label>一瞬の突出（身長比）<input data-id="pose_spike_threshold" type="number" step="any" min="0.001" required></label>
      <label>骨格長の急変（身長比）<input data-id="pose_bone_change_threshold" type="number" step="any" min="0.001" required></label>
      <button data-id="run" type="submit">統計を計算</button></form>
      <p data-id="status" role="status"></p><div data-id="results" hidden>
      <div class="statistics-filters"><label>集計グループ<select data-id="group"></select></label><label>対象範囲<select data-id="scope"></select></label>
      <label>clip並べ替え<select data-id="sort"><option value="speed_p95">速度P95</option><option value="gap_p95">欠損長P95</option><option value="pose_flags">pose飛び候補数</option><option value="interpolated">補間率</option></select></label>
      <button data-id="download">集計JSONを保存</button></div>
      <p data-id="summary"></p><div data-id="charts" class="statistics-charts"></div>
      <details open><summary>注釈構成・各種割合</summary><div data-id="rates"></div></details>
      <details open><summary>32frame窓とstride比較</summary><div data-id="windows"></div></details>
      <details><summary>分布統計（全体分布／clip間の分布）</summary>
      <div class="statistics-filters"><label>集約単位<select data-id="aggregation"><option value="pooled">全サンプルを集約</option><option value="between_clips">clipごとの指標の分布</option></select></label>
      <label>指標の絞り込み<input data-id="metric" placeholder="pose / speed / gaps / median…"></label></div><div data-id="metrics"></div></details>
      <details open><summary>clip一覧（選択すると詳細と原文を表示）</summary><div data-id="clips"></div></details>
      <section data-id="detail"></section></div>`;
    this.$=id=>dialog.querySelector(`[data-id="${id}"]`);
    this.$('close').onclick=()=>dialog.close();
    this.$('form').onsubmit=event=>{event.preventDefault();this.run();};
    for(const id of ['group','scope','sort'])this.$(id).onchange=()=>{this.selectedClip=null;this.render();};
    for(const id of ['aggregation','metric'])this.$(id).oninput=()=>this.renderMetrics();
    this.$('download').onclick=()=>{
      const url=URL.createObjectURL(new Blob([JSON.stringify(this.report,null,2)],{type:'application/json'}));
      const a=node('a');a.href=url;a.download='ball-dataset-statistics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    };
    options(this.$('scope'),Object.entries(SCOPES));
  }
  async open(dataset) {
    if(!dataset)throw new Error('データセットを選択してください。');
    if(this.dataset!==dataset){this.token++;this.dataset=dataset;this.report=null;this.selectedClip=null;this.$('results').hidden=true;this.$('status').textContent='';this.$('run').disabled=false;}
    this.$('dataset').textContent=dataset;
    if(!this.settings){this.settings=await api('/api/statistics/config');for(const [key,value]of Object.entries(this.settings))this.$(key).value=Array.isArray(value)?value.join(','):String(value);}
    if(!this.dialog.open)this.dialog.showModal();
  }
  async run() {
    const token=++this.token,dataset=this.dataset;
    this.$('run').disabled=true;this.$('status').textContent='計算を開始しています…';this.$('results').hidden=true;this.report=null;this.selectedClip=null;
    try{
      const settings=Object.fromEntries(Object.keys(this.settings).map(key=>[key,['strides','pose_jump_speeds'].includes(key)?this.$(key).value.split(',').map(v=>Number(v.trim())):Number(this.$(key).value)]));
      let job=await api('/api/statistics/jobs',{dataset,settings});if(token!==this.token)return;const jobId=job.id;this.job=jobId;
      while(['pending','running'].includes(job.state)){
        if(token!==this.token)return;
        this.$('status').textContent=`計算中 ${job.completed} / ${job.total} clips（全clip後に全体集約）`;
        await new Promise(resolve=>setTimeout(resolve,600));job=await api(`/api/statistics/jobs/${jobId}`);
      }
      if(token!==this.token)return;
      if(job.state!=='complete')throw new Error(job.error);
      const report=await api(`/api/statistics/jobs/${jobId}/result`);
      if(token!==this.token)return;
      this.report=report;options(this.$('group'),Object.keys(report.groups).map(key=>[key,key]));
      this.$('status').textContent='計算完了。設定を変えた場合は再計算してください。';this.$('results').hidden=false;this.render();
    }catch(error){if(token===this.token)this.$('status').textContent=`計算できません: ${error.message}`;}
    finally{if(token===this.token)this.$('run').disabled=false;}
  }
  render(){
    if(!this.report)return;
    const scope=this.$('scope').value, group=this.report.groups[this.$('group').value], metrics=group[scope].pooled;
    this.$('summary').textContent=`${group[scope].clips} clips · ${number(metrics.counts.frames)} frames · 延べ ${number(metrics.views['annotation/seconds_total'])}秒 · 元動画 ${this.report.independent_videos}件（データセット全体） · 採用範囲stride ${this.report.config.scope_stride} · pose利用可 ${this.report.pose_available_clips}/${this.report.clip_count} clips。原本の利用状態: ${JSON.stringify(this.report.raw_annotation_availability)}。シーン切替の確定判定・自動除外は行いません。`;
    const views=metrics.views;
    this.$('charts').replaceChildren(heatmap(views['spatial/observed/occupancy'],'実測位置の分布（件数、左上が画像左上）'),heatmap(views['motion/observed/movement'],'移動量の集中（正規化座標）'));
    const transitions=views['motion/observed/transitions'];
    if(transitions){const rows=[...transitions];rows.sort((a,b)=>b[2]-a[2]);
      const cell=index=>`(${index%this.report.config.grid_x}, ${Math.floor(index/this.report.config.grid_x)})`;
      const section=node('details');section.append(node('summary','領域間遷移（多い順、上位20）'),table(['出発領域(x,y)','到着領域(x,y)','回数'],rows.slice(0,20).map(([a,b,n])=>[cell(a),cell(b),number(n)])));this.$('charts').append(section);}
    this.$('rates').replaceChildren(table(['指標','該当数','分母','全体割合','clip平均','clip中央値','clip P5','clip P95'],Object.entries(metrics.rates).map(([key,r])=>{const between=group[scope].between_clips[`rates/${key}`];return[label(key),number(r.numerator),number(r.denominator),percent(r.value),percent(between.mean),percent(between.median),percent(between.p5),percent(between.p95)];})));
    this.renderWindows(group.windows.pooled);this.renderMetrics();this.renderClips();this.$('detail').replaceChildren();
  }
  renderWindows(m){this.$('windows').replaceChildren(table(['stride','窓数','一意frame','延べframe','被覆率','実測数 平均/窓','欠損最大長 P95','pose飛びframe 平均/窓'],this.report.config.strides.map(s=>{const p=`windows/stride_${s}`;return[s,number(m.counts[p+'/count']),number(m.counts[p+'/unique_frames']),number(m.counts[p+'/frame_occurrences']),percent(m.rates[p+'/coverage'].value),number(m.distributions[p+'/observed'].mean),number(m.distributions[p+'/coordinate_gap_max'].p95),number(m.distributions[p+'/pose_flagged'].mean)];})));
    const prefix=`windows/stride_${this.report.config.scope_stride}`;
    this.$('windows').append(heatmap([m.views[prefix+'/position_supervision'],m.views[prefix+'/position_missing']],'窓内位置0–31：上＝実測教師数、下＝座標欠損数。MDDの位置0は常にゼロ。'));
  }
  renderMetrics(){if(!this.report)return;const group=this.report.groups[this.$('group').value][this.$('scope').value];
    const entries=this.$('aggregation').value==='pooled'?group.pooled.distributions:group.between_clips;
    const search=this.$('metric').value.toLowerCase();
    this.$('metrics').replaceChildren(table(['指標','有効数','対象数','平均','中央値','P5','P95','単位'],Object.entries(entries).filter(([k])=>`${k} ${label(k)}`.toLowerCase().includes(search)).map(([key,r])=>[label(key),number(r.n),number(r.eligible),number(r.mean),number(r.median),number(r.p5),number(r.p95),r.unit])));
  }
  renderClips(){
    const group=this.$('group').value,scope=this.$('scope').value,key=this.$('sort').value;
    let clips=this.report.clips.filter(c=>group==='all'||group===`source/${c.source}`||group===`split/${c.split}`||group===`source_split/${c.source}/${c.split}`);
    clips=clips.sort((a,b)=>(b.scopes[scope][key]??-Infinity)-(a.scopes[scope][key]??-Infinity));
    this.$('clips').replaceChildren(table(['clip','source/split','frame','実測率','補間率','欠損長P95','速度P95(px/s)','pose候補数'],clips.map(c=>{const m=c.scopes[scope];return[button(c.clip_id,()=>this.showClip(c.clip_id)),`${c.source}/${c.split}`,number(m.frames),percent(m.observed),percent(m.interpolated),number(m.gap_p95),number(m.speed_p95),number(m.pose_flags)];})));
  }
  async showClip(id){
    const token=this.token;this.selectedClip=id;this.$('detail').textContent='clip詳細を読み込み中…';
    try{const report=await api(`/api/statistics/jobs/${this.job}/clip?${new URLSearchParams({scene:`${this.dataset}::${id}`})}`);
      if(token!==this.token||id!==this.selectedClip)return;
      const scope=this.$('scope').value,m=report.scopes[scope];
      const navigate=frame=>{this.dialog.close();this.navigate({...report.clip,dataset:this.dataset},frame);};
      const points=report.trajectory.filter(([i])=>report.intervals[scope].some(([a,b])=>a<=i&&i<b));
      const content=[node('h3',id),button('画像で確認',()=>navigate(report.intervals[scope][0]?.[0]??0)),trajectory(points,report.boundary_frames,navigate)];
      content.push(node('p',`原本: ${report.original_annotation.availability} ${report.original_annotation.reason??''} / pose: ${report.pose_availability}`));
      const issues=node('ul');report.original_annotation.issues.forEach(text=>issues.append(node('li',text)));content.push(issues);
      const sourceNotes=node('ul');report.original_annotation.source_notes.forEach(text=>sourceNotes.append(node('li',text)));content.push(sourceNotes);
      if(!report.original_annotation.notes_supported)content.push(node('p','frame単位のnotesは、この原本形式では未提供または未取得です。'));
      content.push(table(['frame','注釈メモ'],Object.entries(report.original_annotation.notes).map(([frame,text])=>[button(frame,()=>navigate(Number(frame))),text])));
      content.push(table(['frame','確認候補','人物・関節','値'],m.findings.map(f=>[button(String(f.frame),()=>navigate(f.frame)),f.kind,[f.player,f.joint].filter(Boolean).join(' / '),number(f.value)])));
      content.push(table(['指標','平均','中央値','P5','P95','有効数','単位'],Object.entries(m.distributions).map(([key,r])=>[label(key),number(r.mean),number(r.median),number(r.p5),number(r.p95),number(r.n),r.unit])));
      this.$('detail').replaceChildren(...content);this.$('detail').scrollIntoView({block:'start',behavior:'smooth'});
    }catch(error){if(token===this.token&&id===this.selectedClip)this.$('detail').textContent=error.message;}
  }
}
