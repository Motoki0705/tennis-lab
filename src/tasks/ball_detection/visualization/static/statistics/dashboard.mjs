// Adapt existing, exact statistics to charts. No re-estimation of distributions.
import {node, number} from './charts.mjs';
import {PALETTE, selectControl, svgNode} from './plot_base.mjs';
import {composition, ranges, spatialMap, strideComparison, clipScatter, KINDS} from './plots.mjs';
const SCOPES={clip:'clip全体',play:'プレイ候補',selected:'採用範囲',excluded:'非採用範囲'};
const SOURCES={tracknet:'TrackNet',meiji:'Meiji',chat_annotation:'Chat'};
const JOINTS={nose:'鼻',left_eye:'左目',right_eye:'右目',left_ear:'左耳',right_ear:'右耳',left_shoulder:'左肩',right_shoulder:'右肩',left_elbow:'左肘',right_elbow:'右肘',left_wrist:'左手首',right_wrist:'右手首',left_hip:'左腰',right_hip:'右腰',left_knee:'左膝',right_knee:'右膝',left_ankle:'左足首',right_ankle:'右足首'};
export function groupLabel(key){const parts=key.split('/');if(key==='all')return '全source';if(parts[0]==='split')return `${parts[1]} split`;return [SOURCES[parts[1]]??parts[1],parts[2]].filter(Boolean).join(' / ');}
export function sourceGroups(report,key) {
  if(key==='all')return Object.keys(report.groups).filter(k=>k.startsWith('source/')).map(k=>[k,groupLabel(k)]);
  if(key.startsWith('split/'))return Object.keys(report.groups).filter(k=>k.startsWith('source_split/')&&k.endsWith(`/${key.split('/')[1]}`)).map(k=>[k,SOURCES[k.split('/')[1]]??k]);
  return [[key,groupLabel(key)]];
}
export function chartClips(report,key,scope){
  return report.clips.filter(c=>key==='all'||key===`source/${c.source}`||key===`split/${c.split}`||key===`source_split/${c.source}/${c.split}`).map(c=>({...c,...c.scopes[scope]}));
}
export function spatialData(metrics,mode){
  const views=metrics.views,counts=views['motion/observed/edge_count'];
  if(mode==='occupancy'){
    const matrix=views['spatial/observed/occupancy'],total=matrix.flat().reduce((a,b)=>a+b,0);
    return {matrix:matrix.map(row=>row.map(v=>total?v/total*100:null)),unit:'%',title:'球が現れる場所',
      note:`グリッド内の実測点に占める割合（時間占有率ではありません）。点数 ${number(total)}、グリッド外 ${number(metrics.counts['spatial/observed/outside_endpoint_grid'])}点。画像左上が(0,0)。`};
  }
  if(mode==='speed')return {matrix:views['motion/observed/speed_sum'].map((row,y)=>row.map((v,x)=>counts[y][x]>0?v/counts[y][x]:null)),counts,unit:'正規化/秒',title:'場所ごとの平均速度',
    note:'実測球の隣接フレーム間の速度を、出発位置のセルで平均。斜線セルは有効な移動ペアなし。座標は幅・高さを個別に0–1へ正規化。'};
  return {matrix:views['motion/observed/movement'].map((row,y)=>row.map((v,x)=>counts[y][x]>0?v:null)),counts,dx:views['motion/observed/direction_x_sum'],dy:views['motion/observed/direction_y_sum'],unit:'正規化座標',title:'移動量と平均変位の方向',
    note:'色＝出発セル別の累積移動量。矢印＝平均変位（速度ではありません）。矢印の長さは図内の最大に合わせて表示し、逆向きの移動は相殺されます。カメラ位置の確定推定ではありません。'};
}

export function dashboard(report,key,scope,onClip,state={}) {
  const root=node('div','','statistics-chart-grid'),group=report.groups[key],context=`${groupLabel(key)} · ${SCOPES[scope]}`;
  const identify=figure=>{
    figure.querySelector('svg').append(svgNode('metadata',{},JSON.stringify({schema:report.schema,identity:report.identity,config:report.config,selection:report.selection,group:key,scope,view_state:{...state}})));
    return figure;
  };
  function card(defaults,draw){
    for(const [field,value]of Object.entries(defaults))if(!(field in state))state[field]=value;
    const host=node('div','','statistics-chart-slot');root.append(host);
    const choose=(caption,field,entries)=>selectControl(caption,entries,state[field],value=>{state[field]=value;render();});
    const render=()=>{const figure=identify(draw(choose));host.classList.toggle('statistics-chart-slot-wide',figure.dataset.chart==='stride');host.replaceChildren(figure);};render();
  }
  card({compositionAxis:'source'},choose=>{
    const rows=state.compositionAxis==='source'
      ?sourceGroups(report,key).map(([id,label])=>({label,metrics:report.groups[id][scope].pooled}))
      :Object.entries(SCOPES).map(([id,label])=>({label,metrics:group[id].pooled}));
    return composition(rows.map(r=>({label:r.label,total:r.metrics.counts.instances,counts:Object.fromEntries(KINDS.map(k=>[k.key,r.metrics.counts[`instances/${k.key}`]]))})),{
      context:state.compositionAxis==='source'?context:`${groupLabel(key)} · 対象範囲の比較`,
      controls:[choose('比較軸','compositionAxis',[['source','source'],['scope','対象範囲']])]});
  });
  card({spatialMode:'occupancy'},choose=>{
    const data=spatialData(group[scope].pooled,state.spatialMode);
    return spatialMap(data.matrix,{...data,context,key:'spatial',controls:[choose('位置と動き','spatialMode',[['occupancy','位置の分布'],['speed','平均速度'],['movement','移動量・方向']])]});
  });
  card({distributionMetric:'speed',distributionBasis:'pooled'},choose=>{
    const metrics={speed:['motion/observed/speed_normalized','正規化座標/秒','実測球の速度'],gap:['gaps/coordinate_gap/bounded/seconds','秒','座標欠損の連続長'],interpolation:['interpolation/run_seconds','秒','補間区間の長さ']};
    const [metric,unit,title]=metrics[state.distributionMetric];
    const isClip=state.distributionBasis==='clip';
    const rows=Object.entries(SCOPES).map(([id,label])=>({label,color:id===scope?PALETTE.blue:'#9dabbc',summary:isClip?group[id].between_clips[`median/${metric}`]:group[id].pooled.distributions[metric]}));
    return ranges(rows,{title:`${title}のばらつき`,context:`${groupLabel(key)} · 対象範囲別 · ${isClip?'clip中央値の分布':'全サンプルの分布'}`,unit,key:'distribution',
      note:isClip?'各clipの中央値を1点として集計。有効数はclip数です。':'有効数は速度なら移動ペア数、欠損・補間なら区間数です。',
      controls:[choose('分布の指標','distributionMetric',[['speed','実測球速度'],['gap','座標欠損長'],['interpolation','補間長']]),choose('分布の対象','distributionBasis',[['pooled','全サンプル'],['clip','clip中央値']])]});
  });
  card({poseMetric:'relative_speed'},choose=>{
    const metrics={relative_speed:['人物移動を除いた関節速度','人物身長/秒'],speed:['関節の移動速度','人物身長/秒'],spike:['前後から突出する関節位置','人物身長比']};
    const [title,unit]=metrics[state.poseMetric];
    const threshold=state.poseMetric==='spike'?report.config.pose_spike_threshold:report.config.pose_jump_primary;
    const rows=Object.entries(JOINTS).map(([joint,label])=>({label,summary:group[scope].pooled.distributions[`pose/${joint}/${state.poseMetric}`]}));
    return ranges(rows,{title,context,unit,key:'pose',threshold,color:PALETTE.teal,
      note:'全人物・時刻の有効サンプルを集約。閾値超過は確認候補であり、誤推定の確定や自動除外ではありません。',
      controls:[choose('poseの指標','poseMetric',[['relative_speed','人物移動を除いた速度'],['speed','移動速度'],['spike','一瞬の突出']])]});
  });
  card({coverageBasis:'play',coverageZoom:'full'},choose=>{
    const m=group.windows.pooled;
    const rows=report.config.strides.map(stride=>{const prefix=`windows/stride_${stride}`;return {stride,count:m.counts[prefix+'/count'],unique:m.counts[prefix+'/unique_frames'],occurrences:m.counts[prefix+'/frame_occurrences'],rate:m.rates[prefix+(state.coverageBasis==='play'?'/play_coverage':'/coverage')]};});
    return strideComparison(rows,{context:`${groupLabel(key)} · 全プレイ候補からの窓（対象範囲フィルタとは独立）`,selected:report.config.scope_stride,coverageBasis:state.coverageBasis,zoom:state.coverageZoom==='zoom',
      controls:[choose('被覆率の分母','coverageBasis',[['play','プレイ候補'],['clip','clip全体']]),choose('被覆率の縦軸','coverageZoom',[['full','0–100%'],['zoom','差を拡大']])]});
  });
  const scatter=identify(clipScatter(chartClips(report,key,scope),{context,onClip}));
  const scatterHost=node('div','','statistics-chart-slot statistics-chart-slot-wide');scatterHost.append(scatter);root.append(scatterHost);
  return root;
}
