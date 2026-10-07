import {number, percent} from './charts.mjs';
import {PALETTE, finite, svgNode, text, compact, tick, niceMax, scale, axes, legend, frame, empty, summaryLines} from './plot_base.mjs';

export const KINDS=[
  {key:'observed',label:'実測',color:PALETTE.blue},
  {key:'interpolated',label:'補間',color:'#59a7b2'},
  {key:'occlusion_estimated',label:'遮蔽推定',color:PALETTE.amber},
  {key:'unresolved',label:'位置未確定',color:PALETTE.purple},
  {key:'out_of_frame',label:'画面外',color:'#98a4b3'},
];
export function composition(rows,options={}) {
  const plot=frame({title:'球注釈の構成',subtitle:options.context,key:'composition',height:180+Math.max(rows.length,1)*54,
    note:'分母は保存された球注釈の個数。球注釈なしのframeは含みません。確認済み率は別指標で、各sourceの注釈基準も異なります。',controls:options.controls});
  const {svg,interact}=plot;legend(svg,KINDS,28,78);
  const left=132,right=672,width=right-left,y0=118;
  [0,25,50,75,100].forEach(v=>{text(svg,left+width*v/100,y0-12,`${v}%`,{'text-anchor':'middle','font-size':10,fill:PALETTE.muted});});
  rows.forEach((row,index)=>{
    const y=y0+index*54;
    text(svg,left-14,y+18,row.label,{'text-anchor':'end','font-size':12});
    const total=row.total;
    if(!(total>0)){
      svg.append(svgNode('rect',{x:left,y,width,height:28,rx:3,fill:'#f0f3f7'}));
      text(svg,left+width/2,y+18,'球注釈なし',{'text-anchor':'middle','font-size':11,fill:PALETTE.muted});
    }else{
      let offset=0;
      KINDS.forEach(kind=>{
        const count=row.counts[kind.key],fraction=count/total,w=fraction*width;
        const rectangle=svgNode('rect',{x:left+offset,y,width:w,height:28,fill:kind.color,'data-kind':kind.key,'data-value':count,'data-denominator':total});
        svg.append(rectangle);
        if(fraction>=.085)text(svg,left+offset+w/2,y+18,`${(fraction*100).toFixed(1)}%`,{'text-anchor':'middle','font-size':11,fill:'#fff','font-weight':600});
        offset+=w;
      });
      const hit=svgNode('rect',{x:left,y,width,height:28,fill:'transparent','data-series':row.label});
      svg.append(interact(hit,[row.label,`球注釈 ${number(total)}個`,...KINDS.map(k=>`${k.label}: ${number(row.counts[k.key])} (${percent(row.counts[k.key]/total)})`)]));
    }
    text(svg,728,y+18,`n=${compact(total)}`,{'text-anchor':'end','font-size':11,fill:PALETTE.muted});
  });
  return plot.figure;
}

export function ranges(rows,{title,context,unit,key,controls=[],note='',threshold=null,color=PALETTE.blue}={}) {
  const valid=rows.filter(r=>r.summary?.n>0&&[r.summary.p5,r.summary.median,r.summary.p95].every(finite));
  const rowHeight=34,height=valid.length?166+Math.max(rows.length,1)*rowHeight:245;
  const plot=frame({title,subtitle:context,key,height,controls,note:`線はP5–P95、丸は中央値、菱形は平均。信頼区間や四分位範囲ではありません。${note}`});
  const {svg,interact}=plot;
  if(!valid.length){empty(svg,{x:28,y:85,w:704,h:123},'選択した範囲に有効な分布がありません');return plot.figure;}
  const upper=niceMax(Math.max(0,...valid.flatMap(r=>[r.summary.p95,finite(r.summary.mean)?r.summary.mean:0]))*1.06);
  const box={x:160,y:96,w:455,h:Math.max(rows.length,1)*rowHeight};
  const {x}=axes(svg,box,{xDomain:[0,upper],xLabel:unit,horizontal:false});
  text(svg,727,79,'中央値 / 有効数',{'text-anchor':'end','font-size':10,fill:PALETTE.muted});
  if(finite(threshold)&&threshold<=upper){
    svg.append(svgNode('line',{x1:x(threshold),x2:x(threshold),y1:box.y-5,y2:box.y+box.h,stroke:PALETTE.amber,'stroke-dasharray':'5 4','data-threshold':threshold}));
    text(svg,x(threshold)-4,83,`閾値 ${tick(threshold)}`,{'text-anchor':'end',fill:PALETTE.amber,'font-size':10});
  }else if(finite(threshold))text(svg,box.x,80,`候補閾値 ${tick(threshold)} は表示範囲外`,{fill:PALETTE.amber,'font-size':10});
  rows.forEach((row,index)=>{
    const y=box.y+(index+.5)*rowHeight,s=row.summary,c=row.color??color;
    text(svg,box.x-14,y+4,row.label,{'text-anchor':'end','font-size':12});
    if(!s?.n||![s.p5,s.median,s.p95].every(finite)){
      text(svg,box.x+12,y+4,'有効データなし',{'font-size':11,fill:PALETTE.muted});
      text(svg,727,y+4,'—',{'text-anchor':'end',fill:PALETTE.muted});return;
    }
    svg.append(svgNode('line',{x1:x(s.p5),x2:x(s.p95),y1:y,y2:y,stroke:c,'stroke-width':8,'stroke-opacity':.22,'stroke-linecap':'round','data-p5':s.p5,'data-p95':s.p95}));
    [s.p5,s.p95].forEach(v=>svg.append(svgNode('line',{x1:x(v),x2:x(v),y1:y-5,y2:y+5,stroke:c,'stroke-width':1.5})));
    svg.append(svgNode('circle',{cx:x(s.median),cy:y,r:4.5,fill:c,'data-median':s.median}));
    if(finite(s.mean))svg.append(svgNode('path',{d:`M${x(s.mean)},${y-4}l4,4 -4,4 -4,-4Z`,fill:'#fff',stroke:c,'stroke-width':1.5,'data-mean':s.mean}));
    text(svg,727,y+4,`${tick(s.median)} / ${compact(s.n)}`,{'text-anchor':'end','font-size':11,fill:PALETTE.muted});
    const hit=svgNode('rect',{x:box.x,y:y-13,width:box.w,height:26,fill:'transparent','data-series':row.label});
    svg.append(interact(hit,[...summaryLines(row.label,s),`単位: ${unit}`]));
  });
  text(svg,28,height-7,'P5–P95  ━  ·  中央値 ●  ·  平均 ◇',{'font-size':10,fill:PALETTE.muted});
  return plot.figure;
}

const COLORS=['#f1f6fa','#c7e5e3','#7cc9c0','#319c98','#116873','#173b58'];
function sequential(value,max){
  if(!finite(value))return null;
  const t=Math.max(0,Math.min(1,max>0?value/max:0))*(COLORS.length-1),lo=Math.floor(t),hi=Math.ceil(t);
  const rgb=hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
  const a=rgb(COLORS[lo]),b=rgb(COLORS[hi]);
  return `rgb(${a.map((c,i)=>Math.round(c+(b[i]-c)*(t-lo))).join(',')})`;
}
export function spatialMap(matrix,{title,context,unit,key='spatial',controls=[],note='',counts=null,dx=null,dy=null}={}) {
  const plot=frame({title,subtitle:context,key,height:444,controls,note});
  const {svg,interact,id}=plot, box={x:84,y:99,w:548,h:265};
  const rows=matrix?.length??0,cols=matrix?.[0]?.length??0;
  if(!rows||!cols){empty(svg,box);return plot.figure;}
  const known=matrix.flat().filter(finite),maximum=Math.max(0,...known),cw=box.w/cols,ch=box.h/rows;
  const defs=svgNode('defs');
  const gradient=svgNode('linearGradient',{id:`${id}-colors`,x1:'0%',x2:'0%',y1:'100%',y2:'0%'});
  COLORS.forEach((color,i)=>gradient.append(svgNode('stop',{offset:`${i*100/(COLORS.length-1)}%`,'stop-color':color})));
  const pattern=svgNode('pattern',{id:`${id}-empty`,width:6,height:6,patternUnits:'userSpaceOnUse'});
  pattern.append(svgNode('rect',{width:6,height:6,fill:'#f6f8fa'}),svgNode('path',{d:'M0 6L6 0',stroke:'#d5dee7','stroke-width':1}));
  defs.append(gradient,pattern);svg.append(defs);
  text(svg,box.x,80,'画像上端  y=0',{'font-size':11,fill:PALETTE.muted});
  for(let i=0;i<=4;i++){
    const f=i/4;text(svg,box.x+box.w*f,box.y+box.h+22,tick(f),{'text-anchor':'middle','font-size':11,fill:PALETTE.muted});
    text(svg,box.x-12,box.y+box.h*f+4,tick(f),{'text-anchor':'end','font-size':11,fill:PALETTE.muted});
  }
  let maxVector=0;
  if(dx&&dy)for(let y=0;y<rows;y++)for(let x=0;x<cols;x++)if(counts?.[y]?.[x]>0)maxVector=Math.max(maxVector,Math.hypot(dx[y][x]/counts[y][x]*box.w,dy[y][x]/counts[y][x]*box.h));
  matrix.forEach((row,y)=>row.forEach((value,x)=>{
    const rectangle=svgNode('rect',{x:box.x+x*cw,y:box.y+y*ch,width:cw,height:ch,fill:finite(value)?sequential(value,maximum):`url(#${id}-empty)`,stroke:'#ffffff','stroke-width':1,'data-cell':`${x},${y}`,'data-value':finite(value)?value:'missing'});
    svg.append(rectangle);
    if(dx&&dy&&counts[y][x]>0&&maxVector>0){
      const u=dx[y][x]/counts[y][x]*box.w,v=dy[y][x]/counts[y][x]*box.h,k=Math.min(cw,ch)*.65/maxVector;
      const cx=box.x+(x+.5)*cw,cy=box.y+(y+.5)*ch;
      const length=Math.hypot(u,v)*k;
      if(length>0){
        const vx=u*k,vy=v*k,ux=vx/length,uy=vy/length,head=Math.min(7,length*.45);
        const tipX=cx+vx/2,tipY=cy+vy/2,bx=tipX-ux*head,by=tipY-uy*head;
        const color=maximum>0&&value/maximum>.45?'#ffffff':'#243d50';
        const arrow=svgNode('g',{'pointer-events':'none','data-vector':`${x},${y}`});
        arrow.append(svgNode('line',{x1:cx-vx/2,y1:cy-vy/2,x2:bx,y2:by,stroke:color,'stroke-width':Math.min(1.6,Math.max(.6,length/6))}),
          svgNode('polygon',{points:`${tipX},${tipY} ${bx-uy*head*.48},${by+ux*head*.48} ${bx+uy*head*.48},${by-ux*head*.48}`,fill:color}));
        svg.append(arrow);
      }
    }
    const lines=[`領域 (${x}, ${y})`, `x: ${tick(x/cols)}–${tick((x+1)/cols)} · y: ${tick(y/rows)}–${tick((y+1)/rows)}`,finite(value)?`${number(value)} ${unit}`:'有効な観測なし'];
    if(counts)lines.push(`有効な移動ペア: ${number(counts[y][x])}`);
    if(dx&&dy&&counts[y][x]>0)lines.push(`平均変位: (${number(dx[y][x]/counts[y][x])}, ${number(dy[y][x]/counts[y][x])})`);
    interact(rectangle,lines);
  }));
  svg.append(svgNode('rect',{x:box.x,y:box.y,width:box.w,height:box.h,fill:'none',stroke:'#a8b7c5'}));
  text(svg,box.x+box.w/2,box.y+box.h+45,'x / 画像幅（左 → 右）',{'text-anchor':'middle','font-size':11,fill:PALETTE.muted});
  svg.append(svgNode('rect',{x:661,y:box.y,width:14,height:box.h,fill:!known.length?`url(#${id}-empty)`:maximum>0?`url(#${id}-colors)`:COLORS[0],rx:3}));
  if(maximum>0)[0,.5,1].forEach(f=>text(svg,684,box.y+box.h*(1-f)+4,tick(maximum*f),{'font-size':11,fill:PALETTE.muted}));
  else text(svg,684,box.y+box.h/2+4,known.length?'0':'—',{'font-size':11,fill:PALETTE.muted});
  text(svg,655,80,unit,{'font-size':10,fill:PALETTE.muted});
  if(!known.length)text(svg,box.x+box.w/2,box.y+box.h/2,'有効な観測なし',{'text-anchor':'middle','font-size':14,fill:PALETTE.muted});
  return plot.figure;
}

export function strideComparison(rows,{context,selected,coverageBasis='play',zoom=false,controls=[]}={}) {
  const plot=frame({title:'strideとサンプル量・被覆率',subtitle:context,key:'stride',width:940,height:409,controls,
    note:`32frame窓。左は窓数、右は一意フレームの被覆率。横軸は比較したstrideを等間隔に並べています。被覆率の分母: ${coverageBasis==='play'?'プレイ候補内のフレーム':'clip全体のフレーム'}。${zoom?'右の縦軸は差を見るため拡大しています。':'右の縦軸は0–100%です。'}`});
  const {svg,interact}=plot,left={x:84,y:102,w:330,h:216},right={x:548,y:102,w:325,h:216};
  const max=Math.max(1,Math.ceil(niceMax(Math.max(0,...rows.map(r=>r.count))*1.05)/4))*4;
  const rates=rows.map(r=>r.rate?.value).filter(finite).map(v=>100*v);
  const lo=zoom&&rates.length?Math.max(0,Math.floor((Math.min(...rates)-2)/5)*5):0;
  const hi=zoom&&rates.length?Math.min(100,Math.ceil((Math.max(...rates)+2)/5)*5):100;
  const a=axes(svg,left,{vertical:false,yDomain:[0,max],yLabel:'窓数',yFormat:compact});
  const b=axes(svg,right,{vertical:false,yDomain:[lo,Math.max(lo+1,hi)],yLabel:zoom?`被覆率（${lo}–${Math.max(lo+1,hi)}%）`:'被覆率（%）',yFormat:v=>`${tick(v)}%`});
  const x=(box,i)=>box.x+box.w*(i+.5)/Math.max(rows.length,1);
  let prior=null;
  rows.forEach((r,i)=>{
    const chosen=r.stride===selected,c=chosen?PALETTE.teal:'#9cb7cd',width=Math.min(43,left.w/Math.max(rows.length,1)*.62);
    const rect=svgNode('rect',{x:x(left,i)-width/2,y:a.y(r.count),width,height:left.y+left.h-a.y(r.count),rx:3,fill:c,'data-stride':r.stride,'data-count':r.count});
    svg.append(interact(rect,[`stride ${r.stride}`,`窓 ${number(r.count)}個`,`一意 ${number(r.unique)} frame`,`延べ ${number(r.occurrences)} frame`]));
    text(svg,x(left,i),a.y(r.count)-9,compact(r.count),{'text-anchor':'middle','font-size':11,fill:c});
    for(const box of [left,right])text(svg,x(box,i),box.y+box.h+22,r.stride,{'text-anchor':'middle','font-size':12,'font-weight':chosen?700:400,fill:chosen?PALETTE.teal:PALETTE.muted});
    if(finite(r.rate?.value)){
      const point={x:x(right,i),y:b.y(r.rate.value*100)};
      if(prior)svg.append(svgNode('line',{x1:prior.x,x2:point.x,y1:prior.y,y2:point.y,stroke:PALETTE.blue,'stroke-width':2}));
      const circle=svgNode('circle',{cx:point.x,cy:point.y,r:chosen?6:4.5,fill:chosen?PALETTE.teal:PALETTE.blue,stroke:'#fff','stroke-width':2,'data-stride':r.stride,'data-coverage':r.rate.value});
      svg.append(interact(circle,[`stride ${r.stride}`,`被覆 ${percent(r.rate.value)}`,`${number(r.rate.numerator)} / ${number(r.rate.denominator)} frame`]));
      text(svg,point.x,point.y-13,`${(r.rate.value*100).toFixed(2)}%`,{'text-anchor':'middle','font-size':10,fill:chosen?PALETTE.teal:PALETTE.blue});prior=point;
    }else{prior=null;text(svg,x(right,i),right.y+right.h/2,'—',{'text-anchor':'middle',fill:PALETTE.muted});}
  });
  for(const box of [left,right])text(svg,box.x+box.w/2,box.y+box.h+47,'開始位置のstride（frame）',{'text-anchor':'middle','font-size':11,fill:PALETTE.muted});
  legend(svg,[{label:`採用範囲のstride: ${selected}`,color:PALETTE.teal,width:250}],28,389);
  return plot.figure;
}

export function windowProfile(supervision,missing,count,{context=''}={}) {
  const plot=frame({title:'窓内の位置と教師・欠損の割合',subtitle:context,key:'window-profile',width:940,height:350,
    note:`各位置の分母は${number(count)}窓。この2系列は全状態を網羅せず、補間などは別の状態です。位置0のMDDはゼロです。`});
  const {svg,interact}=plot,box={x:74,y:99,w:801,h:174};
  const {x,y}=axes(svg,box,{xDomain:[0,31],yDomain:[0,100],xLabel:'32frame窓内の位置',yLabel:'割合（%）',xValues:[0,4,8,12,16,20,24,28,31],yFormat:v=>`${tick(v)}%`});
  if(!count){empty(svg,box,'採用窓がありません');return plot.figure;}
  legend(svg,[{label:'実測教師',color:PALETTE.blue,width:140},{label:'座標欠損',color:PALETTE.amber}],28,75);
  for(const [values,color,name] of [[supervision,PALETTE.blue,'実測教師'],[missing,PALETTE.amber,'座標欠損']]){
    const points=values.map((v,i)=>[x(i),y(v/count*100)]);
    svg.append(svgNode('polyline',{points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2}));
    values.forEach((v,i)=>{const p=svgNode('circle',{cx:x(i),cy:y(v/count*100),r:3,fill:color,'data-position':i,'data-value':v/count});svg.append(interact(p,[`${name} / 位置 ${i}`,`${percent(v/count)} (${number(v)} / ${number(count)}窓)`]));});
  }
  return plot.figure;
}

export function clipScatter(rows,{context,onClip,xKey='speed_p95',yKey='gap_p95'}={}) {
  const title='確認するclipを探す';
  const plot=frame({title,subtitle:context,key:'clip-scatter',width:940,height:428,
    note:'1点＝1clip。速度P95と欠損長P95の関係を表示します。クリック／Enterでclip詳細へ移動。欠損区間がないclipの欠損長分布は未定義のため、散布図には置きません。'});
  const {svg,interact}=plot,box={x:91,y:109,w:750,h:244};
  const colors={tracknet:PALETTE.blue,meiji:PALETTE.teal,chat_annotation:PALETTE.amber};
  const valid=rows.filter(r=>finite(r[xKey])&&finite(r[yKey]));
  const xMax=niceMax(Math.max(0,...valid.map(r=>r[xKey]))*1.05),yMax=niceMax(Math.max(0,...valid.map(r=>r[yKey]))*1.08);
  const {x,y}=axes(svg,box,{xDomain:[0,xMax],yDomain:[0,yMax],xLabel:'clip内の実測球速度 P95（元画像px/秒）',yLabel:'座標欠損長 P95（frame）'});
  legend(svg,Object.entries(colors).map(([label,color])=>({label,color,width:160})),28,77);
  text(svg,900,77,`${valid.length} / ${rows.length} clips`,{'text-anchor':'end','font-size':11,fill:PALETTE.muted});
  if(!valid.length)empty(svg,box,'両方の指標を持つclipがありません');
  valid.forEach(r=>{
    const circle=svgNode('circle',{cx:x(r[xKey]),cy:y(r[yKey]),r:4.5,fill:colors[r.source]??PALETTE.gray,'fill-opacity':.67,stroke:'#fff','stroke-width':.7,'data-clip':r.clip_id});
    svg.append(interact(circle,[r.clip_id,`${r.source} / ${r.split}`,`速度P95 ${number(r[xKey])} px/秒`,`座標欠損長P95 ${number(r[yKey])} frame`],()=>onClip(r.clip_id)));
  });
  return plot.figure;
}

export function trajectoryPlot(points,boundaries,onFrame,{context=''}={}) {
  const plot=frame({title:'clip内の実測位置と軌跡',subtitle:context,key:'trajectory',width:940,height:485,
    note:'色はフレーム順（青 → 紫）。連続したframeだけを結び、欠損・注釈境界・区間の外を跨いで線を引きません。点をクリックすると画像へ移動できます。'});
  const {svg,interact}=plot,box={x:84,y:97,w:766,h:303},cuts=new Set(boundaries);
  const {x}=axes(svg,box,{xDomain:[0,1],yDomain:[0,1],xLabel:'x / 画像幅',yLabel:'y / 画像高さ（下向き）',yFormat:v=>tick(1-v)});
  const y=value=>box.y+value*box.h;
  if(!points.length){empty(svg,box,'この範囲には実測位置がありません');return plot.figure;}
  points.forEach(([f,u,v],i)=>{
    const prior=points[i-1],color=`hsl(${210+70*i/Math.max(1,points.length-1)} 54% 46%)`;
    if(prior&&f===prior[0]+1&&!cuts.has(f))svg.append(svgNode('line',{x1:x(prior[1]),y1:y(prior[2]),x2:x(u),y2:y(v),stroke:color,'stroke-opacity':.4,'stroke-width':1.2}));
    const point=svgNode('circle',{cx:x(u),cy:y(v),r:2.8,fill:color,'data-frame':f});
    svg.append(interact(point,[`frame ${f}`,`x=${number(u)} · y=${number(v)}`],()=>onFrame(f)));
  });
  text(svg,box.x,454,`開始 frame ${points[0][0]}`,{fill:PALETTE.blue,'font-size':11});
  text(svg,box.x+box.w,454,`終了 frame ${points.at(-1)[0]}`,{'text-anchor':'end',fill:PALETTE.purple,'font-size':11});
  return plot.figure;
}
