// Small, dependency-free SVG primitives shared by the statistical views.
// Every export contains its own typography, axes, title and legend.
import {node, number} from './charts.mjs';
const NS = 'http://www.w3.org/2000/svg';
let serial = 0;
export const PALETTE = {blue:'#2563a6', teal:'#087f8c', amber:'#cb811b', purple:'#8b5daf', gray:'#64748b', ink:'#223348', muted:'#64748b', grid:'#e6edf3'};
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export function svgNode(tag, attrs = {}, text) {
  const element = document.createElementNS(NS, tag);
  for (const [key,value] of Object.entries(attrs)) element.setAttribute(key, String(value));
  if (text !== undefined) element.textContent = String(text);
  return element;
}
export function text(svg, x, y, value, attrs = {}) {
  const t = svgNode('text', {x, y, fill:PALETTE.ink, 'font-size':12, ...attrs}, value);
  svg.append(t); return t;
}
export const compact = value => new Intl.NumberFormat('ja-JP', {notation:'compact', maximumFractionDigits:1}).format(value);
export function tick(value) {
  if (!finite(value)) return '—';
  if (Math.abs(value) >= 10000) return compact(value);
  return new Intl.NumberFormat('ja-JP', {maximumFractionDigits: Math.abs(value) < 1 ? 3 : 2}).format(value);
}
export function niceMax(value) {
  if (!(value > 0)) return 1;
  const rough=value/4, power=10**Math.floor(Math.log10(rough)), fraction=rough/power;
  const step=(fraction<=1?1:fraction<=2?2:fraction<=2.5?2.5:fraction<=5?5:10)*power;
  return Math.ceil(value/step)*step;
}
export function scale(domain, range) {
  const span = domain[1]-domain[0];
  return value => range[0] + (value-domain[0]) / (span || 1) * (range[1]-range[0]);
}
export function ticks(domain,count=5){
  const [lo,hi]=domain,rough=(hi-lo)/Math.max(1,count-1);
  if(!(rough>0))return [lo];
  const power=10**Math.floor(Math.log10(rough));
  const step=[1,2,2.5,5,10].map(v=>v*power).sort((a,b)=>Math.abs(Math.log(a/rough))-Math.abs(Math.log(b/rough)))[0];
  const result=[];
  for(let n=Math.ceil(lo/step-1e-9);n*step<=hi+step*1e-9;n++)result.push(Number((n*step).toPrecision(12)));
  return result;
}
export function axes(svg, box, {xDomain=[0,1],yDomain=[0,1],xLabel='',yLabel='',xFormat=tick,yFormat=tick,xTicks=5,yTicks=5,xValues=null,yValues=null,vertical=true,horizontal=true}={}) {
  const x=scale(xDomain,[box.x,box.x+box.w]), y=scale(yDomain,[box.y+box.h,box.y]);
  if(horizontal)for(const value of yValues??ticks(yDomain,yTicks)){
    const py=y(value);
    svg.append(svgNode('line',{x1:box.x,x2:box.x+box.w,y1:py,y2:py,stroke:PALETTE.grid}));
    text(svg,box.x-10,py+4,yFormat(value),{'text-anchor':'end',fill:PALETTE.muted,'font-size':11,'data-axis':'y','data-tick':value});
  }
  if(vertical)for(const value of xValues??ticks(xDomain,xTicks)){
    const px=x(value);
    svg.append(svgNode('line',{x1:px,x2:px,y1:box.y,y2:box.y+box.h,stroke:PALETTE.grid}));
    text(svg,px,box.y+box.h+22,xFormat(value),{'text-anchor':'middle',fill:PALETTE.muted,'font-size':11,'data-axis':'x','data-tick':value});
  }
  svg.append(svgNode('path',{d:`M${box.x},${box.y}V${box.y+box.h}H${box.x+box.w}`,fill:'none',stroke:'#a8b7c5'}));
  if(xLabel)text(svg,box.x+box.w/2,box.y+box.h+46,xLabel,{'text-anchor':'middle',fill:PALETTE.muted,'font-size':11});
  if(yLabel)text(svg,box.x,box.y-14,yLabel,{fill:PALETTE.muted,'font-size':11});
  return {x,y};
}
export function legend(svg, items, x=28, y=74) {
  let offset=x;
  for(const item of items){
    svg.append(svgNode('rect',{x:offset,y:y-8,width:9,height:9,rx:2,fill:item.color}));
    text(svg,offset+15,y,item.label,{'font-size':11,fill:PALETTE.muted});
    offset+=item.width??(item.label.length*10+31);
  }
}
export function frame({title,subtitle='',width=760,height=410,key='chart',note='',controls=[]}) {
  const id=`statistics-plot-${++serial}`;
  const figure=node('figure','','statistics-figure statistics-chart-card');
  figure.dataset.chart=key;
  const caption=node('figcaption',title,'statistics-sr-only');figure.append(caption);
  const toolbar=node('div','','statistics-chart-toolbar');
  toolbar.append(...controls);
  const save=node('button','SVG保存','statistics-chart-save');save.type='button';save.setAttribute('aria-label',`${title}をSVG保存`);toolbar.append(save);figure.append(toolbar);
  const scroll=node('div','','statistics-plot-scroll');
  const svg=svgNode('svg',{xmlns:NS,viewBox:`0 0 ${width} ${height}`,width,height,class:'statistics-plot',role:'img','aria-labelledby':`${id}-title ${id}-description`});
  svg.append(svgNode('title',{id:`${id}-title`},title),svgNode('desc',{id:`${id}-description`},[subtitle,note].filter(Boolean).join('。')));
  svg.append(svgNode('style',{},'text{font-family:"Noto Sans JP","Yu Gothic",system-ui,sans-serif;stroke:none} .plot-focus:focus{outline:none;stroke:#172b4d;stroke-width:2}'));
  svg.append(svgNode('rect',{width,height,fill:'#ffffff'}));
  text(svg,28,29,title,{'font-size':17,'font-weight':650});
  if(subtitle)text(svg,28,51,subtitle,{'font-size':11,fill:PALETTE.muted});
  scroll.append(svg);figure.append(scroll);
  if(note)figure.append(node('p',note,'statistics-chart-note'));
  const tooltip=node('div','','statistics-tooltip');tooltip.hidden=true;tooltip.setAttribute('role','status');figure.append(tooltip);
  function hide(){tooltip.hidden=true;}
  function show(lines, event, target) {
    tooltip.replaceChildren(...lines.map((line,index)=>node(index===0?'strong':'div',line)));
    tooltip.hidden=false;
    const bounds=figure.getBoundingClientRect(), hit=target.getBoundingClientRect();
    const px=event?.clientX??(hit.left+hit.width/2),py=event?.clientY??(hit.top+hit.height/2);
    tooltip.style.left=`${Math.max(8,Math.min(px-bounds.left+12,figure.clientWidth-tooltip.offsetWidth-8))}px`;
    tooltip.style.top=`${Math.max(48,Math.min(py-bounds.top+12,figure.clientHeight-tooltip.offsetHeight-8))}px`;
  }
  function interact(element, lines, onSelect) {
    element.classList.add('plot-focus');element.setAttribute('tabindex','0');
    element.setAttribute('aria-label',lines.join('、'));
    if(onSelect){element.setAttribute('role','button');element.style.cursor='pointer';}
    element.addEventListener('pointerenter',event=>show(lines,event,element));
    element.addEventListener('pointermove',event=>show(lines,event,element));
    element.addEventListener('pointerleave',hide);
    element.addEventListener('focus',()=>show(lines,null,element));element.addEventListener('blur',hide);
    element.addEventListener('keydown',event=>{if(event.key==='Escape')hide();if(onSelect&&['Enter',' '].includes(event.key)){event.preventDefault();onSelect();}});
    if(onSelect)element.addEventListener('click',onSelect);
    return element;
  }
  save.onclick=()=>{
    const copy=svg.cloneNode(true);copy.removeAttribute('class');copy.setAttribute('style','background:white');
    const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml;charset=utf-8'}));
    const link=node('a');link.href=url;link.download=`ball-statistics-${key}.svg`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  return {figure,svg,interact,id,width,height};
}
export function selectControl(caption, entries, value, onChange) {
  const label=node('label',caption,'statistics-chart-control');
  const select=node('select');select.setAttribute('aria-label',caption);
  select.append(...entries.map(([key,title])=>new Option(title,key)));select.value=value;
  select.onchange=()=>onChange(select.value);label.append(select);return label;
}
export function empty(svg, box, message='有効なデータがありません') {
  svg.append(svgNode('rect',{x:box.x,y:box.y,width:box.w,height:box.h,rx:6,fill:'#f4f7fa',stroke:'#dde6ee','stroke-dasharray':'4 4'}));
  text(svg,box.x+box.w/2,box.y+box.h/2,message,{'text-anchor':'middle',fill:PALETTE.muted,'font-size':13});
}
export const summaryLines = (name,s) => [name,`P5 ${number(s.p5)} · 中央値 ${number(s.median)} · P95 ${number(s.p95)}`,`平均 ${number(s.mean)} · 有効 ${number(s.n)} / 対象 ${number(s.eligible)}`];
