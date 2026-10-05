export function trajectorySegments(points, valid, start=0, end=valid.length) {
  const result=[];let segment=[];
  for(let frame=start;frame<end;frame++){
    if(valid[frame]) segment.push({frame,point:points[frame]});
    else if(segment.length){result.push(segment);segment=[];}
  }
  if(segment.length)result.push(segment);
  return result;
}

export function reasonName(domain,code,labels){
  if(domain==='root'&&code===1)return 'NO_RECOVERED_BODY';
  return labels[String(code)]??`UNKNOWN_CODE_${code}`;
}

export function frameFromTime(time,fps,count){
  return Math.min(count-1,Math.max(0,Math.floor(time*fps+1e-4)));
}

export function videoTimeForFrame(frame,fps){return (frame+.5)/fps;}

export function frameRows(scene,frame){
  if(!scene)return [];
  const a=scene.arrays;
  const rows=scene.player_ids.map((id,p)=>({name:`player ${id}`,observed:a.player_observed[p][frame],
    valid:a.player_valid[p][frame],heading:a.player_heading_valid[p][frame],mesh:a.player_smpl_valid[p][frame],
    joints:a.player_kp_3d_vis[p][frame].filter(Boolean).length,
    reason:reasonName('root',a.player_rejection_code[p][frame],scene.rejection_labels),
    jointReasons:Object.entries(a.player_kp_3d_rejection_code[p][frame].reduce((counts,code)=>{
      if(code)counts[code]=(counts[code]??0)+1;return counts;
    },{})).map(([code,count])=>`${reasonName('point',Number(code),scene.rejection_labels)}: ${count}`).join(', ')}));
  rows.push({name:'ball',observed:a.ball_vis.reduce((n,view)=>n+Number(view[frame]),0),valid:a.ball_3d_valid[frame],
    reason:reasonName('point',a.ball_rejection_code[frame],scene.rejection_labels)});
  return rows;
}
