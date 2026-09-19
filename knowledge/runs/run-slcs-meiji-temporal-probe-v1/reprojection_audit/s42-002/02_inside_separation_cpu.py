"""CPU-only separation of large pose residuals by inside-image status (read-only inputs).

Inputs (read-only): quality_report s42-probe-001, per-clip run dirs
  refined_scene.npz / refined_scene.metadata.json / quality_arrays.npz
Outputs (only OUT dir): inside_separation.json, top40_inside.csv,
  large100_by_clip.csv, hips_root.csv, both_inside_stats.json
No GPU, no torch, no source/config edits.
"""
import json, csv
from pathlib import Path
import numpy as np

WT = Path("/home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb")
QR_PATH = WT/"outputs/tennis_scene/analyze/meiji_rgb_v7_quality/s42-probe-001/quality_report.json"
OUT = WT/"outputs/tennis_scene/analyze/meiji_pose_residual_audit/s42-002"
OUT.mkdir(parents=True, exist_ok=True)
JOINTS = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

def project(pts, cam):
    R=np.asarray(cam['R'],float); t=np.asarray(cam['t'],float); K=np.asarray(cam['K'],float)
    local=pts@R.T+t; pix=local@K.T; depth=local[...,2]
    out=np.full(pix.shape[:-1]+(2,),np.nan); valid=np.isfinite(pix).all(-1)&(depth>1e-6)
    np.divide(pix[...,:2],pix[...,2:3],out=out,where=valid[...,None])
    return out,valid

def summ(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if a.size==0: return {"count":0}
    return {"count":int(a.size),"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.percentile(a,95)),"max":float(a.max())}

QR=json.load(open(QR_PATH))
clips=[k for k,v in QR['clips'].items() if v.get('status')=='completed']
all_top=[]; perclip={}; hips_rows=[]; both_inside_all=[]
for clip_id in sorted(clips):
    row=QR['clips'][clip_id]; run=Path(row['run_directory'])
    qa=np.load(run/'quality_arrays.npz'); err=qa['pose_reprojection_px']  # P,N,T,J
    d=np.load(run/'refined_scene.npz'); meta=json.load(open(run/'refined_scene.metadata.json'))
    W,H=int(d['width']),int(d['height'])
    uv=np.asarray(d['human_kp_2d'],np.float64); vis=np.asarray(d['human_kp_vis'],np.float64)
    kp3d=np.asarray(d['player_kp_3d'],np.float64); pos=np.asarray(d['player_position'],np.float64)
    P,N,T,J=vis.shape
    lq=meta['label_quality']; tw=np.asarray(lq['player_weight'],np.float32); ts=np.asarray(lq['player_source'],np.uint8)
    conf=vis.mean(axis=(1,3))
    finite=np.isfinite(pos).all(-1)&np.isfinite(np.asarray(d['player_yaw']))
    valid=finite&(conf>=0.3-1e-12)
    slcs_w=np.where(valid,conf.astype(np.float32),0.0).astype(np.float32)*tw
    slcs_valid=valid&(tw>0)
    fits=meta['reference']['camera_fits']
    # teacher projections per view
    tproj=np.full((P,N,T,J,2),np.nan); tfront=np.zeros((P,N,T,J),bool)
    rproj=np.full((P,N,T,2),np.nan); rfront=np.zeros((P,N,T),bool)
    for n in range(N):
        pr,fr=project(kp3d,fits[n])  # P,T,J,2
        tproj[:,n]=pr; tfront[:,n]=fr
        pr2,fr2=project(pos,fits[n])  # P,T,2
        rproj[:,n]=pr2; rfront[:,n]=fr2
    obs_px=uv*np.array([W,H])
    obs_inside=np.isfinite(obs_px).all(-1)&(obs_px[...,0]>=0)&(obs_px[...,0]<W)&(obs_px[...,1]>=0)&(obs_px[...,1]<H)
    tea_inside=np.isfinite(tproj).all(-1)&(tproj[...,0]>=0)&(tproj[...,0]<W)&(tproj[...,1]>=0)&(tproj[...,1]<H)
    counted=np.isfinite(err)&np.broadcast_to(slcs_valid[:,None,:,None],err.shape)
    large=counted&(err>100)
    # categories
    cats={}
    for name,m in [("A_obsIn_teaIn",obs_inside&tea_inside),("B_obsIn_teaOut",obs_inside&~tea_inside),("C_obsOut_teaIn",~obs_inside&tea_inside),("D_bothOut",~obs_inside&~tea_inside)]:
        sel=large&m
        v=err[sel]
        cats[name]={"count":int(sel.sum()),"frac_of_large":float(sel.sum()/max(1,int(large.sum()))),"mean":float(v.mean()) if v.size else None,"max":float(v.max()) if v.size else None}
    # both-inside subset of counted
    both=counted&obs_inside&tea_inside
    bi_stats={"frac_retained":float(both.sum()/max(1,int(counted.sum()))),"overall":summ(np.where(both,err,np.nan)),
              "by_camera":{f"cam{n}":summ(np.where(both[:,n],err[:,n],np.nan)) for n in range(N)},
              "by_player":{f"player{p}":summ(np.where(both[p],err[p],np.nan)) for p in range(P)}}
    for v in np.where(both,err,np.nan).reshape(-1):
        if np.isfinite(v): both_inside_all.append(float(v))
    # collect top candidates global later; also per-clip large rows for hips analysis (distinct slots)
    # store arrays for second pass (keep memory small: save per-clip npz refs via dict of small summaries + iterate top)
    perclip[clip_id]={"W":W,"H":H,"shape":[P,N,T,J],"counted":int(counted.sum()),"large100":int(large.sum()),
        "cats":cats,"both_inside":bi_stats,
        "large100_cam_dist":{f"cam{n}":int((large[:,n]).sum()) for n in range(N)},
        "large100_player_dist":{f"player{p}":int((large[p]).sum()) for p in range(P)}}
    # gather all counted samples for global top40
    idx=np.dstack(np.unravel_index(np.argsort(np.nan_to_num(np.where(counted,err,-1)).reshape(-1))[-60:],err.shape))[0][::-1]
    for (p,n,t,j) in idx:
        v=float(err[p,n,t,j])
        if not np.isfinite(v): continue
        all_top.append({"clip":clip_id,"frame":int(t),"player":int(p),"camera":f"cam{n}","joint":JOINTS[int(j)],"joint_id":int(j),
            "residual_px":v,"vis":float(vis[p,n,t,j]),
            "obs_px":[float(obs_px[p,n,t,j,0]),float(obs_px[p,n,t,j,1])],"obs_inside":bool(obs_inside[p,n,t,j]),
            "tea_px":[float(tproj[p,n,t,j,0]),float(tproj[p,n,t,j,1])],"tea_inside":bool(tea_inside[p,n,t,j]),"tea_infront":bool(tfront[p,n,t,j]),
            "hip_vis_L":float(vis[p,n,t,11]),"hip_vis_R":float(vis[p,n,t,12]),
            "teacher_source":int(ts[p,t]),"teacher_weight":float(tw[p,t]),"slcs_weight":float(slcs_w[p,t]),"player_conf":float(conf[p,t])})
# global top40 by residual among counted
all_top=sorted(all_top,key=lambda r:r['residual_px'],reverse=True)[:40]
# hips/root detail for top40 distinct slots + cross-view agreement (reload per clip as needed)
seen=set(); detail=[]
cache={}
for r in all_top:
    key=(r['clip'],r['frame'],r['player'],r['camera'])
    if key in seen: continue
    seen.add(key)
    clip_id,fr,p,cam=r['clip'],r['frame'],r['player'],int(r['camera'].replace('cam',''))
    if clip_id not in cache:
        run=Path(QR['clips'][clip_id]['run_directory'])
        dd=np.load(run/'refined_scene.npz'); mm=json.load(open(run/'refined_scene.metadata.json'))
        qa=np.load(run/'quality_arrays.npz')
        cache[clip_id]=(dd,mm,qa)
    dd,mm,qa=cache[clip_id]
    W,H=int(dd['width']),int(dd['height'])
    uv=np.asarray(dd['human_kp_2d'],np.float64); vis=np.asarray(dd['human_kp_vis'],np.float64)
    kp3d=np.asarray(dd['player_kp_3d'],np.float64); pos=np.asarray(dd['player_position'],np.float64)
    fits=mm['reference']['camera_fits']; err=qa['pose_reprojection_px']
    lq=mm['label_quality']; tw=np.asarray(lq['player_weight'],np.float32); ts=np.asarray(lq['player_source'],np.uint8)
    # hip participation across views
    both_views=int((((vis[p,:,fr,11]>=0.3)&(vis[p,:,fr,12]>=0.3))).sum())
    # root vs hip-center per view
    perview={}
    for n in range(vis.shape[1]):
        rp,_=project(pos[p,fr][None],fits[n]); rp=rp[0]
        hc=np.nan
        if vis[p,n,fr,11]>=0.3 and vis[p,n,fr,12]>=0.3:
            hc_px=(uv[p,n,fr,11]+uv[p,n,fr,12])/2*np.array([W,H])
            hc=float(np.linalg.norm(rp-hc_px)) if np.isfinite(rp).all() and np.isfinite(hc_px).all() else float('nan')
        e=err[p,n,fr,:]; f=e[np.isfinite(e)]
        perview[f"cam{n}"]={"hipL":float(vis[p,n,fr,11]),"hipR":float(vis[p,n,fr,12]),
            "root_proj":[float(rp[0]),float(rp[1])],"root_vs_hipcenter_px":hc,
            "n_counted_joints":int(np.isfinite(e).sum()),"mean_resid":float(f.mean()) if f.size else None,"max_resid":float(f.max()) if f.size else None}
    # temporal: Y and speed
    ytraj=pos[p,:,1].tolist()
    spd=qa['player_speed_mps'][p]  # T-1
    detail.append({"clip":clip_id,"frame":fr,"player":p,"focus_camera":f"cam{cam}",
        "hip_views_both_ge03":both_views,"triangulatable_ge2":bool(both_views>=2),
        "teacher_source":int(ts[p,fr]),"teacher_weight":float(tw[p,fr]),
        "Y_traj_win":[float(x) for x in pos[p,max(0,fr-2):fr+3,1]],
        "speed_win":[float(x) for x in spd[max(0,fr-2):min(len(spd),fr+3)]],
        "perview":perview})
    hips_rows.append({"clip":clip_id,"frame":fr,"player":p,"camera":f"cam{cam}","hipL":float(vis[p,cam,fr,11]),"hipR":float(vis[p,cam,fr,12]),"both_ge03":bool(vis[p,cam,fr,11]>=0.3 and vis[p,cam,fr,12]>=0.3),"views_both_ge03":both_views})

res={"clips":perclip,"top40":all_top,"top40_detail":detail,
 "both_inside_pooled":{"count":len(both_inside_all),"mean":float(np.mean(both_inside_all)) if both_inside_all else None,"median":float(np.median(both_inside_all)) if both_inside_all else None,"p95":float(np.percentile(both_inside_all,95)) if both_inside_all else None,"max":float(np.max(both_inside_all)) if both_inside_all else None}}
json.dump(res,open(OUT/"inside_separation.json","w"),indent=2)
with open(OUT/"top40_inside.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["clip","frame","player","camera","joint","joint_id","residual_px","vis","obs_px","obs_inside","tea_px","tea_inside","tea_infront","hip_vis_L","hip_vis_R","teacher_source","teacher_weight","slcs_weight","player_conf"])
    w.writeheader()
    for r in all_top:
        d2=dict(r); d2['obs_px']=str([round(x,1) for x in r['obs_px']]); d2['tea_px']=str([round(x,1) if np.isfinite(x) else None for x in r['tea_px']])
        w.writerow(d2)
with open(OUT/"large100_by_clip.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["clip","counted","large100","frac_large","A_in_in","B_obsIn_teaOut","C_obsOut_teaIn","D_bothOut","both_inside_retained_frac","both_inside_mean","both_inside_p95","both_inside_max"])
    w.writeheader()
    for cid,pc in perclip.items():
        w.writerow({"clip":cid,"counted":pc['counted'],"large100":pc['large100'],"frac_large":round(pc['large100']/max(1,pc['counted']),5),
            "A_in_in":pc['cats']['A_obsIn_teaIn']["count"],"B_obsIn_teaOut":pc['cats']['B_obsIn_teaOut']["count"],"C_obsOut_teaIn":pc['cats']['C_obsOut_teaIn']["count"],"D_bothOut":pc['cats']['D_bothOut']["count"],
            "both_inside_retained_frac":round(pc['both_inside']['frac_retained'],4),"both_inside_mean":round(pc['both_inside']['overall'].get('mean',0) or 0,3),"both_inside_p95":round(pc['both_inside']['overall'].get('p95',0) or 0,2),"both_inside_max":round(pc['both_inside']['overall'].get('max',0) or 0,1)})
with open(OUT/"hips_root.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["clip","frame","player","camera","hipL","hipR","both_ge03","views_both_ge03"])
    w.writeheader(); w.writerows(hips_rows)
print("saved to",OUT)
for cid,pc in perclip.items():
    print(cid, "counted",pc['counted'],"large",pc['large100'],pc['cats'],"both",pc['both_inside']['frac_retained'],pc['both_inside']['overall'])
print("pooled both-inside",res["both_inside_pooled"])
