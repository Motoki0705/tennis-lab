"""CPU-only audit of NEW final 12m/s speed rejections under shoulder-gated candidate.

Reads (read-only): run dirs scene.npz/scene.metadata.json/refined metadata + s42-001 probe.
Writes only s42-002. No GPU, no threshold changes, no production edits.
"""
import json, csv, sys
from pathlib import Path
import numpy as np
WT = Path("/home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb")
sys.path.insert(0, str(WT))
from src.tennis_scene.dataset_pipeline.geometry import TriangulationSettings, triangulate_ball, fill_triangulation_gaps
from src.tennis_scene.dataset_pipeline.quality import project
QR = json.load(open(WT/"outputs/tennis_scene/analyze/meiji_rgb_v7_quality/s42-probe-001/quality_report.json"))
OUT = WT/"outputs/tennis_scene/analyze/meiji_root_support_probe/s42-002"
OUT.mkdir(parents=True, exist_ok=True)
CONF=0.3; ST=dict(max_reprojection_px=40.0,max_speed_mps=12.0,max_abs_xy_m=(12.0,25.0),height_range_m=(0.25,1.8))
TARGETS=[("video_001/clip_001",1),("video_000/clip_007",1),("video_000/clip_009",1)]
res={"targets":{}}
all_rows=[]
for clip_id,p in TARGETS:
    run=Path(QR['clips'][clip_id]['run_directory'])
    raw=np.load(run/'scene.npz'); rmeta=json.load(open(run/'scene.metadata.json'))
    fmeta=json.load(open(run/'refined_scene.metadata.json'))
    W,H=int(raw['width']),int(raw['height']); fps=float(raw['fps']); fits=rmeta['reference']['camera_fits']
    uv=np.asarray(raw['human_kp_2d'],np.float64); vis=np.asarray(raw['human_kp_vis'],np.float64)
    prior=np.asarray(raw['player_position'],np.float64)
    T=vis.shape[2]; gap=round(0.1*fps); kw=dict(size=(W,H),fps=fps,settings=TriangulationSettings(**ST))
    hips=np.take(uv[p],[11,12],axis=2).mean(axis=2)
    vh=(np.take(vis[p],[11,12],axis=2)>=CONF).all(axis=2); vs=(np.take(vis[p],[5,6],axis=2)>=CONF).all(axis=2)
    vc=vh&vs
    tcur=triangulate_ball(hips,vh,fits,**kw); tcand=triangulate_ball(hips,vc,fits,**kw)
    pcur,scur=fill_triangulation_gaps(tcur,prior[p],max_gap_frames=gap)
    pcand,scand=fill_triangulation_gaps(tcand,prior[p],max_gap_frames=gap)
    # replica check
    rec=np.asarray(np.load(run/'refined_scene.npz')['player_position'],np.float64)[p]
    rec_src=np.asarray(fmeta['label_quality']['player_source'],np.uint8)[p]
    fidelity={"max_abs_diff_m":float(np.abs(pcur-rec).max()),"src_match":bool((scur==rec_src).all())}
    def zmask(pos):
        sp=np.linalg.norm(np.diff(pos,axis=0),axis=-1)*fps
        return (np.r_[sp>12.0,False]|np.r_[False,sp>12.0]),sp
    zm_cur,sp_cur=zmask(pcur); zm_cand,sp_cand=zmask(pcand)
    new_frames=sorted(set(np.flatnonzero(zm_cand).tolist())-set(np.flatnonzero(zm_cur).tolist()))
    resolved_frames=sorted(set(np.flatnonzero(zm_cur).tolist())-set(np.flatnonzero(zm_cand).tolist()))
    cnt_cur=vh.sum(0); cnt_cand=vc.sum(0)
    rows=[]
    for t in new_frames:
        # causative jump segments touching t: (t-1,t) and (t,t+1)
        segs=[]
        for a in [t-1,t]:
            if 0<=a<T-1:
                jc=float(np.linalg.norm(pcur[a+1]-pcur[a])*fps) if False else None
                jcur=float(np.linalg.norm(pcur[a+1]-pcur[a])*fps); jcand=float(np.linalg.norm(pcand[a+1]-pcand[a])*fps)
                segs.append({"seg":[a,a+1],"spd_cur":round(jcur,2),"spd_cand":round(jcand,2),
                    "fast_cur":bool(jcur>12.0),"fast_cand":bool(jcand>12.0),
                    "views_cur":[int(vh[n,t if False else a]),int(0)] if False else [int(cnt_cur[a]),int(cnt_cur[a+1])],
                    "views_cand":[int(cnt_cand[a]),int(cnt_cand[a+1])],
                    "codes_cur":[int(tcur.rejection_code[a]),int(tcur.rejection_code[a+1])],
                    "codes_cand":[int(tcand.rejection_code[a]),int(tcand.rejection_code[a+1])]})
        # boundary flag: participation differs cur vs cand at t-1/t/t+1
        win=range(max(0,t-1),min(T,t+2))
        bnd=any(int(cnt_cur[i])!=int(cnt_cand[i]) for i in win)
        chg=any(int(cnt_cand[i])!=int(cnt_cand[i+1]) for i in range(max(0,t-1),min(T-1,t+1)))
        # cam0/1 hip-center residuals both variants at t
        hr={}
        for tag,pp in [("cur",pcur),("cand",pcand)]:
            for n in [0,1]:
                pj,_=project(pp[t][None],fits[n])
                if vis[p,n,t,11]>=CONF and vis[p,n,t,12]>=CONF:
                    hc=(uv[p,n,t,11]+uv[p,n,t,12])/2*np.array([W,H])
                    hr[f"{tag}_cam{n}"]=round(float(np.linalg.norm(pj[0]-hc)),2) if np.isfinite(pj).all() and np.isfinite(hc).all() else None
                else: hr[f"{tag}_cam{n}"]=None
        # class guess (auditor labels, evidence-linked):
        # B=boundary-switch (participation differs in window), H=hidden-bad-view (cand 2-view but cam0/1 residual large), S=2-view jitter (small residuals, small jump just over 12)
        jmax=max([s["spd_cand"] for s in segs])
        hmax=[v for k,v in hr.items() if k.startswith("cand") and v is not None]
        hmax=max(hmax) if hmax else None
        if bnd: cls="B-boundary-switch"
        elif hmax is not None and hmax>8.0: cls="H-bad-remaining-view"
        else: cls="S-2view-jitter"
        rows.append({"frame":t,"spd_cand_max":jmax,"class":cls,"boundary_in_window":bnd,"cand_switch_in_window":chg,
            "views_cur_win":{str(i):int(cnt_cur[i]) for i in win},"views_cand_win":{str(i):int(cnt_cand[i]) for i in win},
            "segs":segs,"hip_res":hr,
            "src_cur":int(scur[t]),"src_cand":int(scand[t]),
            "pos_cur":list(map(lambda x:round(float(x),3),pcur[t])),"pos_cand":list(map(lambda x:round(float(x),3),pcand[t]))})
        all_rows.append({"clip":clip_id,"player":p,**{k:v for k,v in rows[-1].items() if k in ("frame","spd_cand_max","class","src_cur","src_cand")},"hip_res":str(rows[-1]["hip_res"])})
    res["targets"][f"{clip_id}/P{p}"]={"fidelity":fidelity,"fps":fps,
        "n_speed_cur":int(zm_cur.sum()),"n_speed_cand":int(zm_cand.sum()),
        "new_frames":new_frames,"resolved_frames":resolved_frames,
        "code6_cur":int((tcur.rejection_code==6).sum()),"code6_cand":int((tcand.rejection_code==6).sum()),
        "details":rows}
json.dump(res,open(OUT/"speed_reject_audit.json","w"),indent=2)
with open(OUT/"new_reject_frames.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["clip","player","frame","spd_cand_max","class","src_cur","src_cand","hip_res"]); w.writeheader(); w.writerows(all_rows)
print("saved",OUT)
for k,v in res["targets"].items():
    from collections import Counter
    print(k,"fidelity",v["fidelity"],"cur",v["n_speed_cur"],"cand",v["n_speed_cand"],"new",v["new_frames"],"resolved",v["resolved_frames"],"c6",v["code6_cur"],v["code6_cand"])
    print("  classes",Counter([d["class"] for d in v["details"]]))
