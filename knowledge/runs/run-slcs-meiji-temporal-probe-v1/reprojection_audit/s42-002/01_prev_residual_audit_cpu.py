import json, numpy as np
from pathlib import Path
WT = Path("/home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb")
OUT = WT/"outputs/tennis_scene/analyze/meiji_pose_residual_audit/s42-001"
OUT.mkdir(parents=True, exist_ok=True)
QR = json.load(open(WT/"outputs/tennis_scene/analyze/meiji_rgb_v7_quality/s42-probe-001/quality_report.json"))
JOINTS = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
# QualityConfig matching report
MIN_CONF=0.3
result={"clips":{},"overall":{}}
all_rows=[]
for clip_id, row in QR["clips"].items():
    if row.get("status")!="completed":
        continue
    run = Path(row["run_directory"])
    qa = np.load(run/"quality_arrays.npz")
    pose_err = qa["pose_reprojection_px"]  # (P,N,T,J)
    ref_npz = np.load(run/"refined_scene.npz")
    meta = json.load(open(run/"refined_scene.metadata.json"))
    vis = ref_npz["human_kp_vis"]  # (P,N,T,J)
    lq = meta["label_quality"]
    teacher_w = np.asarray(lq["player_weight"], np.float32)  # (P,T)
    teacher_src = np.asarray(lq["player_source"], np.uint8)
    # SLCS mask recompute: conf = mean over N,J ; valid = finite & conf>=0.3 ; weight = conf * teacher_w ; valid &= teacher_w>0
    P,N,T,J = vis.shape
    conf = vis.mean(axis=(1,3))  # (P,T)
    player_pos = ref_npz["player_position"]; player_yaw = ref_npz["player_yaw"]
    finite = np.isfinite(player_pos).all(-1) & np.isfinite(player_yaw)
    valid = finite & (conf>=MIN_CONF-1e-12)
    slcs_w = np.where(valid, conf.astype(np.float32), 0.0).astype(np.float32) * teacher_w
    slcs_valid = valid & (teacher_w>0)
    # slcs positive mask broadcast to joints
    mask = slcs_valid[:,None,:,None]  # (P,1,T,1) -> broadcast
    vals = np.where(mask, pose_err, np.nan)
    finite_vals = vals[np.isfinite(vals)]
    # also total possible vs counted
    total_possible = int(mask.sum()*N*J) if False else int(np.broadcast_to(mask, pose_err.shape).sum())
    # nan breakdown: among slcs-positive slots, how many nan due to vis<0.3/behind?
    slcs_slots = np.broadcast_to(mask, pose_err.shape)
    nan_in_positive = int((slcs_slots & ~np.isfinite(pose_err)).sum())
    # vis of counted samples
    vis_counted = vis[np.isfinite(vals)] if finite_vals.size else np.array([])
    # stats
    def summ(a):
        a=np.asarray(a,float); a=a[np.isfinite(a)]
        if a.size==0: return {"count":0}
        return {"count":int(a.size),"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.percentile(a,95)),"max":float(a.max()),"frac_gt40":float((a>40).mean()),"frac_gt100":float((a>100).mean())}
    clip_res={"shape":list(pose_err.shape),"total_slcs_slots":int(total_possible),"finite_in_slcs":int(finite_vals.size),"nan_in_slcs_positive":int(nan_in_positive),"overall":summ(finite_vals)}
    # by joint
    by_joint={}
    for j in range(J):
        by_joint[JOINTS[j]]=summ(vals[:,:,:,j])
    clip_res["by_joint"]=by_joint
    # by camera
    by_cam={}
    for n in range(N):
        by_cam[f"cam{n}"]=summ(vals[:,n,:,:])
    clip_res["by_camera"]=by_cam
    # by player
    by_p={}
    for p in range(P):
        by_p[f"player{p}"]=summ(vals[p])
    clip_res["by_player"]=by_p
    # by vis bin (only counted samples)
    vis_bins={}
    edges=[(0.3,0.5,"0.30-0.50"),(0.5,0.8,"0.50-0.80"),(0.8,1.0,"0.80-1.00"),(1.0,1.0001,"saturated_1.0")]
    # vis array aligned: need vis values at counted positions
    flat_err=vals.reshape(-1); flat_vis=vis.reshape(-1)
    # note vals includes nan where excluded; select finite
    m=np.isfinite(flat_err)
    fe=flat_err[m]; fv=flat_vis[m]
    for lo,hi,name in edges:
        if name=="saturated_1.0":
            sel=(fv>=1.0-1e-6)
        elif name=="0.80-1.00":
            sel=(fv>=0.8)&(fv<1.0-1e-6)
        else:
            sel=(fv>=lo)&(fv<hi)
        vis_bins[name]=summ(fe[sel])
        vis_bins[name]["share_of_counted"]=float(sel.sum()/max(1,m.sum()))
    clip_res["by_vis_bin"]=vis_bins
    # by teacher source
    by_src={}
    for code in [0,1,2,3]:
        # expand source (P,T) to (P,N,T,J)
        src_exp = np.broadcast_to(teacher_src[:,None,:,None], pose_err.shape)
        sel = np.broadcast_to(mask, pose_err.shape) & (src_exp==code)
        by_src[f"src{code}"]=summ(np.where(sel, pose_err, np.nan))
    clip_res["by_teacher_source"]=by_src
    # by teacher weight value
    by_tw={}
    w_exp = np.broadcast_to(teacher_w[:,None,:,None], pose_err.shape)
    for label, cond in [("w1.0", np.abs(w_exp-1.0)<1e-6), ("w0.4", np.abs(w_exp-0.4)<1e-6), ("w0", w_exp<=0)]:
        sel = np.broadcast_to(mask, pose_err.shape) & cond
        by_tw[label]=summ(np.where(sel, pose_err, np.nan))
    clip_res["by_teacher_weight"]=by_tw
    # per-(P,T,N) frame-level concentration: mean/median/max over joints for counted joints
    # for each (p,t,n) with slcs valid and at least 1 finite joint
    frame_stats=[]
    for p in range(P):
        for t in range(T):
            if not slcs_valid[p,t]:
                continue
            for n in range(N):
                v = pose_err[p,n,t,:]
                f = v[np.isfinite(v)]
                if f.size==0:
                    continue
                frame_stats.append((float(f.mean()), float(np.median(f)), float(f.max()), p, t, n, int(f.size), float(vis[p,n,t,:][np.isfinite(v)].mean()) if np.isfinite(vis[p,n,t,:]).any() else float('nan'), int(teacher_src[p,t]), float(teacher_w[p,t]), float(conf[p,t])))
    frame_stats=np.array(frame_stats, dtype=object) if frame_stats else np.zeros((0,11))
    if len(frame_stats):
        means=np.array([r[0] for r in frame_stats],float); meds=np.array([r[1] for r in frame_stats],float); maxs=np.array([r[2] for r in frame_stats],float)
        clip_res["frame_level"]={"n_frameslots":int(len(frame_stats)),"mean_of_means":float(means.mean()),"frac_frameslots_mean_gt40":float((means>40).mean()),"frac_frameslots_median_gt40":float((meds>40).mean()),"frac_frameslots_max_gt100":float((maxs>100).mean()),"frac_frameslots_max_gt200":float((maxs>200).mean())}
    else:
        clip_res["frame_level"]={"n_frameslots":0}
    # top 30 joint outliers
    idx = np.dstack(np.unravel_index(np.argsort(np.nan_to_num(vals,nan=-1).reshape(-1))[-30:], vals.shape))[0][::-1]
    tops=[]
    for (p,n,t,j) in idx:
        v=float(vals[p,n,t,j])
        if not np.isfinite(v):
            continue
        tops.append({"frame":int(t),"player":int(p),"camera":f"cam{n}","joint":JOINTS[int(j)],"joint_id":int(j),"residual_px":v,"vis":float(vis[p,n,t,j]),"teacher_source":int(teacher_src[p,t]),"teacher_weight":float(teacher_w[p,t]),"slcs_weight":float(slcs_w[p,t]),"player_conf":float(conf[p,t])})
        all_rows.append({"clip":clip_id,**tops[-1]})
    clip_res["top30"]=tops
    # saturation info
    clip_res["vis_saturation_frac"]=float((vis>=1.0-1e-6).mean())
    clip_res["pose_visibility_conversion"]=meta.get("pose_visibility_conversion")
    result["clips"][clip_id]=clip_res

# overall top20 across clips
flat_all = sorted(all_rows, key=lambda r: r["residual_px"], reverse=True)[:40]
result["top40_across_clips"]=flat_all
# save
with open(OUT/"residual_audit.json","w") as f:
    json.dump(result,f,indent=2)
import csv
with open(OUT/"top40_outliers.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["clip","frame","player","camera","joint","joint_id","residual_px","vis","teacher_source","teacher_weight","slcs_weight","player_conf"])
    w.writeheader(); w.writerows(flat_all)
print("saved", OUT)
for cid,cr in result["clips"].items():
    print("==",cid, cr["overall"], "slots",cr["total_slcs_slots"],"finite",cr["finite_in_slcs"],"nan_in_pos",cr["nan_in_slcs_positive"])
    print(" by_cam", {k:(round(v.get('mean',0),2) if v.get('count') else None, v.get('max'), v.get('count')) for k,v in cr["by_camera"].items()})
    print(" by_vis", {k:(round(v.get('mean',0),2) if v.get('count') else None, round(v.get('share_of_counted',0),3)) for k,v in cr["by_vis_bin"].items()})
    print(" by_src", {k:(round(v.get('mean',0),2) if v.get('count') else None, v.get('count'), round(v.get('frac_gt40',0),3)) for k,v in cr["by_teacher_source"].items()})
    print(" frame_level", cr["frame_level"])
