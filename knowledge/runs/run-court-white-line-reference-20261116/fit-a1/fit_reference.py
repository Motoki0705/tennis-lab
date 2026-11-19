"""Reference fitting from RGB paint-edge pairs only; no prediction inputs.

The old H defines search strips. Paired bright-stripe edges define their centre.
Deterministic line consensus rejects occlusion samples; court-line consensus
rejects incompatible static structures. Final optimization starts from old H.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d

BASE=Path(__file__).resolve().parent
POINTS=court_keypoints_3d(STANDARD_COURT_CONFIG)[:14,:2].numpy().astype(float)
EDGES=np.array([(0,1),(2,3),(0,2),(1,3),(4,5),(6,7),(8,9),(10,11),(12,13)])
NAMES=['far_baseline','near_baseline','left_doubles','right_doubles','left_singles','right_singles','far_service','near_service','centre_service']
CFG={'radius_px':48.,'sample_spacing_px':8.,'profile_step_px':.5,'profile_sigma_px':.6,
     'min_gradient':2.,'min_contrast':6.,'min_width_px':1.,'max_width_px':18.,
     'line_angle_limit_degrees':8.,'line_inlier_distance_px':1.5,'ransac_trials':500,
     'min_line_locations':8,'min_frame_count':8,'min_coverage':.35,'court_line_tolerance_px':3.,
     'min_court_lines':5,'loss':'soft_l1','loss_scale_px':1.5,'seed':2190}


def project(h,xy=POINTS):
    a=np.c_[xy,np.ones(len(xy))]@h.T
    return a[:,:2]/a[:,2,None]


def image_lines(h):
    p=project(h);lines=np.cross(np.c_[p[EDGES[:,0]],np.ones(9)],np.c_[p[EDGES[:,1]],np.ones(9)])
    return lines/np.linalg.norm(lines[:,:2],axis=1)[:,None]


def stripe_pairs(profile,offsets):
    smooth=gaussian_filter1d(profile,CFG['profile_sigma_px']/CFG['profile_step_px'])
    derivative=np.gradient(smooth,CFG['profile_step_px'])
    positive,_=find_peaks(derivative,height=CFG['min_gradient'],distance=2)
    negative,_=find_peaks(-derivative,height=CFG['min_gradient'],distance=2)
    pairs=[]
    def subpixel(i):
        d=derivative if derivative[i]>0 else -derivative
        denominator=d[i-1]-2*d[i]+d[i+1]
        delta=.5*(d[i-1]-d[i+1])/denominator if abs(denominator)>1e-8 else 0.
        return offsets[i]+np.clip(delta,-.5,.5)*CFG['profile_step_px']
    for a in positive:
        for b in negative:
            width=offsets[b]-offsets[a]
            if not CFG['min_width_px']<=width<=CFG['max_width_px']:continue
            interior=float(np.mean(smooth[a:b+1]));outside=max(float(np.median(smooth[max(0,a-8):max(1,a-3)])),float(np.median(smooth[min(len(smooth)-1,b+3):min(len(smooth),b+8)])))
            contrast=interior-outside
            if contrast<CFG['min_contrast']:continue
            # Adjacent rising/falling edges bound one stripe, not two stripes.
            if np.any((positive>a)&(positive<b)&(derivative[positive]>.7*derivative[a])):continue
            pairs.append(((subpixel(a)+subpixel(b))/2,contrast,width))
    return sorted(pairs,key=lambda p:(-p[1],abs(p[0]),p[0]))[:5]


def extract(images,h):
    projected=project(h);observations=[]
    offsets=np.arange(-CFG['radius_px'],CFG['radius_px']+.01,CFG['profile_step_px'])
    for index,(a,b) in enumerate(EDGES):
        start,end=projected[[a,b]]
        okay,clipped_a,clipped_b=cv2.clipLine((2,2,1916,1076),tuple(np.rint(start).astype(int)),tuple(np.rint(end).astype(int)))
        if not okay:observations.append({'index':index,'status':'outside_image','samples':[]});continue
        start,end=np.array(clipped_a,float),np.array(clipped_b,float);length=np.linalg.norm(end-start)
        if length<64:observations.append({'index':index,'status':'too_short_visible_segment','samples':[]});continue
        direction=(end-start)/length;normal=np.array([-direction[1],direction[0]])
        locations=np.linspace(.04,.96,max(10,int(length/CFG['sample_spacing_px'])))
        centres=start+locations[:,None]*(end-start)
        mapxy=centres[:,None]+offsets[None,:,None]*normal
        inside=(mapxy[...,0]>=1)&(mapxy[...,0]<1918)&(mapxy[...,1]>=1)&(mapxy[...,1]<1078)
        samples=[]
        for frame_index,image in enumerate(images):
            brightness=image.min(axis=2).astype(np.float32)
            profiles=cv2.remap(brightness,mapxy[...,0].astype(np.float32),mapxy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            for location,profile in enumerate(profiles):
                if not inside[location].all():continue
                for offset,contrast,width in stripe_pairs(profile,offsets):
                    point=centres[location]+offset*normal
                    samples.append([*point,frame_index,location,contrast,width,offset])
        observations.append({'index':index,'status':'extracted','samples':samples,'start':start.tolist(),'end':end.tolist(),'locations':len(locations),'normal':normal.tolist()})
    return observations


def fit_line(observation):
    samples=np.asarray(observation['samples'],float)
    if len(samples)<16:return {**observation,'status':'insufficient_edge_pairs','inliers':[]}
    normal=np.array(observation['normal']);direction=np.array([-normal[1],normal[0]])
    length=np.linalg.norm(np.array(observation['end'])-observation['start'])
    rng=np.random.default_rng(CFG['seed']+observation['index']);best=None
    for _ in range(CFG['ransac_trials']):
        first,second=rng.integers(0,len(samples),2);delta=samples[second,:2]-samples[first,:2]
        if np.linalg.norm(delta)<length*.35:continue
        n=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        if abs(n@normal)<np.cos(np.deg2rad(CFG['line_angle_limit_degrees'])):continue
        line=np.r_[n,-n@samples[first,:2]];dist=np.abs(samples[:,:2]@line[:2]+line[2]);mask=dist<=CFG['line_inlier_distance_px']
        groups=samples[mask,2]*observation['locations']+samples[mask,3]
        score=(len(np.unique(groups)),float(np.mean(np.minimum(samples[mask,4],40))))
        if best is None or score>best[0]:best=(score,line,mask)
    if best is None:return {**observation,'status':'no_straight_line_consensus','inliers':[]}
    _,line,mask=best
    for _ in range(4):
        # One centre per frame/position; do not count multiple stripe proposals.
        candidates=np.flatnonzero(mask);groups=samples[candidates,2]*observation['locations']+samples[candidates,3]
        chosen=[]
        for group in np.unique(groups):
            ids=candidates[groups==group];dist=np.abs(samples[ids,:2]@line[:2]+line[2]);chosen.append(ids[np.argmin(dist)])
        subset=samples[chosen,:2];mid=subset.mean(0);_,_,vh=np.linalg.svd(subset-mid,full_matrices=False);n=vh[-1]
        line=np.r_[n,-n@mid];mask=np.abs(samples[:,:2]@line[:2]+line[2])<=CFG['line_inlier_distance_px']
    ids=np.asarray(chosen);s=samples[ids];coverage=float(np.ptp(s[:,3])/max(observation['locations']-1,1))
    status='ok'
    if len(np.unique(s[:,3]))<CFG['min_line_locations'] or len(np.unique(s[:,2]))<CFG['min_frame_count'] or coverage<CFG['min_coverage']:status='insufficient_spatial_or_temporal_support'
    return {**observation,'status':status,'line':line.tolist(),'inliers':ids.tolist(),'coverage':coverage,'frames':int(len(np.unique(s[:,2])))}


def intersect(lines,a,b):
    p=np.cross(lines[a],lines[b]);return p[:2]/p[2]


def fit_h(initial,points,labels):
    source=np.diag([1/5.485,1/11.885,1]);image=np.array([[1/1920,0,-.5],[0,1/1920,-1080/3840],[0,0,1.]])
    norm=image@initial@np.linalg.inv(source);norm/=norm[2,2]
    def unpack(p):
        h=np.r_[p,1.].reshape(3,3);return np.linalg.inv(image)@h@source
    def residual(p):
        ls=image_lines(unpack(p));return np.einsum('ij,ij->i',np.c_[points,np.ones(len(points))],ls[labels])
    result=least_squares(residual,norm.ravel()[:8],loss=CFG['loss'],f_scale=CFG['loss_scale_px'],max_nfev=400,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    h=unpack(result.x);h/=h[2,2]
    return h,{'success':bool(result.success),'nfev':result.nfev,'rank':int(np.linalg.matrix_rank(result.jac)),'cost':float(result.cost)}


def court_consensus(lines,initial):
    usable=[i for i,l in enumerate(lines) if l['status']=='ok']
    xids=[i for i in usable if i in (2,3,4,5,8)];yids=[i for i in usable if i in (0,1,6,7)]
    if len(xids)<2 or len(yids)<2:return None,[],{'status':'insufficient_line_families'}
    world_lines=np.cross(np.c_[POINTS[EDGES[:,0]],np.ones(9)],np.c_[POINTS[EDGES[:,1]],np.ones(9)])
    observed=np.array([l.get('line',[0,0,0]) for l in lines]);hypotheses=[initial]
    for xs in combinations(xids,2):
        for ys in combinations(yids,2):
            corners=[(x,y) for x in xs for y in ys]
            source=np.array([intersect(world_lines,x,y) for x,y in corners]);target=np.array([intersect(observed,x,y) for x,y in corners])
            if not np.isfinite(target).all():continue
            h,_=cv2.findHomography(source,target,method=0)
            if h is not None:hypotheses.append(h)
    best=None
    for h in hypotheses:
        ls=image_lines(h);medians=np.array([np.median(np.abs(np.c_[np.array(lines[i]['samples'])[lines[i]['inliers'],:2],np.ones(len(lines[i]['inliers']))]@ls[i])) for i in usable])
        supported=[i for i,d in zip(usable,medians) if d<=CFG['court_line_tolerance_px']]
        if min(len(set(supported)&set(xids)),len(set(supported)&set(yids)))<2:continue
        score=(len(supported),-float(np.minimum(medians,CFG['court_line_tolerance_px'])@np.minimum(medians,CFG['court_line_tolerance_px'])))
        if best is None or score>best[0]:best=(score,supported)
    if best is None or len(best[1])<CFG['min_court_lines']:return None,[],{'status':'insufficient_consistent_court_lines'}
    supported=best[1];points=np.concatenate([np.asarray(lines[i]['samples'])[lines[i]['inliers'],:2] for i in supported]);labels=np.concatenate([np.full(len(lines[i]['inliers']),i) for i in supported])
    # H hypotheses classify line evidence only. Optimization always starts at the
    # supplied previous reference, with no substitution on optimizer failure.
    h,diagnostic=fit_h(initial,points,labels)
    if not diagnostic['success'] or diagnostic['rank']<8:return None,supported,{**diagnostic,'status':'optimization_failed'}
    diagnostic['status']='ok' if len(supported)==9 else 'partial_line_evidence'
    return h,supported,diagnostic


def distribution(x):
    a=np.asarray(x,float)
    return {'n':len(a),'signed_median_px':float(np.median(a)) if len(a) else None,'abs_median_px':float(np.median(np.abs(a))) if len(a) else None,'abs_p95_px':float(np.percentile(np.abs(a),95)) if len(a) else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    reference_path=Path('knowledge/runs/run-court-acceptance-implementation-20261116/references.json');old=json.loads(reference_path.read_text());manifest=json.loads((BASE/'frames.json').read_text())
    result={'schema':'court_paint_reference_v1','human_reviewed':False,'approved_for_evaluation':False,'evaluation_status':'provisional_parent_authorized','source_reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),'config':CFG,'cameras':[]}
    for cam in old['cameras']:
        name=cam['camera'];frames=[r for r in manifest['records'] if r['camera']==name];images=[cv2.imread(str(BASE/r['path'])) for r in frames]
        initial=np.array(cam['matrix_court_metres_to_image_pixels']);lines=[fit_line(x) for x in extract(images,initial)]
        h,supported,diagnostic=court_consensus(lines,initial);report=[]
        for index,line in enumerate(lines):
            values=np.asarray(line['samples'],float);ids=line['inliers'];xy=values[ids,:2] if len(ids) else np.empty((0,2));residuals={}
            for mode,matrix in [('old',initial),('new',h)]:
                if matrix is None:continue
                l=image_lines(matrix)[index];residuals[mode]=distribution(np.c_[xy,np.ones(len(xy))]@l)
            report.append({'name':NAMES[index],'status':line['status'] if index in supported or line['status']!='ok' else 'inconsistent_with_court_geometry','supported_for_fit':index in supported,'edge_pairs':len(values),'line_inliers':len(ids),'coverage':line.get('coverage'), 'frames':line.get('frames'),'residuals':residuals})
        entry={'camera':name,'human_reviewed':False,'approved_for_evaluation':False,'evaluation_status':'provisional_parent_authorized','matrix_court_metres_to_image_pixels':h.tolist() if h is not None else None,'initial_matrix':initial.tolist(),'diagnostic':diagnostic,'lines':report}
        (args.output/f'{name}-evidence.json').write_text(json.dumps({'frames':frames,'lines':lines},allow_nan=False))
        result['cameras'].append(entry);(args.output/'references.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(name,diagnostic,'supported',[NAMES[i] for i in supported],flush=True)
        for l in report:print(l['name'],l['status'],l['line_inliers'],l['residuals'],flush=True)


if __name__=='__main__':main()
