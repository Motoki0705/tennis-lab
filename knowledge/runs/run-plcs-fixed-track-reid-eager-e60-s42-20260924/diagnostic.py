"""Recompute test correspondence metrics and stratify errors from saved embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.tasks.plcs.model_io.track_matching import match_track_embeddings


def counts(pred, truth, mask):
    tp = int((pred & truth & mask).sum())
    fp = int((pred & ~truth & mask).sum())
    fn = int((~pred & truth & mask).sum())
    tn = int((~pred & ~truth & mask).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    metrics=json.loads((args.run_root/'predictions/metrics.json').read_text())
    archive=np.load(args.run_root/'predictions/pred_test.npz',allow_pickle=False)
    z=archive['track_embedding'];gt=archive['track_person_id'];valid=archive['track_valid'] & (gt>=0)
    ids=archive['slot_global_ids']; b,v,p,d=z.shape;n=v*p
    threshold=float(metrics['cosine_threshold'])
    target=gt.reshape(b,n);available=valid.reshape(b,n);assigned=ids.reshape(b,n)
    cameras=np.repeat(np.arange(v),p)
    mask=available[:,:,None] & available[:,None,:] & (cameras[:,None] < cameras[None,:])
    same=target[:,:,None]==target[:,None,:]
    predicted=(assigned[:,:,None]==assigned[:,None,:]) & (assigned[:,:,None]>=0) & (assigned[:,None,:]>=0)
    scores=z.reshape(b,n,d).astype(np.float64) @ z.reshape(b,n,d).astype(np.float64).transpose(0,2,1)
    report={'checkpoint_selection':'minimum validation loss; stored checkpoint and threshold unchanged',
        'cosine_threshold':threshold,'scene_count':b,'default_matching':counts(predicted,same,mask),
        'fp64_cosine_pair_diagnostic':counts(scores>threshold,same,mask)}
    side=np.repeat(archive['side_target'],p,axis=1)
    opposite=side[:,:,None]!=side[:,None,:]
    report['same_side_matching']=counts(predicted,same,mask & ~opposite)
    report['opposite_side_matching']=counts(predicted,same,mask & opposite)
    accepted=(assigned>=0) | ~available
    exact=((predicted==same)|~mask).reshape(b,-1).all(-1) & accepted.all(-1)
    eligible=available.any(-1)
    report['exact_group_accuracy']=float(exact[eligible].mean())
    report['track_acceptance_recall']=float(((assigned>=0)&available).sum()/available.sum())
    people=np.array([len(np.unique(row[keep])) for row,keep in zip(target,available)])
    report['by_observed_people']={str(k):dict(scenes=int((people==k).sum()),exact_group_accuracy=float(exact[people==k].mean())) for k in np.unique(people)}
    # Diagnostic only: accept every upstream track, without selecting any new threshold.
    no_head=torch.stack([match_track_embeddings(torch.from_numpy(row),torch.from_numpy(keep),threshold=threshold) for row,keep in zip(z,archive['track_valid'])]).numpy().reshape(b,n)
    no_head_same=(no_head[:,:,None]==no_head[:,None,:]) & (no_head[:,:,None]>=0) & (no_head[:,None,:]>=0)
    report['without_targetness_head_diagnostic']=counts(no_head_same,same,mask)
    report['without_targetness_head_exact_group_accuracy_diagnostic']=float((((no_head_same==same)|~mask).reshape(b,-1).all(-1) & ((no_head>=0)|~available).all(-1))[eligible].mean())
    names=(args.data_root/'test.txt').read_text().splitlines()
    failures=[]
    for i in range(b):
        if exact[i]:continue
        missed=int((same[i]&~predicted[i]&mask[i]).sum()); wrong=int((~same[i]&predicted[i]&mask[i]).sum())
        failures.append(dict(test_index=int(archive['sample_index'][i]),scene=names[int(archive['sample_index'][i])],
            observed_people=int(people[i]),wrong_pairs=wrong,missed_pairs=missed,
            rejected_tracks=int(((assigned[i]<0)&available[i]).sum()),
            gt_ids=gt[i].tolist(),predicted_ids=ids[i].tolist()))
    report['failed_scenes']=sorted(failures,key=lambda row:-(row['wrong_pairs']+row['missed_pairs']))
    (args.output/'test_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label,selection in [('same person',mask&same),('different people',mask&~same)]:
        axes[0].hist(scores[selection],bins=np.linspace(-1,1,61),density=True,alpha=.55,label=label)
    axes[0].axvline(threshold,color='black',linestyle='--',label='validation threshold')
    axes[0].set(xlabel='Cosine similarity (FP64 diagnostic)',ylabel='Density');axes[0].legend()
    groups=report['by_observed_people'];axes[1].bar(list(groups),[x['exact_group_accuracy'] for x in groups.values()])
    axes[1].set(xlabel='Observed ground-truth people',ylabel='Exact group accuracy',ylim=(0,1))
    for x,row in enumerate(groups.values()):axes[1].text(x,row['exact_group_accuracy']+.02,f"n={row['scenes']}",ha='center')
    fig.tight_layout();fig.savefig(args.output/'test_diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps({k:value for k,value in report.items() if k!='failed_scenes'},indent=2))


if __name__=='__main__':main()
