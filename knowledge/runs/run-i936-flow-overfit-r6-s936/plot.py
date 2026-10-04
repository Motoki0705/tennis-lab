"""Plot fixed-noise/time loss probes, separate from noisy training objectives."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
manifest=json.loads((args.run/'manifest.json').read_text())
rows=[json.loads(line) for line in (args.run/'updates.jsonl').read_text().splitlines()]
probes=[{'update':0,'fixed_probe':manifest['initial']}]+[r for r in rows if 'fixed_probe' in r]
fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
for ax,key,title in zip(axes.flat,['x0','reprojection','physics','event'],['x0 regression (normalized MSE)','Robust reprojection (nat)','Masked gravity residual','Event classification (BCE)']):
    values=[r['fixed_probe'][key] for r in probes]
    ax.plot([r['update'] for r in probes],values,color='#1764a0',linewidth=1.8)
    if min(values)>0:ax.set_yscale('log')
    ax.set_title(title);ax.set_xlabel('CPU update');ax.grid(alpha=.25)
fig.suptitle('Fixed 12-rally dataset: tiny overfit on two train prefixes\nSame noise/time probe; all four objectives active; no generalization claim',fontsize=11)
fig.savefig(args.output)
fig.savefig(args.output.with_suffix(".png"),dpi=150)
plt.close(fig)
