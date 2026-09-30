"""Profile the pre-declared first frame of each view count for a GPU cost model."""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
from pathlib import Path

from audit import one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).parent
    sample = json.loads((bundle/'sample.json').read_text())
    settings = json.loads((bundle/'adaptive-settings.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    output = []
    for views in (0,1,2,3):
        row = next(r for r in sample['frames'] if r['observed_views']==views)
        profile = cProfile.Profile()
        profile.enable()
        result = one((row,sample['sources'][row['rally_id']],settings,str(args.output)))
        profile.disable()
        stats = pstats.Stats(profile)
        functions = []
        for (file,line,name), (ncalls, calls, own, cumulative, _) in stats.stats.items():
            if name in ('fit_component', '__init__', 'integrate', '_probes', 'log_prob') and 'probabilistic_triangulation' in file:
                functions.append(dict(file=file,line=line,function=name,ncalls=ncalls,calls=calls,own_seconds=own,cumulative_seconds=cumulative))
        output.append(dict(rally_id=row['rally_id'],frame=row['frame'],observed_views=views,seconds=result['seconds'],functions=functions))
    (args.output/'profile.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
