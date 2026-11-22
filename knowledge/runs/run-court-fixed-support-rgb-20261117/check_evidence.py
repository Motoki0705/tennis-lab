"""Confirm that paired solvers do not mutate their shared raster evidence."""
import hashlib
import json
from unittest.mock import patch

import cv2
import numpy as np

from evaluate import OUT, OLD, prepare, legacy, fixed, LineEvidence, LegacyEvidence, court_template_xy, COURT_HOMOGRAPHY_EDGES


def digest(evidence):
    return {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in
        [('distance',evidence.distance),('points',evidence.points),('weights',evidence.weights),
         ('tangents',evidence.tangents),('coherence',evidence.coherence),('tree_points',evidence.tree.data)]}


def main():
    cv2.setNumThreads(1);report=[]
    records=[r for r in json.loads((OUT/'manifest.json').read_text())['records']
        if r['dataset']=='meiji_4cam' and r['clip_id']=='video_000/clip_000']
    for record in records:
        raw=dict(np.load(OLD/record['old_raw_group']/record['key']/'raw_00.npz'))
        evidence=prepare(raw,OUT/'line-evidence'/record['key']/'evidence_00.npz');before=digest(evidence)
        observed=raw['raw_xy'].astype(float);observed[~raw['valid']]=np.nan
        after={}
        with patch.object(LineEvidence,'from_probability',return_value=evidence),patch.object(LegacyEvidence,'from_probability',return_value=evidence):
            for name,module in [('legacy',legacy),('fixed_support',fixed)]:
                module.estimate_hybrid_homography(court_template_xy().astype(float),observed,raw['scores'],raw['line_probability'],
                    edges=np.asarray(COURT_HOMOGRAPHY_EDGES),image_size_hw=tuple(raw['image_size_hw']))
                after[name]=digest(evidence);assert after[name]==before
        report.append({'key':record['key'],'before':before,'after':after})
    (OUT/'evidence-immutability.json').write_text(json.dumps({'cases':report,'all_unchanged':True},indent=2))


if __name__=='__main__':main()
