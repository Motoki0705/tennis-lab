"""Pin numerical evaluation source separately from a diagnostic-only revision."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE=Path(__file__).resolve().parent
EVALUATED='49f605aabbea65dc09c00c28a72a02b4a4ea59b2'
REGIONS='src/tasks/court_detection/inference/regions.py'


def sha(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def functions(source):
    tree=ast.parse(source)
    return {node.name:ast.get_source_segment(source,node) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}


def main():
    protocol=json.loads((BASE/'protocol.json').read_text())
    unchanged={}
    for filename,digest in protocol['files'].items():
        if filename==REGIONS:continue
        actual=sha(Path(filename).read_bytes());assert actual==digest,filename;unchanged[filename]=actual
    before=subprocess.check_output(['git','show',f'{EVALUATED}:{REGIONS}'],text=True)
    after=Path(REGIONS).read_text();old,new=functions(before),functions(after)
    names=('image_content_region','region_proposals','downstream_keypoints','predict_court_region','assess_region_prediction','assess_prediction_quality')
    protected={}
    for name in names:
        assert old[name]==new[name],name;protected[name]=sha(new[name])
    # Only the diagnostic construction between the decision and publication
    # changed. The complete proposal loop through its acceptance decision and
    # the ranking/publication tail remain byte-identical.
    def parts(source):
        start=source[:source.index('        qualifies =')]
        decision=source[source.index('        qualifies ='):].splitlines()[0]
        tail=source[source.index('        if qualifies:'):]
        return {'proposal_fit_gate':sha(start),'acceptance_decision':sha(decision),'ranking_publication':sha(tail)}
    assert parts(old['select_court_region'])==parts(new['select_court_region'])
    result={'evaluated_commit':EVALUATED,'numerical_and_reference_files_unchanged':unchanged,
        'region_original_sha256':sha(before),'region_diagnostic_revision_sha256':sha(after),
        'unchanged_region_functions':protected,'unchanged_selection_parts':parts(new['select_court_region']),
        'changed_scope':'geometry_diagnostics packaging and candidate diagnostic payload only; no estimator/config/threshold changes',
        'runtime_check':'api-parity.json: four cameras x two methods; decoded heads frozen'}
    (BASE/'source-equivalence.json').write_text(json.dumps(result,indent=2))
    print('numerical source / acceptance / ranking unchanged')


if __name__=='__main__':main()
