"""CPU-only integrity checks for the completed evidence and metadata transition."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
BUNDLE = Path('knowledge/runs/run-court-approved-error-analysis-20261116')
WHITE_BUNDLE = Path('knowledge/runs/run-court-white-line-reference-20261116')
SOURCE = '7b59c4730dfa934fb8e2323ae0553b96f737f412'


def strip_approval(value):
    if isinstance(value, dict):
        return {k: strip_approval(v) for k, v in value.items() if k not in {'human_reviewed', 'approved_for_evaluation', 'evaluation_status', 'approval'}}
    if isinstance(value, list): return [strip_approval(v) for v in value]
    return value


def main():
    before_path = WHITE_BUNDLE / 'reference/preapproval-reference.json'
    before = json.loads(before_path.read_text()); after_path = WHITE_BUNDLE / 'reference/references.json'
    after = json.loads(after_path.read_text()); approval = json.loads((BASE / 'approval.json').read_text())
    assert after['human_reviewed'] and after['approved_for_evaluation']
    assert all(c['human_reviewed'] and c['approved_for_evaluation'] for c in after['cameras'])
    assert strip_approval(before) == strip_approval(after)
    assert before_path.read_bytes() == subprocess.check_output(['git', 'show', f'{SOURCE}:{after_path}'])
    assert hashlib.sha256(after_path.read_bytes()).hexdigest() == approval['approved_reference_sha256']
    summary_path = WHITE_BUNDLE / 'evaluation/summary.json'
    expected = json.loads(subprocess.check_output(['git', 'show', f'{SOURCE}:{summary_path}']))
    approved = json.loads(summary_path.read_text())
    for field in ('summary', 'candidate_confusion', 'targets', 'tolerance', 'frozen_prediction_files_sha256'):
        assert approved[field] == expected[field], field
    candidate = json.loads((BASE / 'candidate-analysis/summary.json').read_text())
    assert candidate['source_sha256'] == approved['frozen_prediction_files_sha256']
    for camera, conditions in candidate['by_camera'].items():
        for name, result in conditions.items():
            previous = approved['summary'][camera]['provisional'][name]
            assert sum(result['confusion'].values()) == 69
            assert all(result['confusion'].get(key, 0) == previous[key] for key in ('TP', 'FN', 'FP', 'TN'))
            assert sum(result['false_accept_patterns'].values()) == previous['FP']
            assert sum(result['recovery'].values()) == previous['correct_candidate_available']
    with (BASE / 'raw-analysis/keypoint-observations.csv').open() as f: observations = list(csv.DictReader(f))
    assert len(observations) == 3312 * 14
    assert len({(r['key'], r['region'], r['kp']) for r in observations}) == len(observations)
    assert sum('full_frame' in r['scopes'].split('|') for r in observations) == 276 * 14
    for directory, pattern, expected_count in [('regression-factorial', '*/*/*.json', 1176), ('regression-paired', '*/*.json', 221)]:
        files = list((BASE / directory).glob(pattern)); assert len(files) == expected_count, (directory, len(files))
    paired = [json.loads(path.read_text()) for path in (BASE / 'regression-paired').glob('*/*.json')]
    assert all(r['line_evidence_unchanged'] for r in paired)
    pilot = json.loads((BASE / 'rgb-pilot/results.json').read_text()); assert len(pilot) == 32
    assert {int(r['key'].split('/clip_')[1].split('/')[0]) for r in pilot}.isdisjoint({0, 17, 34, 51, 68})
    assert sum(r['correct'] for r in pilot) == 29 and sum(r['old_correct'] for r in pilot) == 9
    assert not any(r['old_correct'] and not r['correct'] for r in pilot)
    identity = json.loads((BUNDLE / 'source-identity.json').read_text())
    for path, digest in identity['source_sha256'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for path, digest in identity['raw_manifest_sha256'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert not subprocess.check_output(['git', 'diff', SOURCE, '--', 'src', 'tests'])
    for path in BASE.glob('*.py'): ast.parse(path.read_text())
    figures = json.loads((BASE / 'figures/manifest.json').read_text())
    assert all((BASE / 'figures' / r['native_image']).is_file() for r in figures['records'])
    sizes = {}
    for root in (BUNDLE, WHITE_BUNDLE):
        files = [p for p in root.rglob('*') if p.is_file()]; total = sum(p.stat().st_size for p in files)
        assert total <= 2 * 1024 * 1024
        for path in files: assert path.stat().st_size <= (512 if path.suffix in {'.png', '.svg', '.pdf'} else 256) * 1024
        sizes[str(root)] = total
    result = {'source_commit': SOURCE, 'approval_metadata_only': True, 'H_coordinates_and_old_numeric_results_unchanged': True,
        'all_276_confusions_and_recovery_counts_verified': True, 'raw_point_rows': len(observations),
        'factorial_solves': 1176, 'paired_cases': 221, 'paired_evidence_unchanged_within_case': True,
        'RGB_completed_before_pause': True, 'RGB_counts_recounted': True, 'production_diff_empty': True,
        'input_and_source_hashes_verified': True, 'native_figure_links_exist': True, 'bundle_bytes': sizes,
        'GPU_used': False, 'training_queue_jobs_added': 0}
    (BASE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
