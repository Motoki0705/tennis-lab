"""Record the authorized metadata transition; preserve original numerical evidence."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
BUNDLE = Path('knowledge/runs/run-court-white-line-reference-20261116')
WHITE = BASE.parent / 'court-reference-white-lines-20261116'
COMMIT = '7b59c4730dfa934fb8e2323ae0553b96f737f412'
NOTE = 'https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6917118209'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    path = BUNDLE / 'reference/references.json'
    original_bytes = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'])
    original = json.loads(original_bytes)
    current = copy.deepcopy(original)
    approval = {'date': '2026-11-16', 'evidence': NOTE, 'reviewed_commit': COMMIT,
        'review_method': 'Human inspected original-resolution RGB crops and selected 承認する.',
        'accepted_limitations': ['far_baseline extrapolated from other painted lines', 'approximately 5px temporal change in cam1 and cam3'],
        'temporal_segmentation': False, 'mean_tolerance_px': 10, 'max_tolerance_px': 25,
        'false_accept_target': 0, 'correct_acceptance_target': .9,
        'instruction_turn': 'c101f6257cd34cbaa260dc537a3e2438'}
    for entry in [current, *current['cameras']]:
        entry.update(human_reviewed=True, approved_for_evaluation=True, evaluation_status='human_approved', approval=approval)
    before_geometry = {c['camera']: c['matrix_court_metres_to_image_pixels'] for c in original['cameras']}
    after_geometry = {c['camera']: c['matrix_court_metres_to_image_pixels'] for c in current['cameras']}
    assert canonical(before_geometry) == canonical(after_geometry)
    # Preserve exact preapproval bytes; its freeze.json and scripts remain historical.
    for root in [BUNDLE, WHITE]:
        (root / 'reference/preapproval-reference.json').write_bytes(original_bytes)
        (root / 'reference/references.json').write_text(json.dumps(current, indent=2, allow_nan=False))
    approved_bytes = path.read_bytes()
    record = {'approval': approval, 'preapproval_reference_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'approved_reference_sha256': hashlib.sha256(approved_bytes).hexdigest(),
        'geometry_sha256_before': hashlib.sha256(canonical(before_geometry)).hexdigest(),
        'geometry_sha256_after': hashlib.sha256(canonical(after_geometry)).hexdigest(),
        'geometry_equal': True, 'historical_freeze_targets': 'preapproval-reference.json (same bytes as prior references.json)',
        'historical_result_key': 'provisional now denotes this approved white-line reference; identifiers and numerical values retained'}
    for root in [BUNDLE, WHITE]:
        (root / 'reference/approval.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    (BASE / 'approval.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    for root in [BUNDLE, WHITE]:
        summary_path = root / 'evaluation/summary.json'
        old_summary = json.loads(subprocess.check_output(['git', 'show', f'{COMMIT}:{BUNDLE}/evaluation/summary.json']))
        summary = copy.deepcopy(old_summary)
        summary.update(evaluation_status='human_approved / 承認済み', approved_for_evaluation=True,
            reference_sha256=record['approved_reference_sha256'], reference_geometry_sha256=record['geometry_sha256_after'], approval_evidence=NOTE)
        assert summary['summary'] == old_summary['summary'] and summary['candidate_confusion'] == old_summary['candidate_confusion']
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__': main()
