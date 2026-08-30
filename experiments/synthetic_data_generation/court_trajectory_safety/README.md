# Court trajectory safety benchmark

This directory is the compact tracked evidence boundary for the Attempt-2
repair of Issue #823. The CPU-generated pilot manifest freezes the required
constructor, family, profile, target, anchor, frame-share, and equal-budget
inventories. The fresh GPU pilot, blind annotations, adjudication, final V4
dataset, and complete summary remain pending until they are produced under the
new roots below.

The historical `B00-pilot`, `B00-semantic-phase-pilot`,
`issue-823-blind`, `issue-823-semantic-phase-blind`, `B00-final.staging`, and
`B00-semantic-phase-final` roots belong to superseded candidate inventories.
Their labels, quality decision, contact sheets, and final metrics are stale and
must not be copied into this attempt.

## Frozen pilot protocol

- Scene: B00 public `StandardSceneExport`.
- Manifest: opaque views cover every stratum named in `frozen-config.json`.
- IDs: randomized `review-<16 lowercase hex>` identifiers. Reviewers receive
  only opaque IDs and RGB previews, never feature values, strata,
  baseline/candidate disposition, or another reviewer's labels.
- Split: trajectory-group-disjoint calibration/held-out assignment with seed
  823. Held-out groups are never used to tune a feature threshold or geometry.
- Label: `artifact_heavy=true` only when reconstruction defects materially
  corrupt court/background geometry. Ordinary blur, exposure, lighting, or
  compression is not artifact-heavy.
- Review: two complete independent annotations are mandatory. A third reviewer
  adjudicates only their exact disagreement set.

Each private reviewer record uses
`court_trajectory_blind_annotation_v1`, binds the frozen manifest hash and
reviewer ID, and contains one `{opaque_id, artifact_heavy, note}` record for
every view. No annotation from another manifest is valid.

## Attempt-2 status

`pilot-manifest.json` is regenerated from the public B00 export without GPU
rendering. It is not valid complete evidence until a new 128-view pilot is
rendered, independently reviewed, frozen, and followed by a new final dataset.
`frozen-config.json`, `summary.json`, and `report.md` therefore remain explicitly
pending and contain no reused Attempt-1 observation outcome.

## Integrity and immutability

`frozen-config.json` contains the hash-only observation lock for the pilot
manifest, public features, blind-review manifest, both independent reviewers,
disagreement-only adjudication, and RGB preview inventory. `pilot-manifest.json`
is the tracked randomized protocol and camera inventory. The canonical verifier
recomputes all integrity relationships available from these tracked files and
requires the final bulk dataset and compact-evidence SHA-256 bindings without
claiming that ignored bulk data is present in a fresh checkout.

The Attempt-2 roots are:

- pilot: `outputs/court_trajectory_safety/B00-required-coverage-pilot`
- annotations: `outputs/court_trajectory_safety/issue-823-required-coverage-blind`
- final dataset: `outputs/court_trajectory_safety/B00-required-coverage-final`

Each render or finalizer must target a new, absent root. Never overwrite or
reuse an existing pilot, annotation, audit, staging, or final root. A later
iteration must choose another uniquely named root and regenerate its manifest
and blind labels.

After all Attempt-2 evidence exists, validate it from a fresh checkout with the
following exact read-only B00 command. It performs no render, `mkdir`, contact
sheet generation, or file write:

```bash
cd /home/kamimura/projects/tennis-lab
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.synthetic_data_generation.scripts.evaluate_court_trajectory_safety action=validate_complete_evidence scene_path=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B00/reconstruction/export/scene.json alignment_path=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B00/alignment/alignment.json pilot_manifest_path=/home/kamimura/projects/tennis-lab/experiments/synthetic_data_generation/court_trajectory_safety/pilot-manifest.json pilot_output_root=/home/kamimura/projects/tennis-lab/outputs/court_trajectory_safety/B00-required-coverage-pilot final_evidence_root=/home/kamimura/projects/tennis-lab/outputs/court_trajectory_safety/B00-required-coverage-final annotation_root=/home/kamimura/projects/tennis-lab/outputs/court_trajectory_safety/issue-823-required-coverage-blind audit_output_root=/home/kamimura/projects/tennis-lab/outputs/court_trajectory_safety/issue-823-required-coverage-blind/evidence source_video_path=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/raw/B00.mp4 frozen_config_path=/home/kamimura/projects/tennis-lab/experiments/synthetic_data_generation/court_trajectory_safety/frozen-config.json environment.CUDA_VISIBLE_DEVICES=""
```
