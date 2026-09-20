# Meiji RGB Court DB bounded experiment

Findings and interpretation: [knowledge node](../../nodes/run-slcs-court-db-meiji-rgb-v1.md).

From the experiment worktree root, `bash knowledge/runs/run-slcs-court-db-meiji-rgb-v1/repro.sh NEW_OUTPUT_DIRECTORY` runs fixture checks and the fixed CPU-only recipe. The output directory must not exist. Uses `.venv/bin/python`, thread caps, shared read-only Meiji data/v9 caches; creates no GPU jobs. The DB is regenerated in memory from exact per-view configs/seeds, never stored in git.

- `probe.py`: extraction, retrieval/refinement, all-candidate records, freeze then manual evaluation.
- `summarize.py`: post-selection prior diagnostics, provenance and plots; no reselection.
- `results.json`: complete9view metrics, initial/refined H, all45candidate outcomes/configs.
- `metrics.json`: concise numerical summary.
- `prior_diagnostics.json`: nominal zero-roll camera discrepancy relative to v9.
- `provenance.json`: small input/source hashes, video stat and pre-existing cache identities; videos were not rehashed.
- `contact_sheet.jpg`: all9views, each showing baseline/initial/refined from top to bottom.
- `metrics.png`: same-RGB, temporal-RGB and manual measurements with distinct pixel scales.
- `run.log`: actual experiment output.

Full-resolution query/temporal RGB, extracted masks, per-view overlays, database metadata and pre-manual selection freeze are under the output directory recorded by the node. No video or full synthetic database is included here. `initial` denotes the retrieved camera associated with the selected refinement (or top HOG camera on total refinement failure), not necessarily HOG rank1. All candidate descriptor ranks/distances remain in results.
