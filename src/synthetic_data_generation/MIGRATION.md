# Destructive scene-pipeline migration

Issue #695 replaces the former immutable artifact graph instead of wrapping it.
The production composition root is now only
`src.synthetic_data_generation.scripts.run`.

The inventory taken before the rewrite found these incompatible ownership paths:

| Former path/responsibility | Replacement |
|---|---|
| `scene_contract.py` and `ArtifactRef` | semantic `pipeline.scene.StandardScene` reader |
| `configuration.py` SHA-256 executable/source validation | #688 `RuntimePathRoots` and `PathResolver` plus strict scene config |
| `alignment/artifacts/*` fingerprint publishers | fixed `alignment/` stage publication |
| `alignment/scene_provider/*` immutable bundle | standard NHT `export/scene.json` boundary |
| `composition/*` commit/fingerprint contracts | independent `nht-render` subprocess boundary |
| domain artifact/plan publishers | fixed per-domain `dataset.json` and sample tree |
| `dataset/pipeline.py` / `dataset/execution.py` | typed scene DAG and domain stage functions |
| old alignment/dataset scripts and Hydra configs | one video-first scene CLI and one scene config |
| output refusal and content-addressed directories | descendant invalidation, staging, and canonical replacement |

No compatibility reader or legacy entry point is retained. Reproducibility comes
from the copied input video, resolved configuration, effective seed, typed DAG,
semantic validation, fixed workspace paths, stage logs, and current mutable
`run.json`; it does not depend on byte identity or Git state.
