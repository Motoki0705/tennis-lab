# P1 migration inventory

Inventory fixed before deletion in cycle 01. Hermes read-only investigation
session: `20260728_015809_e08f65`; log:
`logs/agent-auto/hermes/20260727T165807Z-3dgs-p1-inventory-3998545/`.

## Preserved alignment surface

- `src/synthetic_data_generation/scene_contract.py`
- `src/synthetic_data_generation/alignment/artifacts/`
- `src/synthetic_data_generation/alignment/components/`
- alignment stages and orchestrator under
  `src/synthetic_data_generation/scripts/`
- alignment Hydra configs

The scene-provider bundle/export/geometry bridge were alignment prerequisites,
but the top-level `provider/` namespace violated the requested deletion boundary.
They moved without behavioral fallbacks:

| Old path | Alignment-owned path |
|---|---|
| `provider/bundle.py` | `alignment/scene_provider/bundle.py` |
| `provider/export.py` | `alignment/scene_provider/export.py` |
| `provider/geometry_bridge.py` | `alignment/scene_provider/geometry_bridge.py` |
| `scripts/export_scene_provider.py` | `scripts/alignment/export_scene_provider.py` |
| `configs/export_scene_provider.yaml` | `configs/alignment/export_scene_provider.yaml` |
| `tests/unit/synthetic_data_generation/provider/` | `tests/unit/synthetic_data_generation/alignment/scene_provider/` |

## Confirmed removal

- `src/synthetic_data_generation/dataset/`
- `src/synthetic_data_generation/provider/`
- `src/synthetic_data_generation/rendering/`
- `src/synthetic_data_generation/code_identity.py`
- `scripts/publish_b00_ball_pilot.py`
- `scripts/publish_b00_full_scale_dataset.py`
- `scripts/render_b00_static_smoke.py`
- the three corresponding Hydra configs
- their dataset/rendering/code-identity unit tests

These files were dedicated to B00 publication, TrackNet/BLCS scene generation,
CPU fake rendering, or subprocess RGB overlay. Hermes and repository-wide `rg`
found no reverse dependency from alignment into those removed modules.

## Verification

- No directory named `dataset`, `provider`, or `rendering` remains under the
  synthetic-data source/test boundary.
- No Python import uses
  `src.synthetic_data_generation.{dataset,provider,rendering,code_identity}`.
- Export-first real validation produced and strictly loaded a 491-camera B00
  bundle at the artifact path recorded in `STATE.md`.
- Relevant unit/e2e suite: 66 passed.
