# Multi-ball BLCS (`ball_tracking`)

`ball_tracking` is the multi-object companion to the single-ball `src/tasks/blcs`
task. It learns clip-local track slots from unordered ball candidate sets. The
single-ball data/model contracts are intentionally unchanged.

## Tensor contract

`B`, `V`, `T`, `D`, `Q`, and `P` denote batch, views, time, input candidates,
predicted query slots, and GT physical balls. Inputs use
`ball_uv (B,V,T,D,2)`, `ball_score/ball_candidate_mask/ball_visible
(B,V,T,D)`, `frame_mask (B,T)`, and `view_mask (B,V)`. Targets use
`position_3d (B,T,P,3)` and `ball_present (B,T,P)`. Outputs use
`position (B,T,Q,3)` and `presence_logits (B,T,Q)`.

Candidate indices are never coordinates or identities. Synthetic samples retain
`candidate_gt_index` only for validation/debugging; the model never receives it.
The fixed slot index is a clip-local ID, not a global ball identity.

## Model

Each candidate is encoded independently. At each time, learned slots and all
camera candidates are concatenated and processed by unified self-attention. The
spatial M-RoPE coordinates are `(time, camera, role)`:

- slot: `(t, 0, 0)`
- candidate from zero-based camera `v`: `(t, v + 1, 1)`

All candidates in one camera share coordinates, so permutation invariance is
preserved. Learned slot embeddings provide slot identity; no learned type
embedding is added. Slot features are then reshaped to `(B*Q,T,H)` for 1D time
RoPE attention. Spatial and temporal stages alternate.

Clip-level Hungarian matching supervises position and presence. Unmatched queries
receive no-ball targets. Smoothness and gravity priors remain task-local optional
ablations and are both off in the default configuration.

## Synthetic data and training

`data/synthetic.py` composes multiple independent trajectories with birth/death,
multi-view projection, dropout, duplicates, random and coherent false positives,
and independent camera-time shuffling. Fixed maximum tensor sizes plus explicit
masks implement variable `V/T/D/P` padding. Validation/test corruption is
deterministic by sample seed.

```bash
.venv/bin/python -m src.tasks.ball_tracking.scripts.train
.venv/bin/python -m src.tasks.ball_tracking.scripts.train model.role_rope_enabled=false
```

The default command is a compact reproducible baseline and runs test-split
prediction export after fitting.
