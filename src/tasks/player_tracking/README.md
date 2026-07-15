# Multi-person PLCS (`player_tracking`)

`player_tracking` predicts clip-local player tracks from unordered multi-camera
2D pose detections. It is intentionally separate from the existing single-person
`src/tasks/plcs` contract.

## Tensor contract

`B`, `V`, `T`, `D`, `Q`, `P`, and `J` denote batch, views, time, detections,
query slots, GT persons, and joints. Model inputs are `human_kp
(B,V,T,D,J,2)`, `human_vis (B,V,T,D,J)`, `detection_mask/detection_score
(B,V,T,D)`, `bbox (B,V,T,D,4)`, `frame_mask (B,T)`, and `view_mask (B,V)`.
Targets are `position (B,T,P,3)`, `rotation (B,T,P,2)`, and
`person_present (B,T,P)`. Outputs are `position (B,T,Q,3)`, `rotation
(B,T,Q,2)`, and `presence_logits (B,T,Q)`.

Detection order has no semantic meaning. `detection_gt_index` is generated only
for data validation/debugging and never enters the model. Query index is a
clip-local track ID.

## Architecture and supervision

Each detection becomes one token. At every time step the sequence `[Q learned
slots + all V*D player tokens]` undergoes unified self-attention with M-RoPE
coordinates `(time,camera,role)`:

- slot: `(t,0,0)`
- player detection from zero-based camera `v`: `(t,v+1,1)`

There is no detection-index coordinate or learned type embedding. Slot identity
comes only from learned query embeddings. Each stage then applies 1D time-RoPE
self-attention to `(B*Q,T,H)` slots before returning them to the next spatial
stage.

Clip-level Hungarian matching uses position, rotation, and presence costs.
Matched tracks receive position/rotation/presence supervision and unmatched
queries receive no-person presence supervision. Track smoothness is an optional
post-matching regularizer and defaults to `0.0`.

## Synthetic baseline

The on-the-fly generator independently places several articulated motion tracks,
assigns birth/death intervals, projects them into each camera, simulates joint
noise, detector dropout and false positives, and independently shuffles every
camera-time detection set. Explicit masks implement variable `V/T/D/P` padding.

```bash
.venv/bin/python -m src.tasks.player_tracking.scripts.train
.venv/bin/python -m src.tasks.player_tracking.scripts.train model.role_rope_enabled=false
```
