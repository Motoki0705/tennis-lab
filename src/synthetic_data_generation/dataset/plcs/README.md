# PLCS production dataset stage

This package owns the PLCS stage of the scene workspace. It loads every frame
of explicitly selected ACCAD/AMASS clips, evaluates the matching licensed
SMPL-H model, builds an SMPL-H surface Gaussian asset, applies Gaussian LBS at
every source frame, and rejects motion that is only a rigid root transform.

AMASS/ACCAD motion, SMPL-H LBS output, and court placement share a
right-handed, metre, Z-up coordinate contract. SMPL-H LBS already consumes
`global_orient`; PLCS adds only configured yaw about court +Z. Each track is
grounded once from the frame-zero posed full SMPL-H surface minimum, and later
frames preserve the source vertical motion without re-grounding.

`production.py` defines the finite `single_object` and
`multi_object_global_timeline` production modes. `single_object.yaml` selects
one real ACCAD category at source frame zero; `production.yaml` remains the
running/walking/general multi-object B00 authority. Both use the same compact
v5 schema, CUDA articulated execution, generated cameras, and court binding.

`timeline.py` creates the complete compositor interval for every explicitly
configured logical scene. Every logical scene retains the same full configured
source inventory, track presence, and source-frame mappings even when
multi-object clips have different start times or lengths. Logical scenes,
never individual objects or frames, receive one target-court binding.
The complete scene inventory must use every accepted court and keep global and
per-split scene-count spread at most one.

`execution.py` is the constructor-injected numerical boundary. Its production
implementation keeps the model, complete source poses, Gaussian shell, and
current bounded deformation batch on CUDA. The only non-CUDA implementation
marker is the explicit `test-cpu-oracle` dependency used with a non-production
test budget; there is no runtime device selection or fallback. `rendering/`
converts only generated metric camera poses at the public NHT boundary, uploads
each validated static background once, rasterizes and depth-composites on CUDA,
and downloads only visible foreground deltas. It never loads NHT checkpoint
tensors.
`assembler.py` requires the exact global-frame × generated-camera sample set
before `dataset.json` is published.

The renderer evaluates each bounded SMPL-H source batch once and reuses that
validated deformation across the simultaneously produced logical courts. Its
per-camera hot path keeps immutable camera matrices on CUDA, bins projected
Gaussians into tiles with an exact stable vectorized implementation, and uses
two consolidated compact device-to-host payloads per sample. These are only
execution optimizations: every scene, frame, camera, label, court transform,
and sparse output remains independently assembled and validated.

The fixed output is:

```text
datasets/plcs/
├── dataset.json
├── backgrounds/<camera-id>/{rgb,alpha,depth-metric}.npy
├── scenes/<logical-scene-id>/chunks/chunk-000000/
│   ├── chunk.json
│   ├── foreground.npz
│   └── metadata.json
└── diagnostics/
    ├── motion-camera-court.json
    ├── performance.json
    └── summary.txt
```

The aggregate schema records every logical scene, its complete local `0..T-1`
timeline, split, target court/candidate/transform, cameras, and exact motion
inventory. Diagnostics validate every accepted court, aggregate and per-split
balance, aggregate source/global frame counts, source/model load counts, one
batched NHT background invocation, background-cache misses, CUDA allocation,
wall/CPU time, and compact/dense bytes. Frame reduction is not a production
option; chunk size affects only bounded execution and storage batching. The
canonical reader reconstructs a logical full sample by logical scene ID, local
frame index, and camera ID from one shared background plus one sparse delta; no
full-frame compatibility writer or reader exists.

`supervision.npz` stores only `position` in the selected dimensionless PLCS
normalization. `position_court_m`, `human_kp_3d`, and `canonical_pose_3d` remain
physical metres and are never rescaled. The single mathematical authority for
`position_norm = position_court_m / scale_xyz` and the `v1`/`v2` scales is
[`src/utils/schema/court_normalization.py`](../../../utils/schema/court_normalization.py).
The resolved contract (schema version, normalization version, `scale_xyz`,
position unit `m`, and velocity unit `m/s`) is written identically into the
dataset manifest root metadata and every logical-scene record. Assembly and
the canonical reader reject malformed, mixed, or mismatched metadata and
validate the persisted `position` → `position_court_m` round trip within
`1e-5 m`. A wholly metadata-free compact artifact is accepted only when the
reader/validator receives an explicit legacy `v1` runtime contract; no version
is inferred from array values.
