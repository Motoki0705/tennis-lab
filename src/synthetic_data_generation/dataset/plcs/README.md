# PLCS production dataset stage

This package owns the PLCS stage of the scene workspace. It loads every frame
of explicitly selected ACCAD/AMASS clips, evaluates the matching licensed
SMPL-H model, builds an SMPL-H surface Gaussian asset, applies Gaussian LBS at
every source frame, and rejects motion that is only a rigid root transform.

`timeline.py` creates a complete single- or multi-player compositor interval.
It retains track presence and source-frame mappings even when clips have
different start times or lengths. Generated cameras come only from the shared
config-owned camera profile, and target courts come only from the accepted
`MultiCourtLayout` plus the shared deterministic balanced assignment.

`composition.py` keeps the model, complete source poses, Gaussian shell, and
current bounded deformation batch on CUDA. `rendering/` converts only generated
metric camera poses at the public NHT boundary, uploads each validated static
background once, rasterizes and depth-composites on CUDA, and downloads only
visible foreground deltas. It never loads NHT checkpoint tensors.
`assembler.py` requires the exact global-frame × generated-camera sample set
before `dataset.json` is published.

The fixed output is:

```text
datasets/plcs/
├── dataset.json
├── backgrounds/<camera-id>/{rgb,alpha,depth-metric}.npy
├── chunks/chunk-000000/
│   ├── chunk.json
│   ├── foreground.npz
│   └── metadata.json
└── diagnostics/
    ├── motion-camera-court.json
    ├── performance.json
    └── summary.txt
```

The schema records motion source, source frame, gender, native FPS, target
court/candidate/transform, camera profile and sampled parameters, seed,
root/root-relative motion, frame equality, local articulation, and court
balance, source/model load counts, one NHT rig invocation, background-cache
misses, CUDA allocation, wall/CPU time, and compact/dense bytes. Frame reduction
is not a production option; chunk size affects only bounded execution and
storage batching. The canonical reader reconstructs logical full samples from
one shared background plus one sparse delta; no full-frame compatibility writer
or reader exists.
