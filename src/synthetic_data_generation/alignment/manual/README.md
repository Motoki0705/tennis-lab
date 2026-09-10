# Court Alignment Studio

Local web editor for correcting court placement on the measured, ground-projected
heatmap. Start it from the worktree containing this implementation, with an
explicit canonical scene path:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.edit_alignment \
  --scene-root /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B01 \
  --recover-ground-frame --port 8765
```

Open <http://localhost:8765>. The server binds only to loopback and requires its
session token for POST requests. No GPU inference or automatic realignment runs.

## Editing

- Drag each court to translate it; drag the round handle to rotate it. Numeric
  U/V/yaw inputs and arrow keys provide fine adjustment. Shift+arrows move farther.
- Each of the four square corner handles resizes **all** regulation templates.
  The selected court keeps the diagonally opposite corner fixed; its centre moves
  accordingly. Other courts retain their centres. Numeric common-scale edits
  retain every centre. Court dimensions and aspect ratio remain fixed in metres;
  the correction changes the scene-to-metre calibration. Ground tilt stays fixed.
- Add, duplicate, or delete any number of courts; select the primary court explicitly.
  Zero courts may be saved as a draft, but formal application requires at least one.
- Pan by dragging empty space (or Shift+drag), and zoom with the wheel. Undo/redo
  restores complete layouts, including court inventory and common scale.
- The court-line opacity slider (0–100%, initially 55%) reveals the heatmap
  underneath. It also affects the camera overlay; editing handles remain visible.
- Camera selection shows a live projection onto the original captured image.
  The distance diagnostics describe measured support, not a mandatory gate.
- **下書き保存** persists the draft without changing the canonical alignment.
  **確定して適用** asks the human to confirm the complete layout, then publishes it.

## Source evidence and coordinate authority

An existing `alignment/` containing `alignment.json`, `ground-line-map.npz`, and
validated `line-heatmaps/` is required. Incorrect automatic placements can be
replaced completely. An automatic failure that produced no heatmaps cannot be
edited until those observations are available.

The editor imports evidence independently of automatic acceptance. Its initial
reference alignment is parsed structurally, not reported as having passed the
current optimizer or acceptance gates. Plane, camera partitions, similarity and
paired UV/3D observations must agree. For older artifacts such as B01, an absent
plane is **only** recovered with `--recover-ground-frame`: a rank-three affine fit
must reproduce all paired observations within `1e-7` metres and provide an
orthonormal, right-handed frame. The import method and maximum residual are
recorded. There is no guessed ground plane or inference fallback.

Editor U/V coordinates always use the immutable source heatmap frame. If the
shared scale is `s`, a regulation metre occupies `s` source units, and the new
metric scene is the original metric scene divided by `s`. Repeated applications
continue from that original frame, so scale corrections never compound.

## Publication and downstream use

The human-confirmed schema is explicit:

- `alignment.json`: `human_confirmed_multi_court_alignment_v1`, with proper rigid
  court transforms, reciprocal uniform metric/NHT transforms and complete layout.
- `manual-confirmation.json`: confirmed placements, timestamp, source revision and
  evidence digests. Every included court has explicit human confirmation.
- `ground-line-map.npz`: `manual_alignment_source_archive_v1`, retaining original
  reference geometry, verified plane, import provenance and reconstruction identity.
- `line-heatmaps/`: unchanged, validated numeric/PNG observations in the **source**
  coordinate frame. They do not claim to use the corrected metric scale.
- `diagnostics/manual-metrics.json`: for each court and camera partition, 64
  endpoint-inclusive samples per regulation segment are compared to the nearest
  measured projected point. Distances use corrected metres. Actual threshold
  failures are retained; a human decision never fabricates passing checks.

Human inspection can use all camera views, so holdout values are descriptive and
are no longer an independent acceptance test. `validate_alignment_outputs` checks
the reconstruction binding and recomputes the complete manual geometry and
measurements. `load_accepted_layout` and Court/BLCS/PLCS readers consume this same
canonical owner. Automatic optimizer-trace publication is not applicable to this
manual source archive; the editor provides the captured-camera visual check.

Each apply verifies that the source revision is unchanged and acquires the same
scene writer lock as `ScenePipelineRunner`. It validates the replacement before
mutation, invalidates downstream stage records, and atomically exchanges the
complete alignment directory. Old alignment, `run.json`, dataset/report owners
and any publication bundle are retained under
`alignment-editor/history/<revision-id>/`. Exceptions roll back both owners and
manifest. A process interruption leaves a non-completed manifest rather than
claiming a valid completed pipeline; the retained history provides recovery data.
The service refuses to apply while a stage is running.

After confirmation, rerun the desired dataset suffix, for example
`request.from_stage=court_dataset request.targets='[court]' profile=b01` with the
normal scene-pipeline command. Starting from `alignment` explicitly reruns the
automatic optimizer and replaces the human-confirmed owner.

## Implementation and validation

`models.py` owns request validation, `source.py` imports and binds evidence,
`geometry.py` constructs metric geometry and measures support, `artifacts.py`
serializes and validates the manual schema, `service.py` handles drafts and
publication, and `web.py` serves the API and local SVG/JavaScript editor.
Static assets ship with the Python package; no frontend build is needed.

The focused regression tests are in
`tests/unit/synthetic_data_generation/alignment/test_manual.py`. They cover
regulation dimensions, source/NHT binding, acceptance override without falsified
metrics, draft reload, revision conflicts, shared locks, canonical publication,
rollback before and after exchange, reconstruction changes, and the local API.

The pure corner-resize geometry tests run with
`node --test tests/unit/synthetic_data_generation/alignment/manual/resize.test.mjs`.
They verify all four anchors across rotated courts, grab offsets, aspect ratio,
and prevention of scale inversion.
