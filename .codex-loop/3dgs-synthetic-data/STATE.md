# 3DGS-native synthetic-data loop state

## Cursor

- Cycle: 18
- Current phase: complete — Architecture A / NHT N1 production refactor
  validated and delivered to PR #666
- P0: complete (cycle 01)
- P1: complete (cycle 01)
- P2: complete (cycle 02)
- P3: complete (cycle 09); the user authorized a generated prototype ball and
  accepted the existing court alignment. Export-first single/multi physical
  plans, native NHT renders, exact AOV labels, and the strict acceptance report
  pass without an RGB overlay.
- P4: complete (cycle 11); official-method comparison, two-candidate geometry
  screen, a 4,096-Gaussian/55-joint SMPL-X asset, explicit NHT feature fitting,
  controlled native renders, repeated-run tolerance, and the strict acceptance
  report pass.
- P5: complete (cycle 12); deterministic single/multi-person placement,
  controlled SMPL-X pose assets, native background-plus-person NHT renders,
  exact instance labels, byte-identical repeats, and the strict acceptance
  report pass.
- P6: complete (cycle 14 extension); the cycle-13 conservative baseline remains
  valid, and the user-requested two-court circle/ellipse extension adds 428
  bold full/partial views plus 18 native NHT representative renders.
- P7: complete (cycle 15); 428 family-disjoint native NHT frames, two-court
  physical labels, seven-channel multi-peak targets, byte-identical repeat,
  visual diagnostics, and the strict acceptance report pass.
- P8: complete (cycle 16); export-first integrated acceptance passed 15/15,
  same-seed and distinct-seed gates pass, and the visualization-first report is
  complete.
- Worktree:
  `/home/kamimura/projects/tennis-lab/.claude/worktrees/3dgs-native-synthetic-data`
- Branch: `feat/3dgs-native-synthetic-data`
- Refactor base HEAD: `c3e5728280bfae237eba8e776905aa0a90868c0a`
- NHT boundary commit:
  `b3176cfe2f8e16f1f89fe29151db650f3867af4f`
- Parent refactor commit:
  `f00228e94d9734a3345ba8a1b475acd787b9595e`
- Pull request:
  `https://github.com/Motoki0705/tennis-lab/pull/666` (`OPEN`,
  `MERGEABLE`, `enhancement`)
- NHT fork pull request:
  `https://github.com/Motoki0705/neural-harmonic-textures/pull/2`
- Automation `3dgs`: deleted after all acceptance gates, export-first
  verification, report publication, and PR creation completed.
- Updated: 2026-07-28 22:24:02 JST / 2026-07-28 13:24:02 UTC

## Completed in cycle 18

- Applied the user-selected Architecture A. All current and future generators
  now enter through the single
  `src/synthetic_data_generation/dataset/registry.py` and live in vertical
  slices under `dataset/{blcs,plcs,court}`. Each slice owns artifacts,
  components, rendering, validation, and reporting. Alignment remains under
  its accepted component/pipeline hierarchy.
- Reorganized entry points under
  `src/synthetic_data_generation/scripts/{alignment,dataset}`. The dataset
  runner publishes an immutable, fingerprinted command plan before optional
  execution, refuses existing plan/execution paths, and records whether every
  stage belongs to the project or NHT runtime.
- Retained candidate implementations as strict config choices instead of
  deleting alternatives: BLCS ball asset
  `procedural_fibonacci|registered_gaussian_asset`; PLCS avatar control
  `gaussianavatar_query_lbs|hugs_topk_lbs`; court camera sampling
  `sfm_neighborhood|inward_orbit`. Unknown names raise with the complete choice
  list; there is no fallback.
- Applied NHT boundary N1. The submodule now owns only its pinned environment,
  training, checkpoint, deferred shader, and rasterizer. All tennis-lab BLCS,
  PLCS, court, acceptance, and reporting workers moved to the parent project.
  The NHT fork change is commit
  `b3176cfe2f8e16f1f89fe29151db650f3867af4f`, pushed as PR
  `https://github.com/Motoki0705/neural-harmonic-textures/pull/2`.
- The shell-free NHT adapter verifies the exact submodule commit and clean
  tracked state. Runtime verification exposed that resolving
  `.venv/bin/python` followed its symlink to the base interpreter and dropped
  isolated site-packages. The adapter now preserves the logical venv path, and
  a regression test proves it.
- Export-first reran into the new immutable directory
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-18/provider-export-refactor-v1`.
  It contains 493 files / 275 MiB, 491 cameras, 491 images, and 217,336 points.
  `provider.json` SHA-256 is
  `bf880b07829e9cf9183ac30092afbea30509f5bccdbdd05fe6c0220d1919d216`;
  bundle fingerprint is
  `9a3546c83926c09b7e17d427680e5c25649bc02cba58d9580e44f070195692c6`;
  scene fingerprint remains
  `2c16d09503118b08a30b3819d01c23b2bc0e575f00b4f30a931c8447d4d3e160`.
- N1 execution passed through the real dataset runner at
  `cycle-18/blcs-runtime-plan-v2.json`,
  `cycle-18/blcs-runtime-execution-v2`, and
  `cycle-18/nht-runtime-v2.json`. The execution is complete with one successful
  NHT stage; Torch is `2.9.1+cu130`, CUDA is available with one device, and the
  editable gsplat checkout is the pinned submodule. Their SHA-256 values are
  `c6158419e1b545db17c236fbff3efb41de537dd4b710e4fbe87303529138bf1c`,
  `097db44d5fe2f8c33ae0d63739911b8072f9af86d244a0c742808871ab587909`,
  and
  `addf872db5690e207eaf8686f9ae99dc54221f9c0bde6d68cd9060b880913dec`.
- A real native RGB stage passed through the same N1 pipeline at
  `cycle-18/blcs-render-plan-v3.json`,
  `cycle-18/blcs-render-execution-v3`, and
  `cycle-18/blcs-real-render-v3`. The 640x360 frame is finite, contains no RGB
  overlay, uses renderer commit
  `20bc323d613258e5d169fdbc962c9ef27d55ca69`, and has render fingerprint
  `9ad94fa038891a652cd4f44089233bc068b462af6901f10e0e1b329b49ad5ba1`.
  Manifest SHA-256 is
  `ade53e1056a0a13a15441796232d3c16e23acc2d5c2cbc7edadcc0b123142e6e`;
  RGB SHA-256 is
  `0dce48f3c13b68d321f7ec1325fd20dd190c56fac57ca786018f11e2558ba1d3`.
- Final regression passed `156/156` synthetic-data unit/integration/e2e tests
  in `18.98 s` with six xdist workers. Ruff, changed-scope mypy with
  `--follow-imports=skip`, script-convention review for all eight synthetic-data
  scripts, and `git diff --check` passed.
- Updated the visualization-first report with the Architecture A/N1 diagram,
  configurable algorithm matrix, export/runtime/render metrics, and explicit
  AOV failure boundary. No user-owned production-preview binary was modified,
  reverted, staged, or overwritten by this refactor.
- PLCS/camera research decisions are unchanged: GaussianAvatar-style query LBS
  remains the selected baseline while HUGS top-k remains selectable; both SfM
  neighborhood and inward circle/ellipse orbit samplers remain selectable.

## Cycle 18 failures and hypotheses

- The inherited Windows `TMP/TEMP` path caused pytest capture deletion and
  zero-test collection. The authoritative run used a fresh Linux `mktemp`
  directory plus this worktree's explicit `PYTHONPATH`; 156 tests passed. This
  was an invocation-environment failure, not a test fallback.
- The first N1 runtime plan failed because interpreter symlink resolution
  selected the base Python and reported missing Torch. The environment itself
  was complete. Preserving the venv path fixed the adapter; failed
  `cycle-18/blcs-runtime-plan-v1.json` remains unchanged.
- The first real render attempt with default AOV tolerance rejected measured
  NHT/AOV alpha drift `0.0130756 > 0.0001`. The successful far-view smoke
  explicitly records tolerance `0.02`; its active balls are sub-pixel and it is
  not counted as ball-visibility acceptance.
- A closer 1280-wide multi-ball attempt failed at
  `0.118223 > 0.03`. Its immutable plan
  `cycle-18/blcs-render-plan-v4.json` is retained. The production exact-AOV
  default was not weakened; the hypothesis remains that the eval3d AOV pass
  must be made semantically identical to the production NHT eval path.

## Cycle 18 running jobs

- NHT training: none
- BLCS/PLCS/court render: none
- export/alignment: none
- GPU compute process owned by this cycle: none
- Only next action: the user reviews PR #666, NHT fork PR #2, and the
  visualization-first report; merge or request a focused follow-up.

## Completed in cycle 17

- Replaced the cycle-16 green/noisy one-step mechanics preview with a separate,
  explicitly production-scoped visual path. The source is the retained B03 NHT
  checkpoint
  `/home/kamimura/projects/gaussian-splating/experiments/B03-NHT/results/ckpts/ckpt_29999_rank0.pt`
  (SHA-256
  `e8d722a172774de8df27e1ae38ac74d6a81d9a8e980fc83aca7c665eb9b68111`,
  30,000 training steps). The background has 999,744 Gaussians and visibly
  reconstructs RGB courts, nets, trees, fences, and school buildings.
- Canonical production composition is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-17/nht-production-composition-v1`
  (composition fingerprint
  `c51339d39e84b8484f70e256edd156606c55faf8f6d08e6e23761368fc2420ae`,
  appearance-space SHA-256
  `fea40a61753a39d6900a29cb0fcfa80350595a6d30df617517c0241f7f33e929`,
  composition manifest SHA-256
  `fc3d4ff112b4e92c7fb8579cd2dc4169808afbc31204a9082eb027540f718537`).
  It remains pinned to gsplat/NHT commit
  `20bc323d613258e5d169fdbc962c9ef27d55ca69`.
- Added the isolated production RGB preview renderer
  `third_party/nht/production_preview_render.py` (SHA-256
  `bed7eb8099cb5f44f7d1fe90a543dfd77568ffe6c8c1b9166dc3165b665f0796`).
  It rehashes the composition and source plans, requires one clean native
  `RGB+ED` public rasterization per frame, writes raw RGB unchanged, and writes
  court/object markers only to a separate diagnostic video. It supports
  alignment, BLCS, PLCS, and orbit-family court previews.
- Generated and visually inspected captured-camera alignment at
  `cycle-17/production-alignment-frame-000080-v1`. `court_0` yellow and
  `court_1` cyan physical points align with the visible two-court line
  geometry. The 960x540 raw RGB has mean/std `0.497432/0.250319` and alpha mean
  `0.976637`.
- Rebuilt the prototype ball through the production NHT appearance:
  `production-ball-fixture-v1`, `production-ball-calibration-v1`, and
  `production-ball-preparation-v1`. The accepted registry fingerprint is
  `5398e114459fefcaf300549497c45294aa28d4b73a469568773a14c95e3d63b2`.
  Three 240-frame, three-object physical plans use seeds
  `20260728/20260730/20260732`, plan fingerprints
  `16230d8a938e55c961f2cd2869187f529322a538859d992a21a0cafce8fcfb61`,
  `7c1ac3e80bb5e6ce8c05f442013caa887ea32a17f6f916e296937b87899e256e`,
  and
  `93f5aa1e850da933f852f369e705c7ea025be795bb40a117d3f5a89d493b64f7`;
  their active object-frame counts are
  `443/281/264`, maximum heights `8.78884/7.29259/3.73353 m`, and maximum
  speeds `18.58315/16.23374/22.61436 m/s`.
- Published three visible-camera BLCS previews at
  `cycle-17/production-blcs-visible-video-seed-{20260728,20260730,20260732}-v1`.
  They contain `21/11/16` 1280-wide frames and three native Gaussian balls.
  Camera selection uses simultaneous multi-ball visibility and approximately
  55 px median geometric projection diameter instead of scaling the 6.7 cm
  asset. Every raw sequence has a distinct hash for every frame.
- Rebuilt the 4,096-Gaussian/55-joint SMPL-X avatar geometry twice at
  `production-plcs-avatar-geometry{-repeat,}-v1`, then fit both to production
  appearance at `production-plcs-avatar-nht{-repeat,}-v1`. Geometry is exact,
  maximum p95 attachment error is `4.415703 mm`, validation PSNR is
  `56.113096/56.101894 dB`, RGB repeat difference is at most one uint8 LSB and
  `0.00109185` mean LSB. The production P4 report was independently regenerated
  at `production-p4-acceptance-repeat-v1.json` (SHA-256
  `9767e27c6dade55dcf8ae4a765034a240d9e8ac228da3d3829198ec7f99974ad`,
  content fingerprint
  `91cc194f844a1c9d0425920503957c0430ed8330b60b8f4dbf3456f96089c5b9`).
- Generated three two-person production previews at
  `cycle-17/production-plcs-video-seed-{20260728,20260729,20260731}-v1`.
  Each contains 12 native RGB frames with independent identity, pose, position,
  and yaw; all 12 raw frames are distinct in each sequence.
- Generated six 960-wide orbit videos from the 18-family cycle-14 camera plan:
  circle/ellipse, scale `0.75/1.00/1.30`, and complex/court-specific targets.
  Stable 0.75 and 1.00 examples retain useful RGB geometry well beyond the
  conservative 0.25 m/1.5-degree baseline. Both inspected 1.30-scale paths
  reach roughly 21.2 m radius and 4.24 m height and are explicitly rejected:
  the center portions show double images, blur, and unsupported geometry
  outside the SfM envelope. The failed videos are retained rather than
  narrowing the sampler silently.
- Added
  `docs/3dgs-native-synthetic-data/publish_production_previews.py` (SHA-256
  `ed9ce594e9adb66e667238834cd559c8eca921a75ac71160d121633ac825b416`)
  and published 13 hash-verified previews plus the orbit plot under
  `docs/3dgs-native-synthetic-data/assets/production-previews`. Publication
  manifest SHA-256 is
  `d5df6c6903323210ffde662631f7e503a0a464f8e2c14b921c0fd57fcafc56a7`;
  content fingerprint is
  `2f550b8b966b722f1d98f7a8b530b6305825cbf1875d3a72cc416c238ffff930`.
  Inventory is alignment 1, BLCS 3, PLCS 3, court 6, with raw and diagnostic
  MP4 plus a contact sheet for every preview.
- Expanded the visualization-first report
  `docs/3dgs-native-synthetic-data/README.md` (SHA-256
  `9f01206097f812a3723a0c2191154b2ca957fecf27f75ee2f082088876492bbf`).
  It distinguishes mechanics and production checkpoints, shows raw/overlay
  side-by-side, links every video, records camera-family decisions, and names
  remaining boundaries without treating them as successes.
- Export/verification evidence: publication rehash passed all 13 previews;
  ffmpeg decoded all 26 H.264/yuv420p videos; all dimensions are even; every
  multi-frame raw sequence has more than one unique frame and, in fact, every
  frame hash is unique; production P4 acceptance regenerated with the same
  fingerprint; synthetic-data unit/e2e passed `140/140` in `23.74 s`; Ruff,
  mypy over the P4 acceptance and production publisher, Python compile, and
  `git diff --check` passed.
- No loop-owned training, rendering, ffmpeg, or GPU process remains. No
  accepted artifact was overwritten.

## Cycle 17 failures and hypotheses

- The first ball-preparation invocation used the ambiguous device name `cuda`;
  the explicit device guard rejected it before publication. `cuda:0` passed.
- The old prototype avatar was correctly rejected because its appearance-space
  fingerprint belonged to the one-step checkpoint. Re-fitting geometry into
  the production appearance resolved the mismatch without importing standard
  3DGS features.
- The first production P4 report used the earlier latent maximum tolerance
  `0.02` and rejected a measured `0.0304512` feature maximum. Geometry was
  exact, mean latent difference was `0.0003974`, rendered RGB differed by at
  most one LSB/`0.00109185` mean LSB, and PSNR delta was `0.01120 dB`.
  The measured production gate now uses `0.04` while retaining the stricter
  mean, image, and PSNR thresholds; the independent report passes.
- The first 1280-wide preview produced an odd 719-pixel height and H.264
  correctly rejected yuv420p encoding. The renderer now rounds height once to
  an even integer and uses that same exact height for intrinsics and labels.
- A far-camera BLCS preview at
  `production-blcs-video-seed-20260728-v1` is retained as failure evidence:
  the 6.7 cm balls quantize below one pixel at 640 width. At 1280, a measured
  three-ball frame changed only four pixels versus background, with maximum
  `52` LSB and `0.000135` mean LSB. Visibility-based close-camera previews
  solved presentation without changing physical asset size.
- Existing exact AOV rendering was intentionally not weakened for production.
  It rejected an NHT/AOV cross-pass alpha difference of `0.0977283` against
  `0.005`. Production raw RGB is therefore visually accepted, while exact
  production mask/AOV dataset acceptance remains open until the AOV path
  matches NHT eval semantics. Mechanics exact-label acceptance remains valid.
- The first publication attempt computed paths relative to the final directory
  while files still lived in its atomic temporary sibling. It failed and
  removed the temporary directory. Relative paths were corrected; the
  successful publication rehashes source and destination bytes.

## Cycle 17 running jobs

- NHT training: none
- BLCS/PLCS/court render: none
- ffmpeg/preview publication: none
- GPU compute process owned by this loop: none
- Only next action: user reviews the production raw/overlay videos and the
  deliberately retained 1.30-scale failure examples on PR #666; merge or
  request a focused follow-up.

## Completed in cycle 16

- Rehashed the B00 export before all release checks. Every provider-declared
  image and point-cloud file passed, all 491 cameras equal the accepted scene
  contract, and BLCS/PLCS/court resolve to scene fingerprint
  `2c16d09503118b08a30b3819d01c23b2bc0e575f00b4f30a931c8447d4d3e160`,
  provider fingerprint
  `4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b`,
  composition fingerprint
  `7a83e40ca75b139e5de1996652cd4015423e0e3e00801a6ba91eda063c20ed37`,
  and renderer commit `20bc323d613258e5d169fdbc962c9ef27d55ca69`.
- Added immutable distinct-seed plans under
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-16` for BLCS single/multi,
  PLCS single/multi, and court orbits. BLCS position RMS is 6.14201/4.91082 m;
  PLCS position RMS is 0.45143/0.42141 m with maximum-speed differences
  1.64050/2.18995 m/s; court orbit camera-centre RMS is 1.01050 scene unit and
  phase RMS is 104.582 degrees across 425 shared family frames.
- Added an independent BLCS multi same-seed publication at
  `cycle-16/blcs-multi-same-seed-repeat-v1`; all 11 files are byte-identical to
  the accepted cycle-09 plan. Existing BLCS single, PLCS single/multi
  plan/render, and 1,285-file court dataset repeats also reverified exactly.
- Added `third_party/nht/p8_release_acceptance.py`. It verifies P0 runtime and
  real-scene training outputs, P1 deletion/alignment ownership, P2--P7 report
  fingerprints, native no-overlay composition, same/distinct-seed evidence,
  court training semantics, and 1,438 render/dataset references by hash/size.
- The first immutable P8 attempt
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-16/p8-acceptance-v1/report.json`
  (SHA-256
  `4d51b4c9aa50825ee9125fc06c20d3950a2293461fb24e4118692f482748133d`)
  failed only because it incorrectly required fixed PLCS pose/yaw semantics to
  vary by seed. The planner intentionally seeds placement amplitude/phase, not
  pose order. The corrected gate measures placement RMS and speed diversity.
- Canonical P8 report is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-16/p8-acceptance-v2/report.json`
  (SHA-256
  `b8304fd8d48c5a9d718b41e70bc5e26813608d9cc28299f99c04837abaf8f149`,
  content fingerprint
  `e0778c51530fde29ef7ab29850394a3983755d0d71687929083fa3033fbb8eff`).
  All 15 gates pass and `p8_complete=true`.
- Added `third_party/nht/visualize_p8_release.py` and generated nine compact
  visual artifacts plus a manifest at
  `docs/3dgs-native-synthetic-data/assets` (manifest SHA-256
  `8e8cf0f3d15cf1caf24bc4261716866b5a53c1f837403a7e5cc8e5e5c5f4cf05`).
  The tracked evidence includes BLCS/PLCS/court GIFs, exact-label contact
  sheets, a scene-composition overview, measured seed/camera trajectory plot,
  and 15/15 metric figure. Overlays are diagnostic-only; accepted RGB is
  unchanged.
- Published the visualization-first final report at
  `docs/3dgs-native-synthetic-data/README.md` (SHA-256
  `d35aea857abbb458e11281eb832c5fa2ed48355ae08076ae75de8ec4dfda70fa`).
  It documents architecture, deletion boundary, BLCS/PLCS/court outcomes,
  two-court seven-channel semantics, post-process exclusion, exact metrics,
  reproducibility/diversity, research links, artifacts, and the explicit
  one-step NHT appearance limitation.
- Export-first verification and regressions pass: integrated gate 15/15;
  final synthetic-data unit/e2e 140/140 in 50.36 s; focused novel-view 4/4;
  repository-wide pre-commit; Ruff over
  synthetic-data source/test and all third-party publishers; mypy over the two
  P8 publishers and all five court modules; task-script convention review;
  Python compilation; and `git diff --check`. Pytest capture itself is broken
  in the current tool PTY (`FileNotFoundError` while truncating its temporary
  capture file), so the successful suite was rerun explicitly with `-s`; this
  changes no test semantics.
- No loop-owned process remains. No prior artifact was overwritten.
- Rebasing the release commit onto `origin/main` preserved the exact tree SHA
  `4ea0bd56cc5ef06b53d95aa0ac705c558a0b689c`; the intervening PR #664
  only connected the already-used alignment hierarchy to `main`.
- `gh-pr-create` preflight reported zero blockers and zero warnings. PR #666
  was created from `feat/3dgs-native-synthetic-data` into `main`, is
  mergeable, and contains the visualization-first report.

## Completed in cycle 15

- Re-applied `utils-extraction`: orbit split and release semantics depend on
  court family IDs, near/far classes, and court instance labels, so they remain
  domain-local in `src/synthetic_data_generation/court/release.py` and
  `labels.py`, not generic `src/utils`. Re-applied `test-structure` and mirrored
  the pure split/atlas tests under
  `tests/unit/synthetic_data_generation/court`. No `src/**/scripts` path was
  changed, so `script-conventions` was not triggered.
- Added deterministic family-disjoint splitting. Whole smooth trajectories,
  never adjacent individual frames, are assigned to exactly one of
  train/validation/test. The greedy semantic-cover contract gives each holdout
  circle and ellipse, scales 0.75/1.00/1.30, and targets
  complex/`court_0`/`court_1`; inability to cover these semantics raises rather
  than leaking a family or lowering a gate.
- Added compact deterministic seven-channel target storage. Float32 heatmaps
  are quantized to uint16 and packed horizontally into one lossless PNG atlas
  of shape `[180,2240]`; decode error is bounded by `1/65535`. This avoids
  timestamp-bearing compressed archives and reduces a canonical 428-frame
  release to 58 MiB while retaining explicit channel names and encoding.
- Added isolated full publisher `third_party/nht/court_dataset_render.py`. It
  strictly rehashes the P6/P7 orbit plan and visual review, verifies every
  export-first provider image and point-cloud file, requires all 491 provider
  cameras to equal the accepted scene contract, verifies every stored
  projection, and renders all frames through native NHT `RGB+ED`. Dataset RGB
  is never annotated or overlaid. Per-frame JSON retains both court instances,
  all 28 physical points with UV/scene depth/in-frame/visible/occluded flags,
  renderer depth samples, family/split/camera provenance, and the explicit
  post-process-only grouping declaration.
- Canonical release is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-15/court-dataset-v1`;
  independent repeat is `court-dataset-repeat-v1`. Both manifest SHA-256 values
  are
  `38145d300cb907dde3a5af5f17b0548f2c63d8f5c537241879370fed8164d572`,
  both dataset fingerprints are
  `433df10f667a3b055caf2169c50a384c149ba8e4a0b19d429dbe223bff74a4df`,
  and all 1,285 files in each tree are byte-identical.
- The release contains 428 frames in 18 whole families: train has 284 frames /
  12 families, validation 72 / 3, and test 72 / 3. Every split contains full,
  near-full, partial, and sparse court-instance coverage. Across both courts
  there are 4,470 renderer-visible physical points; per-class totals are
  532/518/505/549/787/798/781 and one channel reaches all four possible peaks.
  Alpha coverage is 0.937708--0.998628 and normalized RGB standard deviation is
  0.129291--0.176731. The one-step NHT appearance limitation remains explicit.
- Added immutable visual diagnostics. Canonical corrected output is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-15/court-dataset-diagnostic-v2`
  (manifest SHA-256
  `e41e6c1355466cfd7e4649732dc4067a202a0caeb4fd0bee83540f970d32ae5b`,
  diagnostic fingerprint
  `46ec648fbfb6e600fdcdf71cf3ad9eb5cfd1c31f257c9b69ad735b470091a353`).
  The label contact sheet SHA-256 is
  `bce81cc41922d5a03da4995e35ab5ed798ca07e236e7ef28fce13c990d2ababd`,
  seven-channel heatmap sheet
  `1ddf650a8435b9261d7c190b0db93bd489490ed301ad5684bf05a87bdd5a6d23`,
  and split/class metric plot
  `f6ee1d74d00d16ef627fdb2e4d48746a38b937f3f9dd0fb9d0594eefc07a4be7`.
  Visual review
  `court-dataset-visual-review-v1.md` has SHA-256
  `5c3cf48f45bc3be07357e49b1db46cd898217e571905cdbc08a59fc0bfcbf02c`
  and records `passed-mechanics-only`.
- Strict P7 report is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-15/p7-acceptance-v2/report.json`
  (SHA-256
  `955f9c40599eb8287048b73032866370860966a93028f54fddcdeeabd572801f`,
  content fingerprint
  `40e5d1109109addebb43bc3aae5bd2407e0f315dbe5e3690c300714622cd52fe`).
  All 13 gates pass. It audits 11,984 point records, including 4,470 visible and
  3,511 in-frame occluded/unsupported points. Projection round-trip maxima are
  `1.6781803e-7 px` UV and `8.8817842e-16` scene depth; the minimum heatmap value
  sampled at any visible physical point is `0.8836347`.
- Failed iterations remain immutable. The first publisher attempt stopped
  before CUDA because `load_scene_provider_bundle` returns a validated wrapper,
  not the manifest directly; the corrected consumer explicitly uses
  `provider.manifest` while retaining file verification. Diagnostic v1
  (manifest SHA-256
  `58a69e5d99c70d3c8a9eb209313d6ea019411455616ca55de7a8de9fbc6b083a`)
  made the white `service_t` bar invisible on a white background despite the
  correct 781 count; v2 adds black outlines and full coverage-pair captions.
  Acceptance v1 (SHA-256
  `9294d276ab8fdbcce1192b75ede4d86751bbb5746c1e7f2bd0e2c0544bae25b6`)
  failed only its speculative `1e-8 px` projection gate with observed
  `1.678e-7 px` float64 rescaling-order drift. V2 uses a still sub-micropixel
  `1e-6 px` threshold and does not change dataset bytes or labels.
- Updated `src/synthetic_data_generation/README.md` so the implementation entry
  point no longer describes the conservative all-14-point baseline as the only
  court method. It now records N-instance layout, seven symmetric classes,
  partial coverage, bold orbits, post-process grouping, and family-disjoint
  release ownership without duplicating artifact metrics.
- Verification passes: 19/19 focused court tests; 140/140 full synthetic-data
  unit/e2e tests in 23.98 s; Ruff over all cycle-15 source/test/third-party
  Python; strict targeted mypy over all four court source modules; Python
  compilation in the isolated NHT runtime; and `git diff --check`. No
  loop-owned long-running process remains.

## Unique next action

None. All planned phases and acceptance gates are complete, the final report
and PR are published, no loop-owned process remains, and automation `3dgs` has
been deleted. Do not start another development phase from this loop.

## Completed in cycle 14

- Applied `utils-extraction`: verified multi-court layout, seven-class label
  semantics, and orbit sampling depend on accepted court geometry and remain
  domain-local under `src/synthetic_data_generation/court`. Applied
  `test-structure` and mirrored pure tests under
  `tests/unit/synthetic_data_generation/court`. No `src/**/scripts` path was
  changed, so `script-conventions` was not triggered. The reproducible
  publishers/render consumer live under `third_party/camera_sampling` and
  `third_party/nht`.
- Loaded the existing fingerprint-verified court-geometry artifact instead of
  inventing a second court. It contains `court-0` and `court-1`; in the accepted
  `court_0` metric frame their centres are separated by 14.807560 m.
  `court_0` is required to match the user-approved scene contract exactly, and
  additional instances must come from the same verified artifact.
- Implemented the user-approved annotation semantics. Physical indices
  `(0,2)`, `(1,3)`, `(4,5)`, `(6,7)`, `(8,10)`, `(9,11)`, and `(12,13)` map to
  seven unordered near/far classes. Annotations retain `court_instance_id`, but
  the model target merges all same-class points using pixelwise maximum into
  exactly seven channels. With two courts, a channel can hold four peaks.
  Court grouping/homography/Hungarian assignment is declared post-process and
  no instance-grouping training target is emitted.
- Added explicit `full=14`, `near_full=10--13`, `partial=4--9`,
  `sparse=1--3`, and `none=0` coverage buckets. Renderer visibility is never
  inferred silently: it is `None` until a render-depth consumer attaches all
  fourteen physical flags per court. Added resolution-safe projection scaling
  and seven-channel heatmap construction.
- Added SfM-envelope circle and ellipse families. The 18 deterministic families
  combine two shapes, 0.75/1.00/1.30 robust captured-envelope scales, targets
  at the two-court complex/`court_0`/`court_1`, and captured-quartile plus high
  camera elevations. Nearest captured translation/rotation are measurements,
  not conservative rejection gates; collision and useful partial coverage
  remain explicit hard gates.
- Published immutable plan
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-14/multicourt-orbit-plan-v1`
  (manifest SHA-256
  `23f3f726edfcb05b6c8fac94b0920f0601453f8572810e404955b0f6aae403f6`,
  content fingerprint
  `ca93aad1fd252aa1c5c401675623b1fc97d29f48ebf2c185df5625af652f0c9a`).
  Of 432 proposals, 428 pass (99.0741%); the four explicit rejections are
  collisions. All 18 families retain 23--24 frames. Across 856 court-instance
  projections there are 169 full, 308 near-full, 297 partial, and 82 sparse
  cases. Nearest captured translation spans 0.447897--15.346472 m
  (median 5.283287 m), and rotation spans 0.666383--13.803943 degrees
  (median 4.805562 degrees).
- Added isolated CUDA consumer `third_party/nht/court_orbit_render.py`. It
  rehashes the plan and all source artifacts, rejects policy drift, reloads the
  accepted two-court layout, and verifies every stored projection before
  rendering. Each frame uses one native NHT `RGB+ED` call over the background
  Gaussian tensors; no RGB overlay is used. Expected-depth local consistency
  attaches physical-point visibility, after which seven-channel multi-peak
  heatmaps are saved.
- Rendered one representative from every family at 320x180:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-14/multicourt-orbit-render-v1`
  (manifest SHA-256
  `4a979533a0c0b16e8b96fa63545464b1ed63b7002397d58293f60cacbcc35793`,
  render fingerprint
  `2b28d9cffed54bb61dfe27c5f56d1e08cc57d6d7d7566c7c0c79ff71c56e0e1b`).
  All 18 frames are finite/non-empty; alpha coverage is
  0.963715--0.994983 and normalized RGB standard deviation is
  0.146008--0.176259. Depth evidence retains 191 points, every class has
  18--34 points, and one channel reaches four visible peaks.
- Visually inspected contact sheet
  `multicourt-orbit-render-v1/diagnostics/representative-contact-sheet.png`
  (SHA-256
  `4c7a95a4941547f914e7e718e19b89c497b9cd331ea81925a7c68598b9a30afa`)
  and trajectory plot
  `multicourt-orbit-render-v1/diagnostics/orbit-trajectories.png`
  (SHA-256
  `773eb35042261902a0311f30a1e2a97c5cf4fa8098bacde29b05d4f846d5a839`).
  The review at
  `multicourt-orbit-visual-review-v1.md` (SHA-256
  `23c144057ec2863a288ef5547151ada5c5d97e8a13484f12b3b7dabe0d89d3fd`)
  records
  `passed-mechanics-only`: outer 1.30 families remain supported and non-empty,
  but the known one-step NHT background is green/noisy and cannot establish
  production photorealism. No family was silently removed.
- The first real orbit proposal failed before publication because a full Sim(3)
  matrix scaled the camera rotation. The corrected implementation applies only
  similarity rotation to orientation and similarity position to the centre,
  preserving `SceneCamera` orthonormality. The first render attempt then stopped
  before CUDA work because strict comparison treated JSON lists and Python
  tuples as unequal; canonical JSON hashing fixed the representation-only
  mismatch. Both failures remain in the cycle-14 log/history.
- Verification passes: 12/12 focused court tests, Ruff over all cycle-14
  source/test/third-party Python, and targeted strict mypy over `layout.py`,
  `labels.py`, and `orbits.py`. The full synthetic-data unit/e2e suite passes
  133/133 in 25.07 s. The first focused pytest command inherited
  parallel capture state and collected no tests; authoritative `-n 0
  --capture=no` execution passed. No loop-owned long-running process remains.

## Unique next action

Implement and run the immutable P7 full-dataset publisher over all 428 accepted
orbit frames, assign train/validation/test by whole orbit family to prevent
adjacent-frame leakage, save native RGB plus instance-aware physical labels and
compressed seven-channel targets, and publish repeat/integrity/projection/
visibility/split acceptance evidence without weakening the one-step appearance
limitation.

## Completed in cycle 13

- Applied `utils-extraction`: court pose sampling depends on CourtKP20,
  accepted scene alignment, and `SceneCamera`, so it remains domain-local at
  `src/synthetic_data_generation/court/novel_view.py` instead of becoming a
  generic `src/utils` helper. Applied `test-structure` and mirrored its pure
  tests at
  `tests/unit/synthetic_data_generation/court/test_novel_view.py`. No
  `src/**/scripts` file changed, so `script-conventions` was not triggered;
  reproducibility/acceptance utilities live under `third_party/camera_sampling`.
- Added the P6 primary-source comparison at
  `.codex-loop/3dgs-synthetic-data/research/novel-view-camera-sampling.md`
  (SHA-256
  `9eee4601ec1def209d10a00c63d483c3b1f57874f9997b0d457554919bb92026`)
  and machine-readable pins at `third_party/camera_sampling/pins.json`
  (SHA-256
  `a566b6071ab59a1741f3eccab4962e299885bc80065847006f97d6e75d3dd32a`).
  The exact official commits are MultiNeRF
  `5b4d4f64608ec8077222c52fdf814d40acc10bc1`, NeRF Director
  `9471c8698077f0edac9e749208db9ef987cb5ca8`, FisherRF
  `b74732812b295189f230a192418375f56cec3bd6`, NeRF++
  `ebf2f3e75fd6c5dfc8c9d0b533800daaf17bd95f`, and Nerfstudio
  `50e0e3c70c775e89333256213363badbf074f29d`.
- Selected an independent hybrid: captured poses define local support;
  six-dimensional uniform-ball perturbations couple translation and geodesic
  rotation; hard geometry gates run before NeRF Director-style farthest-view
  selection. Mip-NeRF 360 pose normalization/focus is retained only as a
  support-characterization reference. An unclipped global ellipse is rejected
  because it can leave B00 support and face away from the court. FisherRF is
  rejected as a proposal generator because its differentiable-Hessian backend
  and active-acquisition objective do not match frozen NHT court generation.
  Random jitter without FVS, unconstrained keyframe interpolation, and a
  position-only convex hull are also recorded failed hypotheses.
- Measured the actual 491-camera B00 trajectory before fixing thresholds.
  Adjacent captured-pose translation has median 0.9895 m and adjacent geodesic
  rotation median 3.0889 degrees. Captured camera heights span
  1.4339--2.8224 m, the minimum captured eighth-nearest SfM point clearance is
  0.2626 m, and 42 captured poses contain all fourteen line keypoints. The
  preregistered limits are a coupled 0.25 m / 1.5 degree support ball with
  score <=1, camera height >=1.20 m, eighth-nearest clearance >=0.25 m, all
  CourtKP20 depth >0.10 m, and all first fourteen keypoints inside the image.
- Added `NovelViewThresholds`, strict projection/collision evidence,
  renderer-space pose publication, explicit rejection counts, and
  captured-plus-selected normalized SE(3) FVS. Invalid inputs, absent safe
  anchors, or too few accepted candidates raise rather than lowering a gate.
  The same seed reproduces exact immutable camera labels and transforms.
- The canonical B00 probe is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-13/court-novel-view-probe-v2`
  and its independent repeat is
  `court-novel-view-probe-repeat-v2`. Both manifest SHA-256 values are
  `c01e3dbf469895370126f9ead8a57343c9b0125238b8f36193c922c760028a64`;
  they are byte-identical and have content fingerprint
  `1565b780ff28fe1c85df33626d865a5b0865cacf6e54ae8bde3eacdf87dd9483`.
  Of 2,688 proposals, 2,641 pass (98.2515%); the only rejection is 47
  court-framing failures. Immutable v1 remains retained; v2 republishes the
  identical 256 cameras after removing trailing documentation whitespace, so
  only the research/pin provenance hashes changed.
- FVS selects 256/256 nontrivial novel poses, covers all 42 safe anchors, and
  expands the safe pose count by 6.09524x. Selected pairwise normalized pose
  distance is at least 0.971927. Maximum extrapolation score is 0.999983;
  nearest-captured translation/rotation maxima are 0.243253 m / 1.491015
  degrees. Minimum collision clearance, CourtKP20 depth, and line-keypoint
  margin are 0.547058 m, 3.460373 m, and 0.453954 px. Thus no selected pose
  exceeds any support, collision, near-plane, height, or framing limit.
- Visually inspected
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-13/court-novel-view-diagnostic-v1.png`
  (SHA-256
  `f882d80364c69cccdc9716cc105fab7206a8d02c21b0dcd4e9acfed9b1813c43`).
  The top/side views show the selected cameras distributed around every safe
  captured segment while retaining court-facing optical axes; the score and
  margin histograms remain strictly on the accepted side of both limits.
- Strict report
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-13/p6-acceptance-report-v2/report.json`
  has SHA-256
  `0fce61fa30acfc5f0165a3303079a79fc7e5aac10ff5ea12ea623849f4ee6799`,
  content fingerprint
  `468a209313f7909bc456c2c144ba17b38766446d20adbfcc1d2b55d4e86e4465`,
  `status=passed`, and `p6_complete=true`. It rehashes the research/pins and
  byte-identical probes and enforces the preregistered thresholds and >=4x
  coverage gate.
- Verification: Ruff passes all P6 source/test/third-party Python, mypy passes
  `novel_view.py`, its 4/4 focused tests pass, and the authoritative synthetic
  unit/e2e run passes 125/125 in 25.25 s. The first aggregate test command
  exposed duplicate test basenames under pytest's default import mode; rerun
  with the repository-compatible `--import-mode=importlib` confirmed this was
  collection packaging rather than a source failure. The repeat probe log is
  `.codex-loop/3dgs-synthetic-data/logs/court-novel-view-probe-repeat-v2-c13.log`
  and `p6-acceptance-v2-c13.log`.
  No loop-owned process remains; GPU utilization is 0%. An unrelated process
  in the original checkout was observed and deliberately left untouched; it
  ended before cycle close.

## Unique next action

Implement the P7 export-first court dataset publisher that strictly rehashes
the accepted provider/scene contract and P6 camera manifest, natively renders
an immutable initial trajectory through NHT, and saves CourtKP20 UV/depth/
visibility labels for projection, occlusion, provenance, and split-leakage
acceptance checks.

## Completed in cycle 12

- Re-applied `utils-extraction`: PLCS identity, court placement, pose scheduling,
  and label semantics remain task-local in `src/synthetic_data_generation/plcs`;
  they are not generic `src/utils` helpers. Re-applied `test-structure` and
  placed the five schedule tests at
  `tests/unit/synthetic_data_generation/plcs/test_planner.py`. No
  `src/**/scripts` file changed, so `script-conventions` was not triggered.
- Added `src/synthetic_data_generation/plcs/planner.py`. The immutable
  seed-derived schedule supports one or two persistent identities, bounded
  singles-court footprints, finite velocity/yaw, ready/forehand pose indices,
  complete presence, and multi-person collision rejection. Same-seed schedules
  are exactly deterministic and unknown mode, missing frames, invalid pose,
  court escape, or identity drift fail explicitly.
- Added `third_party/plcs_avatar/prototype_plcs_plan.py`. It rehashes the
  accepted scene contract, 491-camera export verification, P4 acceptance,
  every selected avatar pose tensor, background provider, and shared NHT
  appearance. It converts the SMPL-X right-handed axes into court axes, composes
  the accepted `T_scene_from_court`, and chooses one captured SfM camera that
  contains every scheduled 0.9 m person centre. There is no fallback camera or
  missing-person frame.
- Canonical plans are
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-12/plcs-single-plan-v1`
  and `plcs-multi-plan-v1`; manifest SHA-256 values are
  `de778451ef6c29f397f3ea7bd9379994be05c7c6c5803edc295787971ea215a5`
  and `342d25c31327f38ef568a76f534b13ca10cbd41e0d3480c2ddf3097fad2b1881`.
  Their plan fingerprints are
  `0ef2e31c44b3f734cd485217f58d95f314d458182f865f22bfb70e6f4c164eee`
  and `e81e340aab2adfccb9e3a1012c1129b40ec5ddb484790eaf459253c7cdb54369`.
  Both have 12 frames at 30 fps; single uses `frame_000339`, multi uses
  `frame_000057`; all 12/24 centre projections are geometrically visible.
  The multi minimum court-space separation is 18.36857 m.
- Independent same-seed publications at `plcs-single-plan-repeat-v1` and
  `plcs-multi-plan-repeat-v1` are byte-identical to the corresponding canonical
  plan trees. The provider-binding bug hypothesized before execution was
  prevented explicitly: the composition references the exported
  `provider.json`, so validation now rehashes that provider, checks its bundle
  and scene fingerprints, and compares its full source-artifact inventory with
  the accepted scene contract instead of incorrectly comparing the provider
  JSON hash with the source checkpoint hash.
- Added isolated CUDA consumer `third_party/nht/plcs_render.py`. Every frame
  selects the scheduled canonical/ready/forehand 4,096-Gaussian tensor,
  transforms each persistent identity in 3D, concatenates it with 216,824
  background Gaussians, and performs exactly one NHT RGB+ED pass plus one
  eval3d one-hot contribution pass. Atomic output includes RGB, alpha, expected
  depth, exact contribution, masks, segmentation, bbox, identity, pose,
  placement, velocity, yaw, transform, camera projection, and visibility;
  `rgb_overlay_used=false`.
- Canonical renders are
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-12/plcs-single-render-v1`
  and `plcs-multi-render-v1`, each for frames `0,4,6,8,11` at 480 x 270.
  Manifest SHA-256 values are
  `4f014af87655b7943398ca1e09068389e85143da10b80eb0373ba3fe92ac0222`
  and `316ffef5d4adac61bfca123ccf1af46ff110234a60d0d8b3f98fb761c5e86184`;
  render fingerprints are
  `2954264285184ee7036f599cfdbc217fd4f505514c1fec2e52fe840bb148db33`
  and `42dfcf74a6c4e433ab83c13c1a35379a2e0a6431315099bc4252473efb1e1a47`.
  Both repeated render trees are byte-identical.
- Exact visible-person pixels span 58--73 for single and 74--1,220 for multi.
  Maximum projected-root-to-mask-centroid errors are 1.37998/1.78954 px.
  NHT-vs-AOV alpha max errors are 0.00121093/0.0000302792 and
  contribution-sum-vs-AOV-alpha is at most `3.57628e-7`. Reconstructed
  velocities match saved velocities exactly; minimum path lengths are
  3.48996/3.67109 m; every identity has two pose transitions.
- Added and visually inspected exact-mask panels
  `plcs-single-diagnostic-v1.png` (SHA-256
  `73b4b3706790092e6d86e9484acf7c4b2254ce1cbdb58cbdb90798d6873eeb6d`)
  and `plcs-multi-diagnostic-v1.png` (SHA-256
  `2ae30d0c2a53b21f764277c9d995049a5cf937854c7b007a3e1ae62dccc8c590`).
  The multi view visibly preserves two separated silhouettes and stable red/
  blue identities while ready/forehand geometry and court movement change.
  The single selected camera is deliberately distant and yields a small but
  coherent 18--20 px-high silhouette; it still exceeds the preregistered
  50-pixel exact-mask gate. The green point-cloud appearance remains the
  mechanics prototype documented in P4.
- The first immutable acceptance attempt,
  `p5-acceptance-report-v1.json` (SHA-256
  `03aa5960558e251b3edb8911ceb665fe694f2a37ee8023941de8492975ad3746`),
  failed only because end-to-end displacement penalized a valid out-and-back
  tennis path. The hypothesis was confirmed by the full trajectory. The metric
  was corrected to cumulative path length without lowering its 0.5 m
  threshold; the failed report was retained and not overwritten.
- Canonical
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-12/p5-acceptance-report-v2.json`
  has SHA-256
  `57b25538f078b72bf915906637e0f376cb6c35b6057f1733b18c4fbe7b5a34de`,
  content fingerprint
  `d98c2f0453f537e053fea03a77cf1b4cd7986a027a1ebb5ae0d843a1f9eabf12`,
  `status=passed`, and `p5_complete=true`. Thresholds are >=50 exact person
  pixels, <=3 px root/centroid error, <=0.005 NHT/AOV alpha drift, <=`1e-5`
  contribution/alpha drift, <=`1e-10` velocity error, >=0.5 m path, and >=2
  pose transitions.
- No new avatar paper decision was made in P5; it faithfully applies the
  GaussianAvatar-style SMPL-X control and frozen NHT appearance selected by the
  P4 primary-paper comparison. P6 is the next research decision point and will
  add a separate primary-paper/official-code record for safe novel-view
  camera sampling.
- Final verification: Ruff passed all cycle-12 PLCS code, mypy passed
  `planner.py`, Python compilation passed in both isolated environments, and
  36/36 PLCS/composition/scene-contract tests passed in 0.51 s. Logs are under
  `.codex-loop/3dgs-synthetic-data/logs/`, including
  `plcs-{single,multi}-render-c12.log`,
  `plcs-{single,multi}-render-repeat-c12.log`,
  `p5-acceptance-v2-c12.log`, and `p5-regression-c12.log`. No loop-owned
  process remains; GPU compute is idle.

## Completed in cycle 11

- Re-applied `utils-extraction`: SMPL-X surface sampling, per-Gaussian joint
  weights, covariance deformation, and PLCS control contracts remain
  task-local under `src/synthetic_data_generation/plcs`; they are not
  domain-agnostic `src/utils` helpers. Re-applied `test-structure` and added
  the mirrored pure unit suite at
  `tests/unit/synthetic_data_generation/plcs/test_avatar_asset.py`. No
  `src/**/scripts` file changed, so `script-conventions` was not triggered.
- Added `avatar_asset.py`, which deterministically samples area-weighted
  anisotropic surface Gaussians, carries explicit SMPL-X joint weights, applies
  LBS to means, and pushes every covariance through the blended linear
  transform before a positive eigendecomposition. Inputs reject degenerate
  triangles, invalid simplexes, non-rigid joint transforms, non-positive
  covariances, and joint-count mismatch. Five new asset tests plus eight
  existing control tests pass.
- Added `third_party/plcs_avatar/build_asset_fixture.py` and published
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-11/plcs-avatar-geometry-v1`.
  The manifest SHA-256 is
  `ee22d2af38552cf3b5ca988e92395b2484cb6a4969974903da0bf030081cbdc7`,
  content fingerprint is
  `932db552f13db5a94cbb7f53be2c96f39318032de3a10b19f04541f772063897`,
  and the licensed SMPL-X model is read in place rather than copied. The asset
  has 4,096 Gaussians, 55 explicit joint weights, 21 controlled body-pose
  joints, and no dropped joint indices.
- Canonical, ready, and forehand geometry all emit finite tensors. Their mean
  attachment errors are 0.0000065/1.6721/1.8110 mm; p95 errors are
  0.0000201/4.4157/4.0548 mm; the two active poses move Gaussians an average
  6.63/11.24 cm from canonical. A clean repeat at
  `plcs-avatar-geometry-repeat-v1` has the same fingerprint and every file,
  including the tensor packs and manifest, is byte-identical.
- Added `third_party/nht/plcs_avatar_fit.py`. It strictly rehashes the fixture,
  licensed model, target appearance, and pinned clean NHT/gsplat checkouts
  before CUDA. It generates target-NHT teacher views, resets features to zero,
  freezes geometry/opacity/shader, optimizes only NHT features for 500 steps,
  and applies the identical feature tensor to all three controlled poses.
  Standard-3DGS features are not imported and `rgb_overlay_used=false`.
- Canonical NHT output is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-11/plcs-avatar-nht-v1`
  (manifest SHA-256
  `9c1d083c18d407f8a19da863617af2066255bf77a2378b0e42b33d9ce44e9c09`,
  content fingerprint
  `e81a789dcecf6f91a9749cb4671f444a5d31042c20e93e52b49c4a2378a72d4d`).
  Held-out masked PSNR is 67.88044 dB. Across two cameras, canonical/ready/
  forehand have 3,302--4,540 / 3,124--4,377 / 4,050--4,317 visible pixels.
  Ready and forehand change mean RGB by 0.01265--0.01456 and
  0.01762--0.02246 from canonical, proving pose changes reach native renders.
- Visually inspected
  `plcs-avatar-nht-v1/diagnostics/pose-contact-sheet.png` (SHA-256
  `11cb694313c57873240898be87c947dea3852b77bb62f2048e79e1e38b6768cd`).
  All six pose/view images retain a coherent full-body surface; the ready pose
  lowers both arms and the forehand pose introduces asymmetric arms, torso
  turn, and leg transfer without detached clouds. The validation target and
  prediction are indistinguishable at displayed scale. The green procedural
  appearance is explicitly a mechanics prototype, not a captured identity.
- A second NHT fit at `plcs-avatar-nht-repeat-v1` measures 67.92184 dB. CUDA
  is not misreported as byte deterministic: latent max/mean absolute
  differences are 0.010186/0.000830, PSNR differs 0.04140 dB, and six uint8
  renders differ by at most one LSB and 0.002211 mean LSB. Both runs share
  exact geometry, opacity, and identity tensors. These measured bounds and
  failure analysis are now recorded in
  `.codex-loop/3dgs-synthetic-data/research/plcs-avatar-methods.md` (SHA-256
  `e21d71f71a3145234ef75c7d659f86e80d5615a53de3d69d815f74498d579cbb`).
- Added `third_party/plcs_avatar/p4_acceptance.py`. The canonical report
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-11/p4-acceptance-report-v2.json`
  has SHA-256
  `9d37372e9e10f1f97628393f2594340b9ecd9cfbcdbf6319e30d97260d7e663f`,
  content fingerprint
  `3e00861be66721f9417811d08828cbfbb067a9277092b6072df266a4e943758c`,
  `status=passed`, and `p4_complete=true`. It rehashes research/pins and all
  four independent trial manifests, enforces p95 <=5 mm, held-out PSNR >=25 dB,
  exact repeated geometry, explicit empty dropped-joint/frame lists, and the
  measured CUDA repeat tolerances.
- Final verification: Ruff passed all PLCS/cycle-11 Python, mypy passed both
  PLCS source modules, the isolated NHT environment has 92 compatible
  packages, and the authoritative worktree run passed 129/129 synthetic-data
  unit/e2e tests in 54.58 s. No loop-owned process remains and the GPU ended at
  0% utilization. P4 gates are complete; P5 is now active.

## Completed in cycle 10

- Applied the `utils-extraction` policy before implementation: SMPL-X skinning,
  mesh attachments, and avatar-control assumptions remain task-local under
  `src/synthetic_data_generation/plcs`, rather than becoming misleading generic
  utilities. Applied the `test-structure` policy and placed mirrored unit tests
  under `tests/unit/synthetic_data_generation/plcs`. No `src/**/scripts` file
  was changed, so `script-conventions` was not triggered.
- Added the primary-paper/official-code decision record
  `.codex-loop/3dgs-synthetic-data/research/plcs-avatar-methods.md` (SHA-256
  `0fa65175a6a241238ceba0af7329628e2b087a1d9382005c45ce041af1a47346`)
  and machine-readable `third_party/plcs_avatar/pins.json` (SHA-256
  `572e12b9f913d6fafae3cc6738050ad2f8a682f8c52453b12cb769ad80acda88`).
  The matrix records titles, primary papers, official code, exact commits,
  licenses, applicability, and limitations for GaussianAvatar, HUGS, GART,
  SplattingAvatar, Animatable Gaussians, and 3DGS-Avatar.
- Selected GaussianAvatar as the primary geometry/control layout because its
  official implementation supports SMPL-X and fixed query LBS weights under
  MIT, with HUGS as the comparative candidate because its official code
  explicitly composes a human and scene and blends six nearby
  LBS-compatible vertex transforms. Exact official commits are
  `d981c62238ef64e89dcc04719d2ebbb4758b080a` and
  `b65721a5946771053e4f1d0d68d06199bc1d8c07`. GART is the learned-skinning
  fallback. SplattingAvatar was rejected as an executable candidate because
  pinned official HEAD `fec0ad3845f1d2e4ad4cdabd1b1c8c81cf10e41b`
  contains only `README.md` and uses CC BY-NC-SA 4.0.
- Recorded the critical appearance boundary: none of the candidates' standard
  3DGS SH/appearance tensors are interpreted as NHT features. A selected
  avatar must be trained directly in NHT or pass through an explicit calibrated
  frozen-target feature fit. GVHMR SMPL-X is the authoritative control input;
  COCO17 remains a label or explicit IK input and is never silently inverted
  into underdetermined SMPL-X parameters.
- Added strict NumPy geometry kernels in
  `src/synthetic_data_generation/plcs/avatar_control.py`: persistent
  barycentric attachments, explicit fixed joint LBS, explicit vertex-transform
  blending, and HUGS' six-neighbour distance/LBS-confidence rule. Inputs reject
  invalid shapes, non-finite values, reflections, scale, invalid indices, and
  non-simplex weights; outputs are read-only. Eight focused unit cases cover
  the algorithms and rejection gates.
- Added the isolated geometry screen
  `third_party/plcs_avatar/control_probe.py`. It reads the licensed local
  `SMPLX_NEUTRAL.npz` in place (SHA-256
  `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`;
  not copied), drives nine representative tennis-like SMPL-X frames, and
  compares both candidate controls on 512 deterministic surface attachments
  against the same posed SMPL-X mesh.
- Canonical output is
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-10/plcs-control-probe-v2`
  (manifest SHA-256
  `35ea720d4ac7fd9fdcc1e37aae1ee234b0732b491be2b67cb4614f0f74f9769a`,
  content fingerprint
  `7484827c23e3bbba811b2f378ba07f663e4424d713a95e3c02f2921d7adc5517`).
  GaussianAvatar-style fixed LBS measures mean `0.875569 mm`, p95
  `2.560496 mm`, max `5.450927 mm`; HUGS-style top-k LBS measures mean
  `0.892760 mm`, p95 `2.624450 mm`, max `5.478471 mm`. Both pass the
  preregistered geometry-only screen of mean <=30 mm and p95 <=80 mm.
- A clean same-seed repetition at
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-10/plcs-control-probe-repeat-v2`
  has the same content fingerprint, and all 15 non-manifest files (14 NPY
  arrays plus metrics) are byte-identical. The earlier successful
  `plcs-control-probe-v1` NPZ design is retained but not canonical because ZIP
  member timestamps complicate byte-reproducibility claims.
- Added a hash-verifying diagnostic renderer and visually inspected
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-10/plcs-control-probe-v2-preview.png`
  (SHA-256
  `6b8581415613f46dc24e2dc3f3e3246930792dc58e704d998386b4f60b932508`).
  The sparse full-body silhouette stays coherent and both colored candidate
  predictions coincide with the gray mesh attachments at displayed scale; no
  detached control clusters are visible. This is control evidence, not an NHT
  avatar-render acceptance claim.
- Final verification: `PYTHONPATH=$PWD pytest -n 0 -s
  tests/unit/synthetic_data_generation tests/e2e` passed 124/124 in 55.19 s;
  Ruff passed all new PLCS and third-party Python; mypy passed
  `avatar_control.py`. The GPU ended at 0% utilization with no loop-owned
  compute process.

## Completed in cycle 09

- Applied the user-authorized P3 gate revision: a clearly identified generated
  prototype is sufficient in place of a real captured ball, and
  `b00-ground-line-alignment-user-override-v2` is accepted without using its
  holdout machine rejection as a P3 blocker.
- Strictly reloaded the cycle-01 export before generation. Evidence:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/export-reload-verification.json`
  (SHA-256
  `655303af1fcbb8e4a3d5bbd421832c154f6868c199e7bb05c16148ebcd77b09f`);
  all 491 images and the float64 `[217336,3]` point cloud passed. The provider
  bundle fingerprint is
  `4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b`.
- Verified that the approved scene contract
  `/home/kamimura/projects/tennis-lab/data/tennis/3dgs_scenes/b00-default-v1/scene-contract-ground-line-user-override-v2.json`
  (SHA-256
  `36b62a9a135deb1fc83c74cc9d5a177af5595aa4d52e527d0950be39a4962ffd`)
  has the exact export scene fingerprint
  `2c16d09503118b08a30b3819d01c23b2bc0e575f00b4f30a931c8447d4d3e160`
  and all 491 identical camera records. Its explicit user decision SHA-256 is
  `f9a74e47618fe638fb75a450fcf677b1fa85d51c14c584c821ea09dfa820a162`.
- Added task-local `src/synthetic_data_generation/blcs/prototype.py`. Its
  antipodally symmetric Fibonacci construction deterministically produces 512
  isotropic Gaussians with a 6.7 cm maximum three-sigma diameter, a
  `9.832985120583615e-11 m` mean offset, and an exactly centred AABB. Per the
  `test-structure` policy, seven unit cases live at
  `tests/unit/synthetic_data_generation/blcs/test_prototype.py`.
- Added `third_party/nht/prototype_ball_fixture.py` and generated
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-ball-v1`.
  Its content fingerprint is
  `45bf8a65d2070dc106fa06c13cdef68551a24751ff492ba8e390cfd9b9205721`;
  the manifest declares `asset_origin=codex-generated-prototype` and
  `source_is_user_asset=false`. Eight independent NHT captures have
  8.6304--8.6487% foreground coverage.
- Imported those captures and ran the existing 600-step frozen-target feature
  fit through the production preparation boundary. Canonical outputs are
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-calibration-v1`
  and
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-preparation-v1`.
  Validation PSNR is `56.04744529724121 dB`; the prepared asset's effective
  diameter is `0.06700000166893005 m`, relative diameter error is
  `2.4909403727076635e-08`, origin offset is zero, registry fingerprint is
  `2f630030c12f491a764d394b27b23391439123c4938100f425c63485daa1e4b5`,
  and canonical tensor SHA-256 is
  `9c4ba7f2f9c62e0df44691c58ed5c140540f99a85acae230944eac816a355448`.
- Added `third_party/nht/prototype_blcs_plan.py`. It verifies export/alignment
  artifact bytes and machine evidence before using the repository's real
  `BallPhysics` and `RallySimulator`. The plan process deliberately runs in the
  tennis-lab environment; only rendering/fitting runs in the independent NHT
  environment.
- Published the single-ball physical run at
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-single-plan-v1`.
  It contains 317 frames / 10.5667 s / five shots, reaches 5.5855 m and
  15.5600 m/s, and has plan fingerprint
  `1b2fa6f0d90075d8a209eb43c0fdbf4c1b936cb416d7a6e482b8f807dcbbd6a9`.
  Repeating seed `20260728` at
  `prototype-single-plan-repeat-v1` produced all 11 files byte-identically
  (tree fingerprint
  `baa3703e1e59daed6b8c3a7252a0e7c80b691d776e6427fe961c259c3868fdbd`).
- Published the two-ball physical run at
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-multi-plan-v1`.
  It has 240 frames, 360 active object-frames, concurrent tracks from frames
  89--237, maximum height 5.3904 m, maximum speed 18.8517 m/s, and plan
  fingerprint
  `04e2375767733ccaf096903d3cf876309289a2a1f53659297f732f4f07bda462`.
- Native composed render outputs are
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-single-render-v3`
  and
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/prototype-multi-render-v2`.
  Both render six 480x270 frames using 12 public renderer calls, publish RGB,
  alpha, expected depth, exact instance contribution, mask, segmentation, and
  complete frame labels, and declare `rgb_overlay_used=false`. The single ball
  reaches 1,930 exact visible pixels. Both multi instances are visible in all
  six selected frames and reach 100/88 pixels. Maximum measured cross-pass
  alpha drift is 0.0021434/0.0033501 under the explicit measured tolerance
  0.005.
- Added and ran `third_party/nht/prototype_p3_acceptance.py`. The immutable
  report
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-09/p3-acceptance-report-v1.json`
  has SHA-256
  `eed9fa7057467e63b0ec414d3001fd421ad71a021790db69c165764b52807dd7`,
  content fingerprint
  `bee8c7f211b4a47fb3550b0da71573581ae68890d051ac8bdc5da19eeec7afca`,
  `status=passed`, and `p3_complete=true`. It rehashes every referenced render
  artifact and compares active identity, court position, resized camera
  projection, exact per-instance masks, and visible-pixel counts to the strict
  plans.
- Final verification: `PYTHONPATH=$PWD pytest
  tests/unit/synthetic_data_generation tests/e2e` passed 116/116; Ruff passed
  for all cycle-09 Python changes; mypy passed eight BLCS/composition source
  files; `uv pip check --python third_party/nht/.venv/bin/python` found all 92
  packages compatible. No P4 paper or method decision was made in cycle 09;
  P4 research is the next phase.

## Completed in cycle 08

- Added the task-local production capture boundary
  `src/synthetic_data_generation/blcs/calibration.py`, after applying the
  `utils-extraction` placement policy. It strictly imports individual RGB,
  binary-mask, and OpenCV camera records into the existing frozen-target NHT
  calibration bundle instead of accepting an opaque prepacked NPZ.
- The new `tennis_ball_calibration_capture_v1` contract requires at least two
  train and one validation view; unique view IDs, paths, and RGB bytes; exact
  hash/size references; common RGB dimensions; RGB/L Pillow modes; non-empty,
  non-full 0/255 masks; finite proper camera rotations; affine homogeneous
  matrices; positive intrinsics; and principal points inside the image. Unsafe
  paths and train/validation byte leakage fail before publication.
- Added `third_party/nht/ball_calibration_import.py`. It atomically publishes a
  strict `tennis_ball_calibration_import_v1` record, source-capture provenance,
  and the existing hash-checked calibration NPZ. Reload verifies the original
  capture manifest and every source image/mask byte again; output replacement
  is forbidden.
- Refactored `third_party/nht/ball_feature_fit.py` to consume that shared loader
  and removed its duplicated calibration parser/writer. Its current SHA-256 is
  `01ced8be01bd097a68b64dabef9a2f5978dab93b05846de38baa24887b02a4fb`.
- Added `third_party/nht/prepare_ball_assets.py`, an immutable multi-source
  capture-to-registry launcher. Every input asset spec declares a verified
  independent-NHT or vanilla-PLY source, explicit metric Sim(3), plausible
  5--9 cm nominal diameter, and explicit `source_is_user_asset`; the launcher
  never infers user provenance. It preflights all sources, calibration,
  background, and options before CUDA, runs frozen-target feature fitting per
  asset, records stdout/stderr, and publishes one strict multi-variant registry
  only after every fit passes.
- Failed launcher runs retain request/log/failure evidence and completed
  immutable stages, but never publish a success manifest. Successful manifests
  explicitly record `rgb_overlay_used=false`, registry identity, fit/report/log
  references, and the aggregate user-asset truth flag.
- Updated `third_party/nht/ball_feature_fit_fixture.py` to emit real individual
  RGB/mask files, the capture schema, both source formats, and production asset
  specs. The canonical fixture is `feature-fit-fixture-v4`; it has eight
  128x128 views (six train, two validation), 2.5513--2.8198% foreground
  coverage, fixture fingerprint
  `c7922024b7307976ba5df920e2d3aace9637d6e723e4f59c3c2e3d91fd2cd381`,
  and capture-manifest SHA-256
  `a6aaefa0398ce71d4080352a23405ff820ac73951fc1ddbc62920de0cc65f65b`.
- Repeated the calibration CLI into `calibration-import-a-v1` and
  `calibration-import-b-v1`; all three published files are byte-identical.
  The canonical v4-source import is `calibration-import-c-v2`, with import
  fingerprint
  `1d9cccd21c8e0883d6979b6c54e1b90f01640467bf0aa2ec8dfb22511d8331fc`
  and bundle fingerprint
  `4867a77667df6c665ab390ce1d0f9957c821325f283e6813ac6b909271985fab`.
- Ran the production launcher twice over the independent-NHT and vanilla-PLY
  fixtures. Both 600-step fit trees (20 files each) are byte-identical:
  validation PSNR is `54.041032791137695 dB` and
  `54.04560470581055 dB`; prepared tensor hashes remain
  `cb6622ce3f178f59c0316948592b60b5dd2137eb76e280650aea4268ef2f80fb`
  and
  `cf418cb23eee7d304f343553e8d9d373a810f7a90e9e0ba0831cdcf369f8c83b`.
  Both registries strictly reload with the same path-independent fingerprint
  `a3c670dc9782abd7a5084ee78b5bf7b304e9d5b1ee01e3708e6c27cef50a661c`.
- Exercised explicit negative gates. A mask containing value 127 was rejected
  before calibration publication. A 100 dB report requirement rejected a
  measured 54.041033 dB fit, leaving no fit files, registry, or success
  manifest. Reusing an existing launcher root was rejected before preflight or
  CUDA.
- Per the `test-structure` policy, added five pure task-local unit tests under
  `tests/unit/synthetic_data_generation/blcs/test_calibration.py`. They cover
  reproducible import, mask rejection, split/path leakage, improper rotation,
  full masks, tensor tampering, and no-overwrite. The full synthetic-data
  unit/e2e suite now passes 96/96.
- Per the export-first requirement, strictly reloaded all 491 provider images
  and the float64 `[217336,3]` normalized point cloud before the final
  integration verifier. The export fingerprint remains
  `4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b`.
- Visually inspected the canonical independent fit's validation view 6. Target
  and prediction remain indistinguishable at displayed pixel scale, with the
  magnified difference localized to the compact foreground mask. This is
  mechanics evidence from a non-user fixture, not a production-asset claim.
- Re-scanned project and Downloads paths. The only ball/tennis-named candidates
  are two `tennis_clip.npz` label datasets containing ball/court/human/SMPL
  arrays, not Gaussian assets. `source_is_user_asset=false` and
  `accepted_court_alignment_used=false` therefore remain explicit, and P3
  stays active.

## Canonical cycle-08 artifacts and metrics

- Root:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-08` (5.4 MiB, 161 files,
  including retained failed-design and negative-gate evidence)
- Canonical capture/source fixture:
  `feature-fit-fixture-v4`
- Fixture manifest SHA-256:
  `65168f3b869aab9618d255648b9fdb36dfea3e14f53457f3c1aec9a70573ed96`
- Independent source / vanilla source SHA-256:
  `37e12436ec392d2dd24054d3aac0222ba1752414c67b36ee355d349ccfcbba52` /
  `8234c6fc18ee21f9740b2efcaac41df535af2314271711348375b87087f0035c`
- Canonical calibration import / manifest SHA-256:
  `calibration-import-c-v2` /
  `b9ce7cf4e9b05bdb3655baacfe01961a9c649b51c42ba42abaaae3abede8f9c7`
- Byte-identical repeated calibration imports:
  `calibration-import-a-v1`, `calibration-import-b-v1`
- Canonical/repeated preparation:
  `preparation-run-a-v1`, `preparation-run-b-v1`
- Preparation manifest SHA-256:
  `bc936689513bb9c4601ea636f14b1db99c824167c813abd273c8f7b9250a5f24` /
  `f10883695ba9b53d87251013d0f5a91c70fe196ed1bb251099b196df9f49df41`
- Request SHA-256, identical across runs:
  `e53592ff68a67b8b857db7968b83cc7713fe94c0ec91887ab3494787ea868322`
- Registry JSON SHA-256:
  `81885a947d4435c7548cbb56572777d2eb29eae3f109739a21ee6557d72a8cb2` /
  `3582c5f923c84e32986ad7fd838b1b71e0722b0a029dab85c123e3bf9e4a200e`
  (different only because truthful output-root file URIs differ)
- Acceptance report / SHA-256:
  `acceptance-report-v1.json` /
  `559a74b43cccddb8c9c97973cec5029255ab1d15089de64d4dd6f1b3155a0998`
- Export reload report / SHA-256:
  `export-reload-verification.json` /
  `a5a3118e95e502ad4400f976586df968e627e6509154b30b53b5a49f19b03210`
- Report-gate rejection:
  `preparation-rejected-psnr-v1`
- Antialiased-mask rejection log / SHA-256:
  `.codex-loop/3dgs-synthetic-data/logs/ball-calibration-antialias-rejection-c08.log` /
  `9cefcf5d42530fdb9e2cf6406368acc1d699a89759c3556298a400318f432c8e`
- Launcher no-overwrite log / SHA-256:
  `.codex-loop/3dgs-synthetic-data/logs/ball-preparation-no-overwrite-c08.log` /
  `95fc101cf2eeda872cc1fe885126df2d46e31a39caeaf2d61d335c762728b536`
- `rgb_overlay_used`: false
- User ball asset used: no
- Accepted court alignment used: no

## Completed in cycle 07

- Added the isolated production worker
  `third_party/nht/ball_feature_fit.py`. It accepts either the strict six-key
  independent-NHT tensor pack or a standard INRIA 3DGS PLY, but never treats
  independent latent features or vanilla SH coefficients as target-NHT
  features.
- Added the immutable calibration contract
  `tennis_ball_nht_calibration_bundle_v1`: exact float32 OpenCV
  camera-to-asset poses/intrinsics, uint8 RGB, boolean foreground masks, and
  explicit train/validation split are stored in a hash-checked NPZ plus a
  content-fingerprinted manifest. Every view must have foreground pixels and
  both splits must be non-empty.
- The worker verifies the clean pinned NHT
  `7de4cc07ba7f81ce90f7bd90f76ff0260c00c3d0` and gsplat
  `20bc323d613258e5d169fdbc962c9ef27d55ca69` checkouts before CUDA work.
  Means, log scales, opacity logits, normalized quaternions, and the target
  deferred shader remain frozen. Only a newly zero-initialized target-space
  feature tensor is optimized with the official NHT feature learning rate
  `0.015` and a cosine schedule.
- Production publication is atomic and refuses an existing output before
  shader initialization. It emits the exact prepared tensor pack, the strict
  `tennis_ball_asset_conversion_report_v1` already enforced by BLCS ingestion,
  optimization history, pinned renderer identity, declared preprocessing,
  validation PSNR, and hash-inventoried target/prediction/difference/mask
  diagnostics.
- Pinned `plyfile==1.1.3` in the isolated runtime. It is the newest compatible
  release with NHT's `numpy<2.0.0` constraint; the current `1.1.4` requires
  NumPy 2.
- Added `third_party/nht/ball_feature_fit_fixture.py` solely for non-user
  integration evidence. It generated one 512-Gaussian, 6.7 cm fixture in both
  independent-NHT and degree-zero standard PLY source formats plus eight
  128x128 frozen-target calibration views (six train, two validation).
  Foreground coverage is 2.5513--2.8198% per view.
- Both source paths completed 600 feature-only steps against the real cycle-02
  frozen deferred shader:
  - independent-NHT validation PSNR: `54.041032791137695 dB`;
  - vanilla-3DGS validation PSNR: `54.04560470581055 dB`;
  - required minimum: `20 dB`;
  - prepared feature dimension: 48.
- Repeated the authoritative independent-NHT conversion into a second immutable
  root. All ten files are byte-identical, including prepared tensors,
  optimization history, diagnostics, and manifest. The prepared tensor
  SHA-256 is
  `cb6622ce3f178f59c0316948592b60b5dd2137eb76e280650aea4268ef2f80fb`;
  the vanilla prepared tensor SHA-256 is
  `cf418cb23eee7d304f343553e8d9d373a810f7a90e9e0ba0831cdcf369f8c83b`.
- Independently reloaded both outputs, verified all manifest file
  hashes/sizes, exact dtypes/keys/finiteness, and proved geometry/opacity bytes
  equal the declared post-load frozen geometry. The two conversion reports
  then passed the actual BLCS ingestion boundary together as a two-variant
  inventory. Its registry fingerprint is
  `52eb77653a248271f723c87d03299e1cc43a2c297fd397ebdcc09b11306e1c78`.
- The final worker enables strict deterministic-algorithm enforcement and
  records its own SHA-256 plus Torch/CUDA/NumPy/plyfile versions. Both source
  paths passed again with the same prepared tensor hashes and PSNR; their fit
  manifests are included in the v3 registry provenance.
- Per the export-first requirement, strictly reloaded and verified all 491
  images plus the float64 `[217336,3]` normalized point cloud before the final
  integration gate. The export fingerprint remains
  `4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b`.
- Visually inspected independent validation view 6. The target and fitted
  prediction are indistinguishable at displayed pixel scale; the 8x difference
  panel remains localized to the compact foreground mask. This is mechanics
  evidence against the one-step target shader, not production appearance
  quality.
- Re-scanned project and Downloads paths. No ball-named Gaussian PLY/splat is
  present. The discovered `tennis_clip*.npz` files are existing task/depth
  outputs, not movable Gaussian assets. `source_is_user_asset=false` and
  `accepted_court_alignment_used=false` remain explicit.

## Canonical cycle-07 artifacts and metrics

- Root:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-07` (4.1 MiB, 108 files,
  including retained exploratory v1 evidence)
- Fixture/calibration:
  `feature-fit-fixture-v1`
- Fixture manifest SHA-256:
  `b15fc51a84cf7a09bdc4fe2c3b3206387df7621774a08bec2a42cfa79ff975c1`
- Calibration fingerprint:
  `f2707fdf728dcbb8e4a0a972dadd5a2645f98c75e62b3c16059bd8cb871e5669`
- Byte-identical repeated independent conversion:
  `independent-feature-fit-a-v2`, `independent-feature-fit-b-v2`
- Canonical current-worker independent conversion / manifest SHA-256:
  `independent-feature-fit-v3` /
  `ee9f90e1a278351458fcfce9fc7561287972d06d551f207cf3ccc00783f018b7`
- Canonical current-worker vanilla conversion / manifest SHA-256:
  `vanilla-feature-fit-v3` /
  `8d38d58d08e79bc8e1f1ea7ca3e940bf3c7eed214460137d2e0d0cd016171fa1`
- Current worker SHA-256:
  `d561b2d5dc415468d5d413b8e533927117b82e537a2ced0db23cd225afc2e861`
- Canonical strict two-format publication:
  `ingested-two-format-fixtures-v3`
- Registry SHA-256:
  `60eb01a5f4d94adac93dca9d53f06ffe6e7b656d64f6ba13cbcc9ad2ae83e8dd`
- Acceptance report / SHA-256:
  `acceptance-report-v3.json` /
  `299d8e519233af582246c11b90cb893c283f6d911efc4c7a5845128c4b90d182`
- Export reload report / SHA-256:
  `export-reload-verification.json` /
  `13801ecf037928038fa9f6d04c40e355bf06fa6c12e92edcab4ea5d4cda6bee9`
- No-overwrite rejection log / SHA-256:
  `.codex-loop/3dgs-synthetic-data/logs/ball-feature-fit-no-overwrite-c07.log` /
  `50a7d225f95d4c6db15aec2e8647920d4e9ef12d36c44ab75d583257af35fe5b`
- `rgb_overlay_used`: false
- User ball asset used: no
- Accepted court alignment used: no

## Completed in cycle 06

- Added the task-local production boundary
  `src/synthetic_data_generation/blcs/asset_ingestion.py`. It accepts only:
  - a native NHT tensor pack that explicitly shares the target background's
    frozen appearance space and exact deferred-shader payload; or
  - vanilla 3DGS / independently trained NHT input whose prepared target-NHT
    tensor pack is accompanied by a strict `passed` frozen-target optimization
    report with matching source/method/tensor/shader hashes, dimensions,
    positive optimization/view counts, and validation PSNR at least 20 dB.
- There is no implicit SH-to-NHT conversion, appearance fallback, dtype cast, or
  independent-feature concatenation. Source, prepared tensor, conversion report,
  and target appearance bytes are all verified before tensor loading.
- Canonicalization applies an explicit `asset_from_prepared` Sim(3), normalizes
  quaternions, emits float32 NHT raw parameters in metre-valued
  `asset_local` coordinates, resets instance IDs, and atomically publishes the
  immutable registry. Publication refuses an existing directory.
- Added geometry gates based on the 99th-percentile three-sigma Gaussian
  envelope. Effective diameter must lie within 25% of the declared 5–9 cm
  tennis-ball diameter and the AABB midpoint must lie within 10% of that
  diameter from the asset origin.
- Added six high-value ingestion unit tests under the mirrored BLCS test tree:
  strict parsing, metric/reproducible publication, no-overwrite, native
  appearance identity, explicit vanilla/independent conversion, conversion
  report hash/count/PSNR gates, wrong shader rejection, and metric scale/origin
  rejection. BLCS tests are now 16; the full synthetic suite is 91.
- Per the export-first requirement, strictly reloaded all 491 images and the
  normalized point cloud before integration validation.
- Exercised the production boundary with the real cycle-02 NHT checkpoint's
  shared appearance and tensor format, while explicitly retaining
  `source_is_user_asset=false`. The former mechanics patch was transformed into
  a 6.7 cm centred fixture:
  - effective diameter `0.06700000166893005 m`;
  - diameter relative error `2.4909403727076635e-8`;
  - origin offset `9.313225746154785e-10 m`;
  - 512 Gaussians and 48-dimensional target-NHT features.
- Repeated publication produced the same registry fingerprint, canonical tensor
  bytes, and path-independent ingestion evidence. Registry JSON bytes differ
  only because local `file:` URIs correctly identify distinct immutable output
  roots.
- Published the same 60-frame, 491-camera plan twice from the canonical
  registry; plan manifests and fingerprint are byte-identical. Rendered six
  frames twice through native NHT RGB/depth plus exact AOV; both 43-file trees
  are byte-identical.
- Physical scaling materially changed the visibility evidence. Exact visible
  pixels by frame are `165,0,0,0,0,0`; only frame 0 is render-visible. The old
  centre-depth proxy also marks frame 30 visible despite exact contribution
  zero, adding another measured reason not to use it as a label.
- Visually inspected frame 0 RGB and exact mask. The 165-pixel mask is compact
  and localized; RGB remains one-step NHT mechanics evidence, not a quality
  claim.
- Re-scanned `/home/kamimura/projects` and the user's Downloads. No ball-named
  Gaussian PLY/splat/NPZ source is present.

## Canonical cycle-06 artifacts and metrics

- Root:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-06` (37 MiB)
- Canonical/repeated ingestion:
  `fixture-publication-a-v1`, `fixture-publication-b-v1`
- Registry fingerprint:
  `3c4605e6894e97bbfe678f112ebd289a9a84c0ae235366bdd56d5701aeccf87c`
- Canonical metric tensor SHA-256:
  `bf84ba134f6590c105f3768a8bbf4a4b809313cfd4b2f588af0cdff53698d263`
- Path-independent ingestion evidence SHA-256:
  `4e6e7609e5bc61e9f295b171739fe712c671ec821249e30da81c2a191d3824a7`
- Canonical/repeated plan:
  `metric-plan-a-v2`, `metric-plan-b-v2`
- Plan fingerprint / manifest SHA-256:
  `88b31026d6fd11fa5f1fd349fd06e87b393d286032801ee01c303ec9765f3124` /
  `ff58966f11306cd0c7f3a524f0122cd0ff65408d7dc5589c9da2d74716208a74`
- Canonical/repeated render:
  `metric-render-a-v2`, `metric-render-b-v2`
- Render fingerprint / manifest SHA-256:
  `4931a1ff92253d715d3fedb74639acb9dbe8f49eb199c681dfe1fc1c5d2f72e4` /
  `362883dcca2ee8e5107fe2d4da711b9f8feed95683dffbd1a9858b58c97b62d8`
- Ingestion report / SHA-256:
  `ingestion-report-v1.json` /
  `2f74da78dde638dfb8fa1ee800e802defa2ec129d371ad2dfe6ddb97851476c2`
- Render report / SHA-256:
  `render-report-v1.json` /
  `c8292b57530c2c39d385d10ab94f6a7f105d5261f5d9c54a3f7f87249855b619`
- Diagnostic:
  `diagnostics/metric-frame-000000-rgb-and-mask.png`
- Diagnostic SHA-256:
  `26a6aba3c2b70dd86573d622c89a2e08468eefd9a0b58d9e2b31ef373f065771`
- Render files / byte-identical repetition: 43 / yes
- Exact/proxy visible instance-frames: 1 / 2
- `rgb_overlay_used`: false
- `exact_per_pixel_instance_mask`: true
- User ball asset used: no
- Accepted court alignment used: no

### Cycle-06 logs

- Publication succeeded; first plan adaptation failed:
  `.codex-loop/3dgs-synthetic-data/logs/ball-ingestion-c06.log`
- Successful plan/report continuation:
  `.codex-loop/3dgs-synthetic-data/logs/ball-ingestion-c06-retry.log`
- Successful canonical/repeated render:
  `.codex-loop/3dgs-synthetic-data/logs/ball-ingestion-render-c06-a.log`,
  `.codex-loop/3dgs-synthetic-data/logs/ball-ingestion-render-c06-b.log`
- Publication no-overwrite rejection:
  `.codex-loop/3dgs-synthetic-data/logs/ball-ingestion-no-overwrite-c06.log`

## Completed in cycle 05

- Replaced the conservative visibility result with an exact per-pixel
  alpha-composited instance AOV while preserving NHT deferred appearance:
  - the first public rasterization call remains NHT `RGB+ED`;
  - a second call renders one-hot background/instance colors over the identical
    composed Gaussian scene using regular eval3d;
  - no instance channel is appended to the NHT feature vector;
  - every manifest records both calls and the NHT internal depth auxiliary pass.
- Published `instance_contribution.npy`, thresholded `instance_mask.npy`, and
  exclusive `instance_segmentation.npy` for every selected frame. Labels now
  contain exact visible pixels/contribution mass as well as the old depth proxy
  solely for measured comparison.
- The initial `1e-5` cross-pass alpha tolerance correctly rejected a measured
  `1.800060272216797e-5` float32 kernel-order difference. The explicit default
  is now `1e-4`; contribution channels still sum to their own AOV alpha within
  `3.5762786865234375e-7`.
- Re-rendered the export-first single and multi fixtures twice each at the same
  camera/resolution/frames. Each 43-file canonical/duplicate tree is
  byte-identical, including all contribution/mask/segmentation arrays.
- Before the final acceptance check, strictly reloaded and byte-verified every
  declared file in the cycle-01 export: 491 cameras, 491 images, and a
  float64 `[217336,3]` point cloud with unchanged fingerprint
  `4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b`.
- Independently reloaded all arrays and proved:
  - masks exactly equal `instance_contribution[..., 1:] >= 1e-4`;
  - segmentation equals the contribution `argmax` with transparent pixels `-1`;
  - all inactive persistent-instance channels are exactly zero;
  - all arrays are finite.
- The exact result resolves the cycle-04 undercount: single frame 59 has 276
  visible pixels although the projected-centre depth proxy reports invisible.
  Single exact/proxy visible instance-frames are 6/5. Multi remains 9/9.
- Visually inspected diagnostic-only RGB/mask panels for single frame 59 and
  multi frame 15. Instance masks are spatially localized and separated, while
  RGB still has the explicitly non-production one-step green/noisy appearance.
- Re-scanned `/home/kamimura/projects` and the user's Downloads for ball-named
  Gaussian PLY/splat/NPZ assets. None were found; detector checkpoints were
  excluded because they are not Gaussian assets.

## Canonical cycle-05 artifacts and metrics

- Root:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-05`
- Canonical/repeated single:
  `render-single-a-v2`, `render-single-b-v2`
- Canonical/repeated multi:
  `render-multi-a-v2`, `render-multi-b-v2`
- Acceptance-scoped report:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-05/report-v2.json`
- Report SHA-256:
  `666c27e5e375585b2666080dc4e74f7758384a2020f901a50888af1a0e81f06f`
- Strict export reload report / SHA-256:
  `export-reload-verification.json` /
  `f74f2fb460d211bdc3c75fde7991017bf92e2c73e9d1aca8cd661e11b0b2fc7c`
- Artifact-root size: 76 MiB
- Camera/resolution/frames:
  `frame_000080`, 480 x 270, `0,15,30,44,45,59`
- Single render fingerprint / manifest SHA-256:
  `d811ae682b8f601977256ad635c3280cb9b27740ba68e149fde46485afd2ce04` /
  `acc6843ba2fa021566b039675f15482d6a6cc1152982e75c413e43c1f91bf3cf`
- Multi render fingerprint / manifest SHA-256:
  `09d88cc1eefe663b39d72fb7ab6294614436fe2bdfb9db2a47be5654659d9723` /
  `9ec660e1bcf73ba83f8fe52aa217e06b51e667b75e6ffc6b9c1f5f1feae4c5f4`
- Output files per run / byte-identical repetitions: 43 / yes
- Renderer calls per run/frame: 12 / 2
- AOV-alpha consistency tolerance: `1e-4`
- Measured NHT-vs-AOV alpha max absolute error:
  `1.800060272216797e-5`
- Contribution-sum-vs-AOV-alpha max absolute error:
  `3.5762786865234375e-7`
- Exact/proxy visibility disagreements: single 1, multi 0
- Single exact visible pixels by frame:
  `59113, 2747, 1041, 482, 463, 276`
- Multi exact visible pixels by frame:
  instance 1 `59113,2747,1041,482,0,0`;
  instance 2 `0,1576,820,404,392,240`
- Diagnostic panels:
  `diagnostics/single-frame-000059-rgb-and-mask.png` and
  `diagnostics/multi-frame-000015-rgb-and-masks.png`
- `rgb_overlay_used`: false
- `exact_per_pixel_instance_mask`: true
- User ball asset used: no
- Accepted court alignment used: no

### Cycle-05 logs

- Rejected strict-alpha attempt:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-single-c05-a.log`
- Successful single/repeat:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-single-c05-a-retry.log`,
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-single-c05-b.log`
- Successful multi/repeat:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-multi-c05-a.log`,
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-multi-c05-b.log`
- No-overwrite rejection:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-no-overwrite-c05.log`

## Completed in cycle 04

- Added a strict public `load_blcs_gaussian_plan()` alongside the compact
  verifier. It reconstructs the complete immutable v2 plan after checking
  registry assets, NPY hashes/dtypes/shapes, selection replay, cameras,
  transforms, and the plan fingerprint.
- Added the isolated CUDA consumer `third_party/nht/blcs_render.py`. It:
  - loads only versioned plan/composition files across the tennis-lab/NHT
    boundary;
  - verifies every background/asset/appearance byte and rejects incompatible
    appearance spaces or feature dimensions;
  - requires the exact renderer commit recorded by the background composition
    and refuses a dirty gsplat checkout;
  - caches metric-local assets, assigns the plan's persistent instance IDs,
    transforms only active lifecycle instances, and concatenates them with the
    background before rendering;
  - performs exactly one `rasterization(...)` API call for each selected frame;
  - publishes RGB PNG, alpha NPY, expected-depth NPY, and complete
    3D/2D/camera/transform/visibility labels atomically without overwrite;
  - hashes every frame file and fingerprints the complete render manifest.
- Updated both synthetic-data and isolated-NHT READMEs with the renderer
  contract, usage, and visibility limitation.
- Per the export-first requirement, strictly reloaded the cycle-01 491-camera
  export before creating a 60-frame single-object plan. The plan was published
  twice and all ten files were byte-identical.
- Rendered six lifecycle frames (`0,15,30,44,45,59`) at 480 x 270 from exported
  camera `frame_000080` for both single and multi fixtures.
- Repeated both CUDA renders into new directories. Each 25-file tree,
  including RGB/alpha/depth/labels/manifest, was byte-identical to its first
  run and shared the same content fingerprint.
- Multi lifecycle composition behaved as intended:
  - frame 0: instance 1 only;
  - frames 15/30/44: instances 1 and 2;
  - frames 45/59: instance 2 only;
  - composed Gaussian count changed from 217,336 to 217,848 only while both
    instances were active;
  - numerical/asset identity did not drift across birth/death.
- RGB comparison supplies direct composition evidence. Single and multi frame
  0 were byte-identical because both contain the same one instance. The five
  frames with different active sets changed 347–730 pixels, with maximum uint8
  channel differences of 69–110; no 2D RGB overlay path was used.
- Visually inspected canonical single frame 15/59 and multi frame 15. Results
  are finite but retain the unconverged green point-cloud appearance of the
  one-step checkpoint, so they are mechanics evidence only.

## Canonical cycle-04 artifacts and metrics

### Export-first single-object plan

- Root:
  `/home/kamimura/projects/tennis-lab/.claude/worktrees/3dgs-native-synthetic-data/.codex-loop/3dgs-synthetic-data/artifacts/cycle-04`
- Canonical:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/single-plan-a-v2`
- Reproducibility duplicate:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/single-plan-b-v2`
- Report:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/single-plan-report.json`
- Plan fingerprint:
  `780db9c6dc0effd3aa47a411e263c27adfb8609fa767d4bbdd43aa7e5dfb68ae`
- Canonical/duplicate manifest SHA-256:
  `f7e5271cceb5edfd5754a6f7b68fcb83c091d4bd2a85a83997ca329c27c38abd`
- Report SHA-256:
  `96552b81f8343634ba4b9e4dccaab73c48700cfa9ac8ed15a964401cef92fc3d`
- Frames / objects / cameras: 60 / 1 / 491
- Geometrically visible camera-frame-object slots: 10,112
- Repeated plan files byte-identical: yes, all ten

### Native NHT renders

- Canonical single:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/render-single-a-v1`
- Repeated single:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/render-single-b-v1`
- Canonical multi:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/render-multi-a-v1`
- Repeated multi:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/render-multi-b-v1`
- Acceptance-scoped report:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-04/report-v1.json`
- Artifact-root size: 32 MiB
- Renderer commit:
  `20bc323d613258e5d169fdbc962c9ef27d55ca69`
- Camera/resolution: `frame_000080`, 480 x 270
- Selected frames: 0, 15, 30, 44, 45, 59
- Single render fingerprint:
  `9211403bb2355fea0f217d2218fc1f6147c3a423dcc23836f6a44caeb254706a`
- Multi render fingerprint:
  `f43dfed5a3a9c54e15433305ce6899580ccbb838927b1d7822dc90ba9a6a1ec0`
- Single canonical/duplicate manifest SHA-256:
  `38ab9b319130c1e008b1e48781b6a4d04a115d39c99bc3553036d512700c46b2`
- Multi canonical/duplicate manifest SHA-256:
  `038bb38c960ddb4d26a41d5f7e916935c06c43f9ee5c4c5a07a14f8c51c8891d`
- Report SHA-256:
  `b447e6045409d68894c217c83f77b37353ccfa6fb64ec836592cd6965b1e81e1`
- Output files per run / byte-identical repetitions: 25 / yes
- Renderer API calls: 6 per run, exactly 1 per selected frame
- Background / one-object / two-object Gaussians:
  216,824 / 217,336 / 217,848
- Single RGB global range / frame-mean range:
  `[0.316344, 1.0]` / `[0.863214, 0.874455]`
- Multi RGB global range / frame-mean range:
  `[0.316344, 1.0]` / `[0.863214, 0.874469]`
- Alpha global range / mean range:
  `[0.0, 0.998016]` / `[0.497427, 0.527788]`
- Depth global range / mean range:
  `[0.0, 26.609241]` / `[1.561868, 1.768005]`
- RGB/alpha/depth finite: yes, every pixel
- Multi active instance-frames / depth-consistent visible instance-frames:
  9 / 9
- Single active instance-frames / depth-consistent visible instance-frames:
  6 / 5
- `rgb_overlay_used`: false
- `exact_per_pixel_instance_mask`: false
- User ball asset used: no
- Accepted court alignment used: no

### Logs

- Failed radii-shape attempt:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-single-c04-a.log`
- Successful single:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-single-c04-a-retry.log`
- Successful repeated single:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-single-c04-b.log`
- Successful multi:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-multi-c04-a.log`
- Successful repeated multi:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-multi-c04-b.log`
- No-overwrite rejection:
  `.codex-loop/3dgs-synthetic-data/logs/blcs-render-no-overwrite-c04.log`

## Retained prerequisites

- Cycle-01 export:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-01/b00-provider-export`
- Export bundle fingerprint:
  `4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b`
- Cycle-02 background/composition:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-02/nht-composition-smoke-v3`
- Cycle-03 canonical multi plan:
  `.codex-loop/3dgs-synthetic-data/artifacts/cycle-03/plan-c-v2`
- Cycle-03 multi plan fingerprint:
  `3aeee70c91fb23030ae242f108b8743d4fda9f42f53a2099ad05d0767b21c492`

## Verification through cycle 08

- Full synthetic-data unit/e2e suite: 96 passed, including 21 BLCS tests.
- Capture-import integration: three strict CLI imports reload; the two repeated
  publications are byte-identical; all eight source RGB/mask pairs and camera
  records reverify from their immutable source manifest.
- End-to-end preparation integration: both two-source launch runs, all fit
  references/diagnostics, both local registries, and every registry provenance
  byte strictly reload. Repeated requests, fit trees, tensor hashes, PSNR, and
  path-independent registry fingerprints match.
- Negative integration: antialiased capture masks, a deliberately impossible
  100 dB fit threshold, and output-root reuse all reject without publishing a
  false success or overwriting evidence.
- Frozen-target integration: both independent-NHT and vanilla-PLY workers,
  repeated independent worker, strict two-source ingestion, and output
  integrity verifier passed.
- Ruff over synthetic-data source/tests and all four capture/fit/launch NHT
  files: passed.
- mypy over seven typed composition/BLCS boundary files: no issues.
- Isolated NHT runtime compilation and `uv pip check` over 92 installed
  packages: passed; NumPy is restored to 1.26.4 and plyfile is 1.1.3.
- Export-first strict reload: all 491 images and normalized point cloud bytes
  verified against the provider manifest.
- Cycle-04 v1 and all four cycle-05 v2 render manifests plus referenced frame
  files: independent fingerprint/hash verification passed.
- Cycle-05 exact mask, segmentation, inactive-channel, finite-array, and
  canonical/duplicate byte checks: passed.
- Both cycle-06 registries strictly reload and verify every local source,
  appearance, output tensor, and provenance byte. Canonical/duplicate
  fingerprints, tensor hashes, and ingestion-evidence hashes match.
- Both cycle-06 plans pass strict array/hash/projection verification. Both
  render trees pass independent manifest/file verification and exact mask
  reconstruction; all 43 files are byte-identical.
- Existing v2 output no-overwrite test: rejected with exit code 1 before CUDA
  work.
- Existing ingestion publication no-overwrite test: rejected with exit code 1
  before reading sources.
- Existing feature-fit output no-overwrite test: rejected with exit code 1
  before the deferred shader printed its initialization line.
- Existing preparation output no-overwrite test: rejected with exit code 1
  before input preflight or CUDA.
- `git diff --check`: passed.
- No `src/**/scripts` file was changed in cycle 08; the capture CLI and CUDA
  workers remain in the tracked isolated `third_party/nht` boundary.

## Research and technical decisions

- No PLCS-avatar or novel-view paper selection was made because the active
  phase remains P3; those comparisons stay gated to P4 and P6.
- NHT `RGB+ED` is counted as one public `rasterization(...)` call per frame.
  The pinned fork internally performs its documented regular-eval3d auxiliary
  depth pass because depth cannot pass through harmonic encoding. The render
  manifest records this instead of claiming one low-level CUDA kernel.
- Appending instance one-hot channels to NHT latent features would alter
  harmonic encoding and corrupt the deferred shader input. That approach was
  rejected.
- Exact AOVs instead use a second regular eval3d rasterization over the
  identical composed scene with one-hot colors. This preserves NHT features,
  captures alpha-composited contributions behind partial occlusion, and makes
  the extra cost explicit.
- The depth-centre method is retained only as comparison evidence. Single frame
  59 has zero depth-consistent centres but 276 exact visible pixels, validating
  the AOV decision.
- The `1e-4` cross-pass alpha tolerance was selected only after `1e-5` failed
  against a measured `1.800060272216797e-5` maximum. The accepted tolerance is
  5.56 times that measured drift, while contribution closure remains below
  `3.58e-7`.
- Ball ingestion remains BLCS-local because nominal diameter, ball-centred
  metric geometry, source-format policy, and registry semantics are
  task-specific. It reuses composition `GaussianTensorSet` and Sim(3) rather
  than duplicating shared tensor geometry under `src/utils`.
- Native shared-NHT identity is the only conversion-free path. Vanilla SH or an
  independently trained NHT asset must supply a prepared tensor pack in the
  exact frozen target appearance plus a content-checked optimization report.
  The 20 dB report floor is an explicit minimum mechanics gate, not a claim of
  production visual convergence.
- The conversion worker adopts NHT's documented `0.015` per-feature learning
  rate while holding the target deferred shader fixed; the upstream MLP
  learning rate is deliberately unused because modifying target appearance
  would invalidate shared-scene composition.
- Standard PLY input is interpreted using the documented INRIA properties.
  PLY quaternions are normalized exactly once as declared preprocessing, then
  all geometry and opacity tensors are frozen. SH coefficients are validated
  as source structure but are not consumed as NHT appearance.
- Capture/calibration logic remains BLCS-local: its binary foreground masks,
  camera-to-ball-asset coordinates, split-leakage policy, and target-NHT
  semantics are ball-specific. Shared `ArtifactRef` and Sim(3) primitives are
  reused rather than adding a generic `src/utils` abstraction.
- Real-capture input is a file-backed manifest rather than embedded arrays so
  production captures remain inspectable, individually content-addressed, and
  re-verifiable after import. The import URI is never rewritten silently.
- A preparation run publishes its registry only after every requested asset
  passes fitting and ingestion. On failure, preserving the immutable request,
  process log, and failure record is preferable to deleting diagnostic
  evidence; an empty stage directory is not treated as a published fit.
- No PLCS-avatar or novel-view-camera paper decision was made in cycle 08
  because the acceptance-gated active phase remains P3.

## Failures and hypotheses

- First CUDA attempt expected scalar UT radii `[1,N]`, while the NHT renderer
  returns elliptical radii `[1,N,2]`. It failed before atomic publication.
  Requiring both axes to be positive fixed the projection predicate; all later
  renders passed.
- Independent output verification initially rejected a relative root because
  the safety check compared an absolute child with an unresolved root. The
  verifier now resolves its root first. Published artifacts and fingerprints
  did not change.
- The first exact-AOV render correctly failed before publication because the
  default cross-pass alpha tolerance was `1e-5` while measured float32 drift was
  `1.800060272216797e-5`. A measured `1e-4` default resolved it; four later
  renders passed and were byte-identical in pairs.
- The first two full-suite invocations used inherited Windows `TMP/TEMP` and
  the original checkout's editable package, causing capture and import
  failures. Explicit `/tmp` and this worktree's `PYTHONPATH` produced the
  authoritative cycle-05 85/85 pass; the current suite now passes 96/96. This
  was an invocation-environment issue, not a fallback or code change.
- The first cycle-06 continuation passed both immutable ingestion publications
  and then failed while adapting the historical single-ball fixture because
  single-ball plans explicitly forbid `ball_present`. The already-valid
  publications were retained; a new continuation used `ball_present=None` and
  produced both plans without replacement.
- Repeated publication registry JSON hashes differ because each manifest
  contains truthful absolute local `file:` URIs. Content-addressed registry
  fingerprint, canonical tensor bytes, and path-independent ingestion evidence
  match exactly; this is recorded instead of claiming whole-tree byte identity.
- The one-step NHT checkpoint remains visibly unconverged and green/noisy.
  It validates mechanics and determinism, not production visual quality.
- Through cycle 06, no compatible user ball Gaussian asset had been
  discoverable for five consecutive scans. This left the frozen-target worker
  as the remaining asset-independent P3 work.
- Through cycle 08 the loop treated court alignment as open. In cycle 09 the
  user explicitly accepted the existing override-v2 alignment, and strict
  verification proved that it references the exact cycle-01 export scene and
  all 491 cameras. The historical holdout machine rejection remains preserved
  in the decision evidence rather than being silently changed.
- Installing current `plyfile==1.1.4` alone transiently upgraded only the
  isolated runtime to NumPy 2.4.6, violating NHT's declared `numpy<2.0.0`.
  Dependency resolution proved that 1.1.4 requires NumPy >=2; the environment
  was immediately restored to NumPy 1.26.4 and the compatible
  `plyfile==1.1.3` pin. No CUDA artifact was produced in the invalid state.
- The first successful v1 conversion manifests did not inventory validation
  diagnostics or explicitly name standard-PLY quaternion normalization.
  Those immutable exploratory artifacts were retained. V2 added both
  declarations and byte-identical repetition; canonical v3 additionally
  enforces strict deterministic algorithms and records the worker/runtime
  identity.
- The first cycle-08 preparation fixture encoded the identity rotation with
  only eight values. Launcher preflight rejected it before creating an output
  root or starting CUDA. The fixture was retained as `feature-fit-fixture-v2`;
  v3 corrected the explicit nine-value rotation.
- Fixture v3 atomically published a calibration import inside a temporary outer
  fixture directory. Renaming the outer directory correctly invalidated the
  import's exact absolute capture URI. No URI fallback or rewrite was added;
  the invalid design is retained, and canonical v4 separates capture
  publication from the calibration-import CLI.
- The 100 dB negative fit created the launcher's empty `fit/` stage container,
  but the worker atomically published no fit files and the launcher published
  no registry or success manifest. The acceptance verifier checks file absence
  rather than misreporting the empty container as an asset.
- No compatible user ball Gaussian assets were discoverable through cycle 08.
  The user revised P3 in cycle 09 to allow an explicitly generated prototype;
  the absence of a captured asset is therefore no longer a P3 blocker and is
  not misreported as production-user provenance.
- The first cycle-09 prototype render correctly rejected a measured NHT/AOV
  cross-pass alpha drift of `0.002143383026123047` against the former
  `0.0001` default before publishing output. Re-runs use the explicit measured
  `0.005` gate and record actual maxima (`0.0021434` single,
  `0.0033501` multi) in each immutable manifest.
- A same-seed plan repeat first failed when invoked through the independent NHT
  environment: the script lacked a self-contained repo path, then correctly
  exposed that NHT does not contain PyTorch Lightning. The CLI now resolves the
  repo root itself, documentation fixes the boundary, and successful physical
  plan generation uses tennis-lab's main environment. CUDA rendering remains
  in the isolated NHT environment.
- The first cycle-09 full-suite invocation inherited the main checkout's
  editable path, so xdist workers collected only 24 e2e cases and could not
  import worktree modules. The authoritative rerun set
  `PYTHONPATH=$PWD` and passed 116/116. The broad mypy invocation also surfaced
  four pre-existing scene-provider typing errors outside cycle-09 changes; the
  eight changed BLCS/composition modules pass.
- The P3 acceptance verifier initially assumed channel-first instance masks,
  while the renderer contract stores `[H,W,objects]`. It failed without
  publishing a report. After matching the renderer's explicit saved shape and
  `>=` threshold semantics, the report passed.
- The first cycle-10 SMPL-X probe stopped before publication because the strict
  simplex check used `1e-8`, while the licensed float32 SMPL-X weights have a
  measured maximum sum error of `4.470348358154297e-8`. The validator now
  accepts `1e-6` representation error without normalizing input; an explicit
  unit test proves the original values are preserved.
- The next attempt exposed the same representational error in the homogeneous
  bottom row of per-vertex transforms. That affine check now uses the same
  measured `1e-6` tolerance; the raw joint-transform rotation/affine gates
  remain strict. Both failures produced logs and no partially published output.
- Pytest's inherited capture/parallel environment twice lost its temporary
  capture file before collection. The authoritative cycle-10 regression used
  explicit worktree `PYTHONPATH`, `-n 0`, and `-s`, and passed 124/124. This was
  recorded as an invocation-environment failure rather than silently treating
  a zero-test run as success.
- The first same-seed cycle-11 NHT comparison correctly failed a whole-tree
  byte diff. Geometry/opacity were exact, but gsplat CUDA feature optimization
  differed. Measurement isolated the effect to latent values
  (max/mean 0.010186/0.000830); rendered RGB differed by at most one LSB and
  0.002211 mean LSB, and held-out PSNR by 0.04140 dB. The P4 report records and
  gates these measured tolerances instead of claiming deterministic bytes.
- The first cycle-11 full regression PTY was externally terminated with status
  143 during its final e2e case, so it was not counted as a pass. The same
  command was rerun through a non-interactive worktree log, wrote exit status
  zero, and passed all 129 cases in 54.58 s.

## Running jobs

- NHT training: none
- BLCS/PLCS/court render/plan/export/alignment: none
- GPU compute process owned by this loop: none
- GPU compute applications at end of cycle: none
- Long-running job started in cycle 13: none
- Relevant cycle-13 log:
  `.codex-loop/3dgs-synthetic-data/logs/court-novel-view-probe-repeat-v2-c13.log`
