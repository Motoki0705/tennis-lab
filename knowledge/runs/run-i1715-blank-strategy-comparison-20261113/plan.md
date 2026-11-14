# clip000 blank annotation comparison plan

This plan was written before any model call. It is a development comparison for
issue #1715, under parent #1968. The comparison has **not started**. The input
preflight is blocked by missing reference assets. No policy is selected.

## Fixed experiment

- Use the original bytes of `data/samples/cam0.mp4`, `cam1.mp4`, and `cam2.mp4`.
  The trusted identity is [the original input record](../run-ball3d-evaluation-foundation-20261103/inputs.json).
  Check all 1010 presentation frames, 1920 × 1080 pixels, and nominal 59.94006 fps.
  Preserve decoded integer PTS; do not generate PTS from the rounded frame rate.
- Development scope: original frames **[0, 64)** for each of the three cameras.
  Choose this prefix without inspecting reference coordinates or ball predictions.
  Give each worker only its own camera and these 64 frames, including its support
  frames. Do not silently supply the rest of the clip as context.
- Four conditions, as serialized in [policies.json](policies.json): temporal
  overview, full frame followed by zoom, sparse anchors with inference, and free
  adaptive exploration. Use the existing common instructions and tools without
  changing label semantics or requiring adjacent frames to be viewed in order.
- One independent run per policy and camera: 4 × 3 = **12 camera runs**, 768
  camera/frame decisions. There are no additional seeds, replications, pilots,
  or best-of retries in this plan. Run conditions in the order in `policies.json`;
  allow one active camera worker at a time to limit local resource contention.
- Worker: `worker-decision` returns **gpt-6.1-sol / max**. Pin code, tool version,
  policy, input, runtime and scope before registration. Each camera attempt is
  **5400 seconds**; valid timed-out progress continues in its own workflow.
  Persist every attempt, including failures. Do not merge labels between runs.
- Register a source plan without launching it. Use `blank-register` to share
  immutable images and isolate each condition's worker/revision ledger.
  `publish_dataset=false` for every comparison condition. No detector, GPU work,
  additional learning, manual label edits, or dataset publication is included.
- The comparison consumes part of **clip000**. A later 3030-frame acceptance run
  must be described as **tuned on clip000**, never as an unseen evaluation.

## Forecast and resource gate

[estimate.json](estimate.json) is a planning scenario, not measured usage or an
authorized resource budget. It uses the historical prediction-review estimate
recorded in the [annotation workflow skill](../../../.agents/skills/annotation-workflow/SKILL.md):
about 16 billion input tokens for 500 clips of 780 frames, and 90 worker minutes
per 780-frame clip. This is a deliberately explicit proxy; the white annotation
policies, current tools and context growth have no measured cost yet.

The forecast adds 20,000 input tokens and 60 seconds per camera run for startup.
Output tokens are assumed to be 5% of input, with 0–20% sensitivity reported in
the JSON. These assumptions are estimates, not measurements or token limits.
Cached input is part of input and is never added a second time. Comparison and
acceptance use the same assumptions. Their forecast ratio is approximately
**25.53%**, below the parent's **50%** ceiling. Startup is charged 12 times for
comparison and three times for acceptance; frame-count ratio alone is not used.

The sequential comparison forecast is about **101 worker minutes**, and the
acceptance forecast about **353 worker minutes**. Both remain uncalibrated. The
90-minute attempt limit is not a forecast of every attempt's duration. If a
revised comparison forecast exceeds three hours, detach the controller and
record its process identity plus completion/failure events for the parent.

Before any real run, restore and validate the reference JSON and its six CSVs.
Then measure native JPEG sizes and local free space, choose explicit stage costs
and an approved resource scope, and save the immutable configuration. Do not copy
fixture budgets, historical campaign budgets, or the forecast assumptions into
a new resource scope. No new scope or runtime was created in this turn.

## Measurements and interpretation

Score the completed saved v3 annotations against the pinned user reference using
the official reader. References are parent-only scoring inputs and never enter
worker tasks, images, policy text, handoffs or accessible filesystem roots.

For each condition and camera, retain:

- Every response ID, attempt/thread ID, input/output/cached-input token usage,
  and provider total when present. Also calculate input + output when both are
  known; cache is a subset of input. Missing usage stays null, with known
  subtotals and coverage reported separately. A missing total cannot be ranked
  as cheaper. Do not divide an incomplete subtotal by usable coordinates and
  present it as a complete efficiency estimate.
- Complete totals and total tokens per usable coordinate; all target rows,
  processed/unprocessed, final unknown, observed/inferred coordinates,
  coordinate-unavailable rows, and state/occlusion counts. Divide by the saved
  annotation's usable-coordinate count; a zero denominator yields null.
- Native pixel Euclidean errors on **paired usable coordinates**, with paired
  count and mean/median/p90/p95/max. Report directly observed reference points
  and accepted reference estimates separately, and retain camera/frame/PTS keys.
  Also compare policies on common paired frames so differing coverage cannot
  silently improve conditional error. Excluded extrapolations remain excluded.
- False absence: predicted `no_active_ball` or `out_of_frame` when the reference
  has a usable coordinate. Report numerator and reference-usable denominator.
  Reference unknown/occluded/excluded rows do not establish true absence. Missing
  coordinates from other states are also retained in coordinate-unavailable
  counts, rather than hidden behind this narrower false-absence metric.
- Committed image requests, requested source-frame tiles, rendered image
  presentations, requested-frame revisits and presented-frame revisits;
  attempt count, continuation history, failure reasons and wall time.

Use `blank-compare` for the canonical response/camera report, with additional
parent-only scoring summaries for distributions, common pairs and false absence.
Never make zero usage or zero error stand in for a missing run/reference.

Select after inspecting the joint quality/cost results, including coverage,
false absence and error tails, rather than unknown count alone. Describe
Pareto tradeoffs and any unresolved tradeoff explicitly. Do not introduce a new
token-reduction target or pixel-error pass/fail threshold. If measurements or
reference comparisons are incomplete, leave selection pending.

One prefix and one repetition cannot establish full-clip or cross-recording
quality, behavior at later occlusions/bounces, order effects, seed variance, or
the ability to satisfy the full acceptance counts. Negative or inconclusive
results remain evidence; do not add runs to obtain a preferred outcome.

## Acceptance handoff

The 3030-row acceptance workflow is **not authorized for launch in this turn**.
After comparison and selection, freeze its policy, full [0, 1010) input scope,
runtime, approved scope, publication request and holdout-preserving split. Use
the [existing scene entry point](../../../src/tennis_scene/pipeline/direct_annotation/README.ja.md#一連実行),
with the original fixed bundle and explicit `verified_calibration_frame0`,
`publish_dataset=true`, CPU-only ball reconstruction, and automatic 5400-second
camera-attempt continuations. The request and bundle digests must come from
validated saved files; there is no runnable acceptance command yet.

The parent must issue a later launch instruction. Then save v3, reread it,
triangulate with fixed geometry, publish/reread the dataset and preserve all
receipts and 2D/3D comparison videos. The existing acceptance is unprocessed=0,
coordinate unavailable ≤81 and final unknown ≤25 across all 3030 rows. No
camera-specific limits or extra accuracy gates are added.

Monitor a launched controller through its recorded PID/start time and terminal
event, not repeated progress polling. Confirm successful completion from saved
controller state and all camera completion receipts; exit code 0 alone is
insufficient. A dead controller with incomplete receipts is an abnormal stop.
There is no running controller, run directory or process ID to hand off now.

## Verification design

No production implementation or existing tests are changed. Validate the four
policy objects with `AnnotationPolicy`, the actual media with the published
sample verifier and `probe_video`, and the input gate by its recorded nonzero
exit and all missing-file statuses. Fixed geometry is an acceptance-only
requirement; its absence alone must not block a 2D comparison after the reference
has been restored.

Reuse the existing integration node
`tests/integration/tasks/annotation/test_blank_benchmark.py::test_two_strategies_share_exact_inputs_but_not_progress_and_compare_usage_and_error`
to check camera input equality, separate progress/ledgers, missing usage and
response/error correspondence. It runs real CPU services with a fake model and
synthetic usage; it is not a real GPT trial. Collect that one node first, then
measure setup/call/teardown with the required `pytest_locked.sh` wrapper and four
pytest processes. Do not run the repository suite locally or add tests that
merely repeat the plan's constants. Validate the knowledge graph and reproducible
paths, and run pre-commit before committing the evidence.
