# CPU CI measurements

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

These tools plan and observe the CPU suite using the
[sharding and coverage audit](sharding.md); the default local command remains unsharded and complete. The CPU marker
expression remains owned by `.spin/cmds.py`; the court dataset integration retains
its existing explicit exclusion. Both CI and the local measurement runner use
`spin test -q --no-cov --dist=worksteal`, an explicit worker count (local default two; CI choices in [sharding](sharding.md)), native thread counts of one,
CUDA hidden, `--durations=50 --durations-min=1`, and JUnit.

Pytest stdout/stderr is streamed to the job log and saved in `pytest.log`. This preserves the existing `ci_watch` failure signatures, including failed nodeids and setup exceptions. A closed output pipe fails the launcher and terminates its test process group.


## Lane registry and planning

[`lanes.json`](lanes.json) registers **only file-level exceptions** to always-run
CPU tests. Each suite requires `id` (unique identifier), `files` (existing explicit test
paths), `paths` (repository-relative shell patterns for dependencies), `reason`
and `owner` (nonempty strings). Overlapping files, missing files, unknown fields,
missing owners/reasons and invalid patterns fail planning. Files follow the
pytest discovery patterns in `pyproject.toml`. Unregistered files always run;
an exception covers every collected case in that file, including new cases.
For a suite with nonempty `paths`, changing its own test file also requests it.
An empty `paths` array explicitly defers the suite on ordinary path-selected PRs,
including changes to its own test file. This is used for the user-approved whole-repository
architecture sweep. Full requests described below still include every suite.
Include local helpers, configuration and dependency paths when registering a suite;
the registry owns the suite list, rationale and owner.

PRs compare the event's `base.sha..head.sha`, covering all commits even after a
previous PR run was cancelled. Path selection applies only to PRs. Main pushes
and manual dispatch always request the complete CPU target; there is no scheduled
lane. Shared infrastructure (`src/utils`, `src/tasks/base`, `tests/support`, any
`conftest.py`, dependency configuration/lock, `.spin`, workflows and these CI
tools/registry), renames, deletions and unavailable diffs request all tests with
explicit reasons. Invalid registries/plans fail instead of becoming a subset.

The standard-library-only `plan` job produces one immutable, source/run/attempt
specific `plan.json` for every shard and the `Python` aggregate:

```bash
python3 -m scripts.ci_metrics.plan --output .ci-plan/plan.json
```

The command reads `GITHUB_EVENT_PATH` and the Actions context. The registry and
source test-file inventory are checked against each checkout. Pass that plan to
the normal shard command with `--lane-plan .ci-plan/plan.json`; arbitrary pytest
arguments cannot be combined with a lane plan. Without a plan, local sharded
measurements explicitly request the full CPU target.

Plan and aggregate summaries list deferred suites, owner, reason and the last
verified full main success commit. Read-only Actions API lookup checks up to the
latest 30 successful main push runs and verifies ancestry, policy, run/attempt
and `cpu-test-coverage-*` evidence. Green jobs, partial runs or all-skipped suites
do not establish suite success. Missing/expired/incompatible evidence and API
errors remain **unknown**, with a reason. Coverage artifacts are kept for 90 days;
no success index is committed to the repository.

## Test growth and cost warnings

The `Python` aggregate adds one informational report to the job summary after
the coverage audit. It shows per-file case additions/removals/net changes, cases
above the time budget (new case, newly over budget, existing, or unknown without
a baseline), large parametrized functions, expensive case/fixture setups, and
file-level profile versus measured phase seconds (absolute and percentage drift).
The thresholds and their interpretation are owned by the
[test policy](../../tests/README.md#ケース数と実行時間); exceeding them never fails CI.

The comparison uses the latest verified full successful main push run whose
commit is an ancestor of the PR base (or the current commit for main/manual
runs), excluding the current run. Read-only API lookup checks at most 30 successful
main runs. Missing/expired artifacts, incompatible schemas, malformed/missing
data, unavailable ancestry and API failures are displayed with reasons. Once the
latest full success is identified, incompatible/missing growth data does not
silently select an older baseline. The first run after enabling this report will
normally show no growth baseline; current costs and profile drift still appear.

When D is nonempty, the report explicitly limits comparisons to R's files,
including the baseline side. Deferred tests have no measured cost and are never
reported as removed, skipped, or zero seconds. Pytest `originalname`, parent
identity and callspec metadata determine function families; parameter IDs are
not parsed for brackets. Collection metadata is checked across workers without
changing `inventory.json` or the lane/coverage schema. Missing details disable
the cost report with a reason, leaving the coverage verdict independent.

Fixture rows show scope, actual setup executions, summed setup and the longest
single setup. A fixture exceeds the budget only when a single execution does;
many cheap executions are not one expensive setup. Fixture time is already
contained in case setup and is never added to test phase totals. Missing call
phases on setup skips remain unmeasured, with status shown. Profile comparison
uses the exact scheduling profile fingerprint; missing or mismatched profiles
show a reason, and new files are labelled unmeasured in the profile.

The `cpu-test-coverage-<run>-<attempt>` artifact also keeps `growth.json`
(schema 1, complete requested inventory/details/results/fixture costs and run
identity, or an explicit incomplete reason), `growth-summary.json` (all report rows), and `growth-summary.md` for
90 days. Summary tables show up to 20 rows each; full rows remain in JSON.

## Run and recover

From the checkout being measured, with its locked environment installed:

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/output --workers 2
```

The destination must be empty. Runs do not acquire a host-wide pytest lock,
matching the repository’s parallel pytest policy. `PYTEST_LOCK` is ignored.
For an explicitly coordinated performance comparison, `--lock /absolute/path`
serializes only measurements that opt into that path. It does not exclude ordinary
pytest processes: record competing host load and compare under matching conditions.

Keep the source unchanged from launch, including lock wait. If it changes while
queued, start a fresh process so already imported measurement code cannot be stale.

An optional subset goes after `--`, for example `-- tests/unit/utils/geometry/test_angles.py`.
Any extra arguments make `full_cpu_target` false. A subset cannot establish the
full CPU baseline. Do not add worker, marker, JUnit, output, or plugin overrides.

The runner strips real agent session/Herdr identity from its test subprocess only,
and clears inherited pytest worker identity/options before starting nested pytest.
The caller's environment is unchanged. Setup failures and setup skips normally
have no call report; those remain failed/skipped tests, distinct from missing
phases. Signals produce cancelled observations and are forwarded to the entire
pytest process group. A kill or runner loss can prevent finalization. Recover a
partial directory without rerunning tests:

```bash
.venv/bin/python -m scripts.ci_metrics report --output /absolute/partial/output
```

`run` preserves pytest's nonzero exit status; an otherwise successful run with
incomplete measurements exits 2. `report` exits 2 unless it can establish a
successful test observation. It never repairs missing evidence with another run.
If cancellation or a launcher failure prevents the test command from starting,
`run.json` keeps its test exit status null; the CLI's signal/error status is separate.

## Artifacts and time definitions

| Artifact | Meaning |
|---|---|
| `environment.json` | Source revision and local changes, command, package versions, OS/image, CPU/affinity, RAM, worker/thread counts, run/attempt, cache condition, starting host load |
| `run.json` | Requested/start/finish times, separate lock wait, exit/signal, source consistency |
| `events-*.jsonl` | Flushed per-process observations: selected/deselected nodeids, collection failures/skips, worker collection, fixture setup, test phases, process finish |
| `inventory.json` | Selected nodeids; compared in exact order across every worker |
| `inventory-details.json` | Schema 1 collection metadata, including originalname/callspec, or explicit detail issues; contains U for lane runs |
| `lane-plan.json`, `shard-*.json`, `cpu_universe` events | Fixed request, U/R/D partition and independent collection evidence before lane selection; see [sharding](sharding.md#evidence-and-failure-semantics) |
| `results.json` | Per-node passed/skipped/failed/xfail/xpass/incomplete status and phase times |
| `junit.xml`, `pytest.log` | Independent JUnit totals/identities and standard durations output |
| `summary.json`, `summary.md` | Completeness checks, process startup/collection times, costs by file/fixture, outcome counts |
| Aggregate `growth.json`, `growth-summary.json`, `growth-summary.md` | Audited requested observations and informational growth/cost comparison; see [warnings](#test-growth-and-cost-warnings) |

`full_cpu_baseline` requires the complete default selection and a successful test
observation. Missing runner/session/JUnit evidence, worker inventory disagreement,
missing or duplicate phases, inconsistent JUnit totals/identities, or source
changes invalidate it. Failure and cancellation remain observations, never success.
Collection-time module skips/errors remain separate events because they do not
necessarily correspond to a selected test nodeid.

UTC timestamps support correlation with Actions. Command, lock-wait, startup,
collection and post-test intervals use `time.perf_counter` on the measured host,
so host clock corrections cannot change elapsed costs. Post-test time starts at
the final worker phase report. The wall-clock difference from monotonic command
time is recorded separately. Missing monotonic observations leave elapsed values
unavailable and prevent a successful baseline; UTC subtraction is not a fallback.

Phase totals are **summed elapsed time across workers**, not CPU time or job wall
time. Startup includes Python/spin/pytest before plugin import; worker collection
is session start to collection completion; timestamps and the controller's final
worker-collection notification expose scheduling/startup differences. Post-test
time is the last worker phase report to command exit. Fixture setup counts cover actual
executions after dependency resolution, including errors; they exclude fixture
finalizers, which remain in test teardown. Fixture setup is contained in test
setup and must not be added to it. Shared fixture setup is attributed to its
scope node, not duplicated over every consumer.

RSS is the largest **individual pytest controller/worker** high-water RSS in Linux
KiB. It excludes test subprocesses and is not a simultaneous process-tree peak or
an estimate of aggregate memory needs. Event serialization/flush time and report
generation time are recorded separately; event write time is only a lower bound
on instrumentation overhead. Assess total overhead with matched, alternating
instrumented/uninstrumented small-suite runs under the same lock and environment;
retain all commands, samples, and unsuccessful attempts. Do not infer full-suite
overhead from a startup-dominated sample.

## Actions attempts

CI records the runner before dependency preparation. Summary, REST snapshot, and
artifact steps use `always()`, preserving evidence after preparation/test failures.
Hard cancellation or a lost runner may prevent upload; REST accounting still
records that attempt as cancelled/incomplete. Artifact names include run and
attempt. Cache hit is recorded from `setup-uv`, and unavailable values remain
unknown rather than being assumed warm.

The in-job REST snapshot is provisional: its own job/upload/post-cache steps have
not completed. Fetch again **after** the job ends for final all-runner/post time:

```bash
.venv/bin/python -m scripts.ci_metrics fetch-actions --repo Motoki0705/tennis-lab \
  --run-id 39767861734 --output /absolute/new/actions
.venv/bin/python -m scripts.ci_metrics actions --input /absolute/new/actions/actions-raw.json \
  --output /absolute/new/actions
```

`fetch-actions` uses read-only `gh api` requests, fetching every attempt and every
page of that attempt's jobs. `--attempt` requests one provisional snapshot.
Retrieval can succeed while timestamps remain incomplete; inspect the summary's
`complete_measurement` and `successful_observed_checks`. `actions` exits 2 for
missing/incomplete data. Its input is an array of `{run, jobs}` records containing
unmodified attempt-specific REST fields; optional `requested_at`, `cache_condition`,
and provenance may be retained. Duplicate attempts are rejected. A missing earlier
attempt is listed and prevents a cumulative all-attempt runner total.

Queue accounting separates attempt dispatch to first job and job creation to job
start. For attempt 1, the origin is `created_at`; for retries, exact request time
is usually unavailable, so `run_started_at` is explicitly labelled a proxy.
Provide `requested_at` only when independently observed. Check completion is from
that origin to completion of all named observed checks (`--check Python`, repeat
for several). It is reported separately from the **sum of every job's runner
time**, including failed/cancelled attempts and other concurrent jobs. `Python`
is also the required merge check, as defined in the [repository workflow](../../.agents/README.md#github運用ルール初版).

Step categories distinguish checkout, uv/cache/install, apt, lint, pytest, post,
and other setup/measurement work. Raw step timestamps and conclusions are kept.
Missing timestamps yield unavailable totals with explicit issues; they are not
zero-duration successes. Skipped steps retain the skipped conclusion.

Median and p95 use successful complete attempts only, with `n` and the nearest-rank
formula `sorted[ceil(0.95*n)-1]`. For `n < 20`, p95 is labelled reference only.
Outcome counts and runner totals still include failures/cancellations. Mixed
revisions, runners, cache states or inventories are descriptive observations,
never before/after evidence. No retry is silently substituted for an earlier run.

## Comparison contract

For scheduling comparisons, hold source, collected inventory, runner image/CPU/RAM,
dependencies, worker/thread counts, cache state, and load conditions fixed. For
test consolidation, inventories intentionally differ: retain an explicit mapping
from removed cases to the surviving defect/boundary contracts, plus initialization,
fixture/CLI counts and setup/call/teardown time. Costs may overlap and do not by
themselves estimate job-wall savings. Keep the complete CPU execution path above.

The measured candidate decisions and baseline evidence are in
[baseline.md](baseline.md). They are historical observations, not selection rules.

The implementation uses [pytest hooks](https://docs.pytest.org/en/stable/reference/reference.html#hooks),
[Actions attempt APIs](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run-attempt),
and [`setup-uv` cache outputs](https://github.com/astral-sh/setup-uv#outputs).

Regenerate scheduling weights from complete Actions artifacts with the [`profile` command](sharding.md#regenerate-measured-weights). Failed cases remain visible; incomplete attempts are rejected.
