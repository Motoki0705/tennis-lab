# CPU CI measurements

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

These tools observe the existing CPU suite. They do not delete tests, add markers,
change fixtures, select tests by changed paths, or shard the suite. The CPU marker
expression remains owned by `.spin/cmds.py`; the court dataset integration retains
its existing explicit exclusion. Both CI and the local measurement runner use
`spin test -q --no-cov --dist=worksteal`, two workers, native thread counts of one,
CUDA hidden, `--durations=50 --durations-min=1`, and JUnit.

## Run and recover

From the checkout being measured, with its locked environment installed:

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/output --workers 2
```

The destination must be empty. Local runs acquire the same host-wide lock as
`.agents/ops/pytest_locked.sh`: `${PYTEST_LOCK}` or
`${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/pytest.lock`.
`--lock /absolute/path` explicitly overrides it. Keep the shared default for actual
measurements. Only isolated regression fixtures use temporary locks. GitHub-hosted
Actions runners need no local host lock. Ordinary regression still uses the
existing wrapper with its fixed four workers; this tool does not change it.

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

## Artifacts and time definitions

| Artifact | Meaning |
|---|---|
| `environment.json` | Source revision and local changes, command, package versions, OS/image, CPU/affinity, RAM, worker/thread counts, run/attempt, cache condition, starting host load |
| `run.json` | Requested/start/finish times, separate lock wait, exit/signal, source consistency |
| `events-*.jsonl` | Flushed per-process observations: selected/deselected nodeids, collection failures/skips, worker collection, fixture setup, test phases, process finish |
| `inventory.json` | Selected nodeids; compared in exact order across every worker |
| `results.json` | Per-node passed/skipped/failed/xfail/xpass/incomplete status and phase times |
| `junit.xml`, `pytest.log` | Independent JUnit totals/identities and standard durations output |
| `summary.json`, `summary.md` | Completeness checks, process startup/collection times, costs by file/fixture, outcome counts |

`full_cpu_baseline` requires the complete default selection and a successful test
observation. Missing runner/session/JUnit evidence, worker inventory disagreement,
missing or duplicate phases, inconsistent JUnit totals/identities, or source
changes invalidate it. Failure and cancellation remain observations, never success.
Collection-time module skips/errors remain separate events because they do not
necessarily correspond to a selected test nodeid.

Phase totals are **summed elapsed time across workers**, not CPU time or job wall
time. Startup includes Python/spin/pytest before plugin import; worker collection
is session start to collection completion; timestamps and the controller's final
worker-collection notification expose scheduling/startup differences. Post-test
time is the last phase report to command exit. Fixture setup counts cover actual
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
is an observed check name, not a restored merge condition.

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
