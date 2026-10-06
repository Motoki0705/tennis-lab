# Tennis Lab integration for ARIS Codex

This file is the local adaptation of the installed ARIS Codex workflows.
Apply it before their upstream instructions, examples, helpers or shared references.
The user's request and applicable AGENTS.md remain authoritative.

## Runtime and scope

Work in the dedicated checkout required by AGENTS.md. From that checkout root:

```bash
.venv/bin/python .agents/aris/aris.py setup
.venv/bin/python .agents/aris/aris.py doctor
```

`setup` creates only `.aris/installed-skills-codex.txt`, the helper pointer used
by upstream skills. Repeat after moving to another checkout. `ARIS_REPO` is the
active checkout's `.agents/aris`, never an unrelated clone or `~/.aris/repo`.
Use `.venv/bin/python` for Python helpers. Missing required resources are an
explicit error; report the missing capability rather than inventing results.

Use the existing repository as the implementation base. Import a paper's code
only when needed for the requested experiment, and keep it isolated. Installing
these skills does not launch a research campaign or authorize cloud expenditure.
Local GPU is the default; a user-requested Colab run follows AGENTS.md.

## Campaign inputs and files

Carry forward the user's already supplied purpose, candidate methods, datasets,
allowed code changes, evaluation protocol and compute budget. Ask only for missing
decisions that materially prevent execution. With no execution budget, prepare the
concrete comparison plan and seek the missing budget before starting experiments.
Do not infer unlimited execution from upstream "overnight" instructions.

Keep ARIS working artifacts under `outputs/aris/<campaign-id>/`. Resolve upstream
artifact names such as `RESEARCH_BRIEF.md`, `idea-stage/`, `refine-logs/`,
`review-stage/` and `NARRATIVE_REPORT.md` below this campaign directory.
Run source code and enqueue jobs from the checkout root. Helper scripts live
under `.agents/aris/tools`; shared templates under `.agents/aris/templates`;
skill-local scripts stay under `.agents/skills/<name>/scripts`.
The existing upstream `run_state.py` may store orchestration state under
`<campaign-directory>/.aris/runs/`. Record the actual executor identity.

Fix the baseline, split, seed policy, evaluation script and comparison budget
before running candidates. Preserve the requested comparison candidates even if
the upstream idea-discovery workflow would normally select only one idea.
Distinguish numerical measurements, visual observations, hypotheses and missing
evidence. An LLM review score is not an experimental metric.

## GPU execution

Read and use [training-queue](../skills/training-queue/SKILL.md).
All local GPU work, including pilots, sanity runs, inference and profiling,
goes through that queue. The rule also covers commands proposed by nested skills
or subagents. The native-first example in the queue documentation does not
override the project's queue-only rule.

Use the adapter so linked worktrees share the original repository's queue:

```bash
.venv/bin/python .agents/aris/aris.py queue add '<experiment command>' \
  --name '<unique campaign/run name>' --provider codex --session '<actual session id>' \
  --resource all
.venv/bin/python .agents/aris/aris.py queue start
.venv/bin/python .agents/aris/aris.py queue list
```

Replace the example arguments with the concrete command and actual session ID;
the queue skill documents session identification. Preserve `add`'s exact job ID
and enqueue cwd in the campaign tracker. Use `all` by default; choose `half`
only when the workload's measured memory requirements justify sharing.
The adapter delegates the worker's own resource and cancellation semantics.

Upstream `run-experiment`, `experiment-queue`, SSH/screen launch examples and
`MAX_PARALLEL_RUNS=4` do not control local scheduling. Use one shared worker;
do not create a second ARIS GPU scheduler. For dependent batches, enqueue a
downstream phase only after its required predecessor jobs and artifacts pass.

Put the experiment's explicit wall-time limit inside the queued command when
needed, for example `timeout --signal=TERM --kill-after=30s 1800s ...`.
Count failed attempts and retries in the campaign budget. Stop adding jobs when
the remaining budget cannot cover the next run. Do not automatically lower
resolution, batch size, steps or evaluation coverage after OOM; record a proposed
changed condition as a separate experiment. Retries retain distinct job IDs.

Monitor the queue's real states. `running` may include capacity wait; inspect
`list`. Cancellation is not complete until the queue reports terminal teardown.
Cancel only the campaign's own jobs; never use global `clear`/`stop` to manage one
campaign on the shared queue. Do not embed secret values in queued command text.

## Reviews and delegation

Follow the applicable common AGENTS.md delegation and validator rules, including
the user's explicit review count and the current model/provider/effort.
Installation, experiment count, seed count, a named ARIS skill or its defaults do
not constitute an explicit validator count.

With no requested count, use ordinary author checks and deterministic tests;
set independent-review phases to `skipped` and state "independent review not run".
Do not spawn reviewers, juries, rescue agents, adversarial threads or nested
review loops to bypass that rule. Read-only investigations may use `scout` under
the ordinary delegation policy.

An authorized validator allocation belongs to the whole user task, not each
skill/phase/report. Do not reset it on resume. Complete all artifacts and ordinary
checks before consuming it. Upstream four-round loops, score-based early exit,
model/effort pins, cross-provider overlays and follow-up reviews are replaced by
the common policy. Keep original review results and report consumed/completed
counts when reviews were requested.

Treat upstream `accepted` state as evidence of the particular recorded check,
not independent scientific validation. A deterministic verifier may record only
what it actually checked. Never fabricate a reviewer trace or use file existence
as evidence that a scientific claim is true. If an upstream gate requires a review
that the user did not request, mark that review gate not run instead of executing it
or promoting its result. This explicitly replaces the upstream mandatory jury gates.

## Results and knowledge

Read [knowledge-control](../skills/knowledge-control/SKILL.md) when planning and
recording experiments. `knowledge/` is the canonical library for experiments,
papers and current conclusions; do not start a parallel `research-wiki` library.
ARIS plans, trackers and state are operational scratch artifacts.

After a queue job reaches a terminal state, inspect its actual logs, exit status,
config and result files. For a successful run, validate the measurement outputs
before reporting success. Preserve failures and missing measurements explicitly.
Register the matching queue reproduction bundle with `kg_register.py` using
`--repro-dir <shared-queue>/repro/<exact-job-id>` and the relevant task. Pass
`--status failed` for failed runs rather than the registrar's default `done`.
Describe cancellation or invalid outputs using the existing knowledge schema
and retain their actual reason. Follow
the existing skill's recording/summary/validation steps; do not copy that schema
into this workflow or invent scores to complete a node.

The Japanese comparison report links canonical knowledge node IDs, original
metrics, representative plots/images, failures, runtime/VRAM and reproduction
commands. Summarize conclusions and uncertainty; historical run data remains in
the canonical library. Use the installed `render-html` helper with `--lang ja`;
independent rendering review follows the same allocation above, never its
upstream default. End at the comparison report unless paper writing was requested.

Continue autonomously within the agreed scope and budget. A missing required
dataset, unsupported runtime or unresolved policy choice is reported with its
evidence and partial results. Notifications, uploads, publication, cloud provisioning
and changes to the agent environment are separate actions requiring user scope.
