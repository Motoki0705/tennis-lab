# tennis-lab ChatGPT WSL MCP

This gateway gives ChatGPT a broad execution plane inside `tennis-lab` while
keeping its host control plane outside the project. GitHub MCP remains the
repository control plane.

## Responsibility split

GitHub MCP owns repository exploration, Issues and Pull Requests, branch
creation, source implementation, commits, pushes, and remote state.

WSL MCP owns runtime work only:

- fetch one fixed `origin` branch through an external trusted bare mirror;
- bind every job to a caller-supplied full commit SHA;
- run arbitrary network-disabled shell commands in Docker;
- expose the complete local `tennis-lab` project read-write for real data,
  generated chunks, outputs, checkpoints, artifacts, caches, and experiments;
- serialize CUDA and long-running work through the external training queue and
  the same host GPU lock used by local GitHub Actions;
- return job state and secret-redacted output.

It does not expose MCP tools for source browsing, patching, committing, or
pushing. A shell command can nevertheless modify or delete anything below the
local `tennis-lab` directory. That destruction is explicitly inside the threat
model and cannot affect GitHub unless GitHub MCP separately persists a change.

## Security boundary

The destructible zone is the complete project root, including `src/`, `tests/`,
`data/`, `outputs/`, `ckpt/`, `artifacts/`, `.cache/`, `third_party/`, and all
other project content. The trusted runtime, versioned venv, tunnel credentials,
Git mirror, queue runner, systemd units, and durable MCP state live under
`~/.local/share/tennis-lab-chatgpt-mcp/` and
`~/.local/state/tennis-lab-chatgpt-mcp/`, outside that tree.

The container receives:

- `/workspace`: a private read-write copy of the exact remote revision;
- `/tennis-lab`: the complete local project read-write;
- standard mutable roots in `/workspace` linked to `/tennis-lab`, including
  `data`, `outputs`, `ckpt`, `checkpoints`, `artifacts`, `.cache`,
  `third_party`, and `.training_queue`;
- the external trusted venv and uv Python runtime read-only;
- no network, Docker socket, `/mnt/c`, host credentials, systemd, tunnel
  credentials, queue runner, trusted Git mirror, or MCP runtime source;
- masked `.git` metadata for both execution roots;
- a read-only container root filesystem, all capabilities dropped,
  `no-new-privileges`, private IPC, PID and memory limits, and a bounded timeout.

The host service imports only the external trusted runtime. Runtime promotion is
accepted only from a separate, completely clean checkout at an explicitly
supplied full SHA whose `origin` matches `Motoki0705/tennis-lab`; the canonical
read-write project cannot be promoted into the control plane.

Because all project files are readable, commands can intentionally print project
data into MCP logs. Never place API keys, SSH keys, personal credentials, or
other secrets anywhere below `tennis-lab`. Treat the local project as untrusted
after arbitrary execution and do not run its Python or shell code directly on
the host until it has been restored or reviewed. Disk exhaustion and kernel or
Docker vulnerabilities remain residual host risks; keep backups for valuable
data and outputs.

## MCP tools

1. `get_host_status`
2. `get_execution_layout`
3. `prepare_revision_workspace`
4. `get_revision_status`
5. `start_command`
6. `get_command_job`
7. `list_command_jobs`
8. `get_command_output`
9. `cancel_command_job`
10. `enqueue_training`
11. `get_training_job`
12. `list_training_jobs`
13. `get_training_output`
14. `cancel_training_job`

`start_command` accepts any CPU shell command, a relative working directory,
and one of two roots. `execution_root="revision"` uses exact code with persistent
project data/output/checkpoint roots linked in. `execution_root="project"` uses
the complete current local project tree. Direct jobs may run for up to 24 hours;
at most two run concurrently and each is limited to 24 GiB. GPU or heavier work
uses `enqueue_training`, is serialized, and receives 48 GiB.

Network access is intentionally unavailable. Downloads must use a separately
reviewed workflow rather than an arbitrary MCP command.

## Typical flow

1. GitHub MCP implements and pushes a branch.
2. GitHub MCP obtains its full head SHA.
3. WSL MCP calls `prepare_revision_workspace(branch, expected_sha)`.
4. WSL MCP runs CPU tests, real-data validation, generation, or inspection with
   `start_command`.
5. WSL MCP submits CUDA, evaluation, or training with `enqueue_training`.
6. GitHub MCP alone persists source changes to the remote branch.

Examples:

```text
start_command(
  workspace_id="rev-...",
  expected_sha="<40 chars>",
  execution_root="revision",
  working_directory=".",
  timeout_seconds=3600,
  command="python -m pytest -m local_data -q"
)
```

```text
enqueue_training(
  name="blcs-tracking-chunked",
  workspace_id="rev-...",
  expected_sha="<40 chars>",
  execution_root="revision",
  working_directory=".",
  timeout_seconds=86400,
  command="python -m src.tasks.blcs.scripts.train --config-name train_tracking_chunked"
)
```

## Trusted deployment

The supported deployment route is the self-hosted **Deploy WSL MCP** workflow.
It checks out the reviewed `main` revision into the Actions workspace, provisions
a lockfile-keyed venv outside `tennis-lab`, requires an exact clean SHA and fixed
origin, atomically installs the external runtime, reuses the stored Tunnel ID and
runtime key, restarts both services, and verifies MCP discovery, real project
read-write access, host isolation, CPU tests, CUDA, and the serial queue.

Do not run `configure-secure-tunnel --source-root
/home/kamimura/projects/tennis-lab`; the canonical project is intentionally
rejected as a deployment source. Manual recovery requires a separate clean
checkout and both `--source-root` and `--expected-sha`.

Connector settings remain:

```text
Connection: Tunnel
Tunnel: tennis-lab WSL
Authentication: None
```

Stable services are `tennis-lab-chatgpt-mcp-private.service` and
`tennis-lab-chatgpt-secure-tunnel.service`. The private MCP endpoint is
`http://127.0.0.1:8767/mcp`; tunnel readiness is
`http://127.0.0.1:8768/readyz`. Keep the legacy Quick Tunnel until an actual
ChatGPT Secure Tunnel call succeeds, then disable it.
