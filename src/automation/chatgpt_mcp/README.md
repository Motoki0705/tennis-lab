# tennis-lab ChatGPT WSL MCP

This gateway gives ChatGPT a broad execution plane inside `tennis-lab` while
keeping the host control plane outside the project. GitHub MCP remains the
repository control plane.

## Responsibility split

GitHub MCP owns repository exploration, Issues and Pull Requests, branch
creation, source implementation, commits, pushes, and remote state.

WSL MCP owns runtime work only:

- fetch one `origin` branch through an external trusted bare mirror;
- bind every job to a caller-supplied full commit SHA;
- run arbitrary network-disabled shell commands in Docker;
- expose the complete local `tennis-lab` project read-write for real data,
  generated chunks, outputs, checkpoints, artifacts, caches, and experiments;
- serialize CUDA and long-running work through the external training queue;
- return job state and secret-redacted output.

It does not expose MCP tools for source browsing, patching, committing, or
pushing. A shell command can nevertheless modify or delete anything below the
local `tennis-lab` directory. That destruction is explicitly inside the threat
model and cannot affect GitHub unless GitHub MCP separately pushes a change.

## Security boundary

The destructible zone is the complete project root:

```text
/home/kamimura/projects/tennis-lab
├── src/
├── tests/
├── data/
├── outputs/
├── ckpt/
├── artifacts/
├── .cache/
├── third_party/
└── any other project content
```

The trusted control plane is installed outside it:

```text
~/.local/share/tennis-lab-chatgpt-mcp/
├── current -> releases/<full SHA>
├── releases/
├── venv/
├── repository.git/
└── bin/training_queue.sh

~/.local/state/tennis-lab-chatgpt-mcp/
├── secure-tunnel/
├── revisions/
├── sandboxes/
├── training-specs/
├── training-queue/
└── gateway.sqlite3
```

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

The host executes only external trusted runtime code and the external copied
queue runner. It never imports or executes Python or shell code from the
read-write project tree.

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
and one of two roots:

- `execution_root="revision"`: exact source revision with persistent project
  data/output/checkpoint roots linked in;
- `execution_root="project"`: the complete current local project tree.

Direct jobs may run for up to 24 hours and at most four run concurrently. GPU
or longer work must use `enqueue_training`; the queue serializes it.

Network access is intentionally unavailable. Downloading dependencies or data
must be performed through a separately reviewed workflow, not an arbitrary MCP
command.

## Typical flow

1. GitHub MCP implements and pushes a branch.
2. GitHub MCP obtains its full head SHA.
3. WSL MCP calls `prepare_revision_workspace(branch, expected_sha)`.
4. WSL MCP runs CPU checks or real-data inspection with `start_command`.
5. WSL MCP submits CUDA, dataset generation, evaluation, or training with
   `enqueue_training`.
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
start_command(
  workspace_id="rev-...",
  expected_sha="<40 chars>",
  execution_root="project",
  working_directory="data",
  timeout_seconds=1800,
  command="du -sh . && find . -maxdepth 3 -type f | head"
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

## Trusted runtime deployment

The project `.venv` is moved to the trusted control directory and replaced by
a symlink, preserving `.venv/bin/python` for normal local use. A reviewed source
revision is copied into a versioned external release and atomically activated.
The trusted bare mirror and external queue runner are also initialized.

Manual deployment:

```bash
cd /home/kamimura/projects/tennis-lab
git switch main
git pull --ff-only

.venv/bin/python -m src.automation.chatgpt_mcp configure-secure-tunnel \
  --source-root /home/kamimura/projects/tennis-lab \
  --tunnel-id tunnel_0123456789abcdef0123456789abcdef \
  --reuse-existing-key \
  --start
```

For the first installation, omit `--reuse-existing-key`; the runtime API key is
requested through a hidden prompt.

The repository also contains a self-hosted `Deploy WSL MCP` workflow. After a
reviewed MCP change reaches `main`, the local WSL runner installs that exact
commit, reuses the already stored Tunnel ID and runtime key, restarts both
services, and verifies health, doctor output, tool discovery, Docker isolation,
CPU execution, CUDA, and the external training queue.

Connector settings remain:

```text
Connection: Tunnel
Tunnel: tennis-lab WSL
Authentication: None
```

Stable services:

- `tennis-lab-chatgpt-mcp-private.service`
- `tennis-lab-chatgpt-secure-tunnel.service`

Endpoints:

- private MCP: `http://127.0.0.1:8767/mcp`
- tunnel readiness: `http://127.0.0.1:8768/readyz`

The legacy OAuth-protected Cloudflare Quick Tunnel may remain during migration
but should be disabled after Secure Tunnel verification succeeds.
