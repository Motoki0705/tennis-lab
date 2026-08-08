# tennis-lab ChatGPT WSL MCP

This gateway gives ChatGPT a deliberately narrow execution plane for
`tennis-lab`. GitHub MCP remains the only repository control plane.

## Fixed responsibility split

GitHub MCP owns repository exploration, issues and pull requests, branch
creation, implementation, commits, pushes, and remote state.

WSL MCP owns only:

- fetching one branch from the fixed `origin` remote;
- checking that the fetched branch equals a caller-supplied full commit SHA;
- creating a detached local revision workspace;
- running bounded CPU validation such as pytest, ruff, and mypy;
- queuing CUDA experiments and training through `.training_queue`;
- returning job status and secret-redacted logs.

The WSL MCP does **not** expose file listing, file reading, code search, patch
application, branch creation, commit, or push tools.

## Execution security boundary

- Secure Tunnel mode listens only on `127.0.0.1`; WSL makes the outbound
  connection to OpenAI.
- Remote checkout is fixed to `origin` and requires an exact 40-character SHA.
- Tool callers receive an opaque workspace ID rather than a filesystem path.
- Every execution revalidates the registered SHA and completely clean source
  tree.
- A source worktree is mounted read-only as `/source`.
- Each job receives a fresh private copy at `/workspace`; changes cannot flow
  back into the source worktree or canonical checkout.
- Git metadata is masked inside the container.
- The canonical repository `.venv` and uv Python runtime are read-only mounts.
- Docker runs as the invoking UID/GID with all capabilities dropped,
  `no-new-privileges`, a read-only root filesystem, bounded memory/PIDs, and no
  network.
- Direct commands are CPU-only, limited to 30 minutes, and limited to two
  concurrent containers.
- All GPU work is serialized through `.training_queue`.
- Runtime API keys and profiles are stored below the private state directory
  with mode `0600`; raw commands are absent from Docker metadata and represented
  in durable metadata only by a SHA-256 digest. Private command/spec files are
  deleted after handoff.
- Returned logs redact common OpenAI, GitHub, and bearer-token forms.

The MCP host process still needs access to the Docker daemon and to the fixed
`origin` remote. Those privileges are reachable only through the constrained
tools above.

## Available MCP tools

1. `get_host_status`
2. `prepare_revision_workspace`
3. `get_revision_status`
4. `start_command`
5. `get_command_job`
6. `list_command_jobs`
7. `get_command_output`
8. `cancel_command_job`
9. `enqueue_training`
10. `get_training_job`
11. `get_training_output`

A normal GitHub-MCP-to-WSL-MCP handoff is:

1. GitHub MCP implements and pushes a branch.
2. GitHub MCP obtains the branch's full head SHA.
3. WSL MCP calls `prepare_revision_workspace(branch, expected_sha)`.
4. WSL MCP runs CPU checks with `start_command`.
5. WSL MCP queues CUDA or training with `enqueue_training`.
6. GitHub MCP alone makes any required source changes and pushes a new SHA.
7. WSL MCP prepares a new workspace for that new SHA.

## Persistent OpenAI Secure MCP Tunnel

Create a tunnel in the OpenAI Platform, assign it to the intended ChatGPT
workspace, and create a runtime API key with **Tunnels: Read + Use**.

Persistent services must be installed from the canonical repository checkout,
not from a feature worktree. After this PR is merged and local `main` is updated:

```bash
cd /home/kamimura/projects/tennis-lab
git switch main
git pull --ff-only
uv sync --locked

.venv/bin/python -m src.automation.chatgpt_mcp configure-secure-tunnel \
  --tunnel-id tunnel_0123456789abcdef0123456789abcdef \
  --start

.venv/bin/python -m src.automation.chatgpt_mcp show-secure-connection
.venv/bin/python -m src.automation.chatgpt_mcp doctor-secure-tunnel
```

The API key prompt is hidden. Do not place the key in chat, command-line
arguments, GitHub, or shell history.

Configure the ChatGPT connector with:

- Connection: `Tunnel`
- Tunnel: the configured tunnel ID
- Authentication: `None`

The stable services are:

- `tennis-lab-chatgpt-mcp-private.service`
- `tennis-lab-chatgpt-secure-tunnel.service`

The private MCP endpoint is `http://127.0.0.1:8767/mcp`; tunnel diagnostics use
loopback port `8768`.

## Post-install validation

First verify the local services:

```bash
systemctl --user is-active tennis-lab-chatgpt-mcp-private.service
systemctl --user is-active tennis-lab-chatgpt-secure-tunnel.service
curl --fail --silent http://127.0.0.1:8767/healthz
curl --fail --silent http://127.0.0.1:8768/readyz
```

Then use the ChatGPT connector to validate, in order:

1. `get_host_status` reports Docker, the expected NVIDIA GPU, and queue status.
2. GitHub MCP supplies a branch and its exact head SHA.
3. `prepare_revision_workspace` returns a workspace ID bound to that SHA.
4. `start_command` runs a network-disabled CPU smoke test.
5. `enqueue_training` runs a short CUDA smoke test through the queue.
6. Logs contain no host credentials and the container cannot see `.git`, the
   Docker socket, `/mnt/c`, or any source tree other than its copied snapshot.

Only after those checks succeed should the legacy Quick Tunnel service be
stopped and disabled.

## Legacy temporary HTTPS endpoint

The OAuth-protected Cloudflare Quick Tunnel remains temporarily available for
migration. It exposes the same reduced execution-only tool set. It should be
removed after Secure Tunnel validation succeeds.
