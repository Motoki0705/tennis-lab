# tennis-lab ChatGPT WSL MCP

This service gives one trusted ChatGPT plugin access to tennis-lab code,
CUDA-capable sandbox jobs, and the repository `.training_queue`. The recommended
connection is an OpenAI Secure MCP Tunnel with a persistent `tunnel_id`.

## Security boundary

- The Secure MCP Tunnel makes only an outbound connection to OpenAI; the MCP
  server listens on WSL loopback and has no public URL.
- Access is assigned to the tunnel in the OpenAI Platform instead of being
  delegated to the loopback server through OAuth.
- Commands run in Docker with all Linux capabilities dropped.
- Only `/home/kamimura/projects/tennis-lab` and the read-only uv Python runtime
  are mounted. `/mnt/c`, SSH credentials, and the Docker socket are absent.
- Network access and GPU access are explicit tool arguments.
- Training always enters `.training_queue/` before its sandbox starts.

The MCP process itself needs Docker access to create the sandboxes and uses the
host's Git credentials only in the dedicated `push_workspace_branch` tool. A
legacy public mode additionally supports OAuth 2.1 with PKCE and owner approval.

## Local validation

```bash
.venv/bin/python -m src.automation.chatgpt_mcp serve \
  --public-base-url https://mcp.example.com

npx @modelcontextprotocol/inspector@latest
```

The Streamable HTTP endpoint is `/mcp`. OAuth metadata is served from the
standard MCP protected-resource and authorization-server discovery routes.

## Persistent OpenAI Secure MCP Tunnel

Create a tunnel at
<https://platform.openai.com/settings/organization/tunnels>, assign it to the
personal ChatGPT workspace, and create a runtime API key with **Tunnels: Read +
Use** at <https://platform.openai.com/settings/organization/api-keys>.

Configure WSL from an interactive terminal. The key prompt is hidden, the key is
stored with mode `0600`, and systemd receives only a `file:` reference:

```bash
.venv/bin/python -m src.automation.chatgpt_mcp configure-secure-tunnel \
  --tunnel-id tunnel_0123456789abcdef0123456789abcdef \
  --start
.venv/bin/python -m src.automation.chatgpt_mcp show-secure-connection
.venv/bin/python -m src.automation.chatgpt_mcp doctor-secure-tunnel
```

The doctor uses an ephemeral diagnostic port when the service is already
running. Missing OAuth metadata is reported as an explicit `SKIP` because the
loopback MCP intentionally uses no authentication and the Secure Tunnel is the
access-control boundary. Every other failed check remains fatal.

For non-interactive setup, put only the runtime key in a private file and pass
`--runtime-key-file /absolute/path/to/key`. The setup copies it into the MCP
state directory; the source file is not managed or deleted.

In <https://chatgpt.com/#settings/Connectors>, create the plugin with:

- Connection: `Tunnel`
- Tunnel: the configured `tunnel_id`
- Authentication: `None`

The `tunnel_id` is stable across WSL, MCP, and tunnel-client restarts. The two
enabled user services are `tennis-lab-chatgpt-mcp-private.service` and
`tennis-lab-chatgpt-secure-tunnel.service`. The private MCP endpoint is
`http://127.0.0.1:8767/mcp`; the tunnel diagnostics UI is loopback-only on port
`8768`.

## Legacy temporary HTTPS endpoint

Install and start the user service:

```bash
.venv/bin/python -m src.automation.chatgpt_mcp install-user-service --start
.venv/bin/python -m src.automation.chatgpt_mcp show-connection
```

`serve-public` uses an outbound Cloudflare Quick Tunnel, so no inbound WSL or
router port is opened. Its `trycloudflare.com` URL is temporary and changes if
the service is restarted. Keep this service only as a migration fallback; it
can run alongside the Secure MCP Tunnel because they use different ports.

### Legacy ChatGPT plugin fields

Use the values printed by `show-connection`:

- Connection: `Server URL`
- Authentication: `OAuth`
- Name and description: the printed values

When ChatGPT opens the authorization page, paste the printed owner secret once.
