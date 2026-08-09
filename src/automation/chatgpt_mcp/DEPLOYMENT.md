# WSL MCP deployment diagnostics

`Deploy WSL MCP` runs automatically on the self-hosted WSL GPU runner when the repository owner pushes reviewed MCP changes to `main`. The repository variable `LOCAL_GPU_ACTIONS_ENABLED=true` is the deployment kill switch; this workflow deliberately has no GitHub Environment approval gate.

A separate `Publish WSL MCP deployment status` workflow observes both requested and completed deploy runs from a GitHub-hosted runner. It writes the `wsl-mcp/deploy` Commit Status and Actions run URL to the deployed revision without exposing service logs, environment contents, API keys, Tunnel credentials, or other secrets.

Use the Commit Status target URL with the GitHub Actions API/MCP to inspect job and step logs. Fixes must be made through a reviewed GitHub branch/PR; the WSL MCP remains an execution plane and does not commit or push source changes. Keep the status publisher installed on `main` so requested and completed events are recorded independently of the self-hosted runner.
