# WSL MCP deployment diagnostics

`Deploy WSL MCP` runs automatically on the dedicated `trusted-mcp-deploy` runner when the repository owner pushes reviewed MCP changes to `main`. The runner executes as `kamimura`, because deployment must update the owner-only MCP state and control directories and restart the owner's user services. The repository variable `LOCAL_GPU_ACTIONS_ENABLED=true` is the deployment kill switch; this workflow deliberately has no GitHub Environment approval gate.

The dedicated runner is separate from the restricted `tennis-actions` GPU runner. Its pre-job hook rejects every assignment unless the repository, original actor, triggering actor, `main` ref, workflow path, workflow SHA, job ID, and event type match the trusted deployment contract. Install or repair it from the canonical checkout with `scripts/github_actions/install_trusted_mcp_deploy_runner.sh`; do not weaken `ProtectHome` on the general GPU runner.

A separate `Publish WSL MCP deployment status` workflow observes both requested and completed deploy runs from a GitHub-hosted runner. It writes the `wsl-mcp/deploy` Commit Status and Actions run URL to the deployed revision without exposing service logs, environment contents, API keys, Tunnel credentials, or other secrets.

Use the Commit Status target URL with the GitHub Actions API/MCP to inspect job and step logs. Fixes must be made through a reviewed GitHub branch/PR; the WSL MCP remains an execution plane and does not commit or push source changes. Keep the status publisher installed on `main` so requested and completed events are recorded independently of the self-hosted runner.
