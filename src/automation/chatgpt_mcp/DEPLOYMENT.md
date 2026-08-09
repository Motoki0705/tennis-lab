# WSL MCP deployment diagnostics

`Deploy WSL MCP` runs automatically on the self-hosted WSL GPU runner when the repository owner pushes reviewed MCP changes to `main`. The repository variable `LOCAL_GPU_ACTIONS_ENABLED=true` is the deployment kill switch; this workflow deliberately has no GitHub Environment approval gate.

A separate `Report WSL MCP deployment` workflow observes both requested and completed deploy runs and posts only the run ID, attempt, status/conclusion, head SHA, and run URL to PR #723. Reporting at request time makes runner and concurrency waits visible before the deploy job starts. It intentionally does not post service logs, environment contents, API keys, Tunnel credentials, or other secrets.

Use the reported run ID with the GitHub Actions API/MCP to inspect job and step logs. Fixes must be made through a reviewed GitHub branch/PR; the WSL MCP remains an execution plane and does not commit or push source changes. Keep the reporter installed on `main` before triggering another deployment so the requested event can be observed immediately.
