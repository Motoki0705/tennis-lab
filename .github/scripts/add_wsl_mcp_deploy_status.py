"""Add observable commit status reporting to the WSL MCP deploy workflow."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one patch target in {path}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def status_step(*, final: bool) -> str:
    if final:
        return '''
      - name: Publish final deployment status
        if: always()
        env:
          GH_TOKEN: ${{ github.token }}
          JOB_STATUS: ${{ job.status }}
        shell: bash
        run: |
          python3 - <<'PY'
          import json
          import os
          import urllib.request

          job_status = os.environ["JOB_STATUS"]
          state = {
              "success": "success",
              "failure": "failure",
              "cancelled": "error",
          }.get(job_status, "error")
          payload = {
              "state": state,
              "context": "wsl-mcp/deploy",
              "description": f"WSL MCP deployment {job_status}",
              "target_url": (
                  f"{os.environ['GITHUB_SERVER_URL']}/"
                  f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                  f"{os.environ['GITHUB_RUN_ID']}"
              ),
          }
          request = urllib.request.Request(
              (
                  f"{os.environ['GITHUB_API_URL']}/repos/"
                  f"{os.environ['GITHUB_REPOSITORY']}/statuses/"
                  f"{os.environ['GITHUB_SHA']}"
              ),
              data=json.dumps(payload).encode("utf-8"),
              headers={
                  "Accept": "application/vnd.github+json",
                  "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                  "Content-Type": "application/json",
                  "X-GitHub-Api-Version": "2022-11-28",
              },
              method="POST",
          )
          with urllib.request.urlopen(request, timeout=30) as response:
              if response.status != 201:
                  raise SystemExit(f"unexpected status API response: {response.status}")
          PY
'''
    return '''
      - name: Publish pending deployment status
        env:
          GH_TOKEN: ${{ github.token }}
        shell: bash
        run: |
          python3 - <<'PY'
          import json
          import os
          import urllib.request

          payload = {
              "state": "pending",
              "context": "wsl-mcp/deploy",
              "description": "Deploying WSL MCP on the self-hosted runner",
              "target_url": (
                  f"{os.environ['GITHUB_SERVER_URL']}/"
                  f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                  f"{os.environ['GITHUB_RUN_ID']}"
              ),
          }
          request = urllib.request.Request(
              (
                  f"{os.environ['GITHUB_API_URL']}/repos/"
                  f"{os.environ['GITHUB_REPOSITORY']}/statuses/"
                  f"{os.environ['GITHUB_SHA']}"
              ),
              data=json.dumps(payload).encode("utf-8"),
              headers={
                  "Accept": "application/vnd.github+json",
                  "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                  "Content-Type": "application/json",
                  "X-GitHub-Api-Version": "2022-11-28",
              },
              method="POST",
          )
          with urllib.request.urlopen(request, timeout=30) as response:
              if response.status != 201:
                  raise SystemExit(f"unexpected status API response: {response.status}")
          PY
'''


def patch_workflow() -> None:
    path = Path(".github/workflows/deploy-wsl-mcp.yml")
    replace_once(
        path,
        "permissions:\n  contents: read\n",
        "permissions:\n  contents: read\n  statuses: write\n",
    )
    checkout = '''      - name: Checkout reviewed main revision
        uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6
        with:
          clean: true
          persist-credentials: false
          submodules: false
'''
    replace_once(path, checkout, checkout + status_step(final=False))
    text = path.read_text(encoding="utf-8")
    if not text.endswith("          PY\n"):
        raise SystemExit("deploy workflow no longer ends with the verification heredoc")
    path.write_text(text + status_step(final=True), encoding="utf-8")


def patch_test() -> None:
    path = Path("tests/unit/automation/chatgpt_mcp/test_deploy_workflow.py")
    text = path.read_text(encoding="utf-8")
    addition = '''

def test_deploy_publishes_observable_commit_status() -> None:
    text = _workflow_text()

    assert "statuses: write" in text
    assert "Publish pending deployment status" in text
    assert "Publish final deployment status" in text
    assert "if: always()" in text
    assert text.count('"context": "wsl-mcp/deploy"') == 2
    assert '"state": "pending"' in text
    assert 'JOB_STATUS: ${{ job.status }}' in text
'''
    if "def test_deploy_publishes_observable_commit_status" in text:
        raise SystemExit("deployment status test already exists")
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def main() -> None:
    patch_workflow()
    patch_test()


if __name__ == "__main__":
    main()
