"""Minimal GitHub REST double backed by the test's actual bare Git repository."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


def git(*args: str) -> str:
    return subprocess.check_output(
        [
            os.environ["MEMORY_TEST_GIT"],
            "-c",
            "user.name=Fake GitHub",
            "-c",
            "user.email=github@example.test",
            f"--git-dir={os.environ['MEMORY_TEST_ORIGIN']}",
            *args,
        ],
        text=True,
    ).strip()


def respond(value: object, status: int = 200) -> int:
    print(f"HTTP/2.0 {status}\nContent-Type: application/json\n")
    print(json.dumps(value))
    return int(status >= 400)


def merge(base: str, head: str) -> str:
    tree = git("merge-tree", "--write-tree", base, head)
    sha = git("commit-tree", tree, "-p", base, "-p", head, "-m", "Merge memory PR")
    git("update-ref", "refs/heads/main", sha, base)
    return sha


def main(args: list[str]) -> int:
    assert args[1] == "api" and "--include" in args
    assert args[args.index("--hostname") + 1] == "github.com"
    method = args[args.index("--method") + 1]
    endpoint = next(arg for arg in args if arg.startswith("repos/"))
    data: dict[str, Any] = json.load(sys.stdin) if "--input" in args else {}
    with Path(os.environ["MEMORY_TEST_API_LOG"]).open("a") as trace:
        print(
            json.dumps({"method": method, "path": endpoint, "data": data}), file=trace
        )

    state = Path(os.environ["MEMORY_TEST_PR"])
    case = json.loads(os.environ.get("MEMORY_TEST_CASE", "{}"))
    repo_path = "repos/test-owner/tennis-memory"
    route = f"{repo_path}/pulls"
    if method == "GET" and endpoint == f"{repo_path}/git/ref/heads/main":
        return respond(
            {
                "ref": "refs/heads/main",
                "object": {"type": "commit", "sha": git("rev-parse", "main")},
            }
        )
    if method == "POST" and endpoint == route:
        assert not state.exists()
        assert data["base"] == "main" and data["draft"] is False
        assert git("rev-parse", f"refs/heads/{data['head']}")
        state.write_text(
            json.dumps(
                {
                    **data,
                    "merge_calls": 0,
                    "reads": 0,
                    "base_at_create": git("rev-parse", "main"),
                }
            )
        )
        return respond({"number": 17}, 201)

    pr = json.loads(state.read_text())
    route += "/17"
    branch = f"refs/heads/{pr['head']}"
    base, head = git("rev-parse", "main", branch).splitlines()
    fields = git(
        "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}"
    ).split("\0")
    labels = {"A": "added", "M": "modified", "D": "removed"}
    files = [
        {"filename": path, "status": labels[status]}
        for status, path in zip(fields[:-1:2], fields[1::2], strict=True)
    ]
    if method == "PATCH" and endpoint == route:
        pr.update(body=data["body"])
        state.write_text(json.dumps(pr))
        return respond({"number": 17})
    if method == "GET" and endpoint == route:
        pr["reads"] += 1
        if pr["reads"] == 1 and (next_base := case.get("advance_on_read")):
            git("update-ref", "refs/heads/main", next_base, base)
        state.write_text(json.dumps(pr))
        repo = {"full_name": "test-owner/tennis-memory"}
        record: dict[str, Any] = {
            "number": 17,
            "body": pr["body"],
            "state": "open",
            "merged": False,
            "draft": False,
            "html_url": "https://github.com/test-owner/tennis-memory/pull/17",
            "head": {"ref": pr["head"], "sha": head, "repo": repo.copy()},
            "base": {"ref": "main", "sha": pr["base_at_create"], "repo": repo.copy()},
            "changed_files": len(files),
        }
        for key, value in case.get("pr_fields", {}).items():
            if "." in key:
                outer, inner = key.split(".", 1)
                record[outer][inner] = value
            else:
                record[key] = value
        if case.get("head_after_files") and pr["reads"] == 2:
            record["head"]["sha"] = "0" * 40
        return respond(record)
    if method == "GET" and endpoint.startswith(route + "/files?"):
        query = parse_qs(urlsplit(endpoint).query)
        assert query["per_page"] == ["100"]
        page = int(query.get("page", ["1"])[0])
        if mismatch := case.get("file_fields"):
            files[0].update(mismatch)
        if case.get("duplicate_file"):
            files[0] = files[-1]
        return respond(files[(page - 1) * 100 : page * 100])
    if method == "PUT" and endpoint == route + "/merge":
        assert data == {"merge_method": "merge", "sha": head}
        attempt = pr["merge_calls"]
        pr["merge_calls"] += 1
        state.write_text(json.dumps(pr))
        conflicts = case.get("conflicts", [])
        if attempt < len(conflicts):
            if next_base := case.get("advance_to"):
                git("update-ref", "refs/heads/main", next_base, base)
            elif case.get("advance_empty"):
                tree = git("rev-parse", f"{base}^{{tree}}")
                next_base = git(
                    "commit-tree", tree, "-p", base, "-m", "Parallel update"
                )
                git("update-ref", "refs/heads/main", next_base, base)
            if case.get("broken_conflict_body"):
                print(f"HTTP/2.0 {conflicts[attempt]}\n\nIncomplete API response")
                return 1
            return respond({"message": "merge refused"}, conflicts[attempt])
        merged = merge(base, head)
        if case.get("response_lost_after_merge"):
            return respond({"message": "service failure"}, 503)
        if case.get("change_branch_after_merge"):
            git("update-ref", branch, merged, head)
        return respond({"merged": True, "sha": merged})
    raise AssertionError((method, endpoint))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
