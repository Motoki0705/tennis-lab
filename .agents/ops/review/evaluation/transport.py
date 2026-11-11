"""Pinned git reads and recorded GitHub responses, with no network fallback."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from lib.review_tools.retrieval import TEXT, RetrievalError, Retrieved, Tool, text_page
from lib.review_tools.review_server import ReviewRetrieval
from worker_logs.journal import atomic, locked

from .catalog import DATA, case_record, digest, read, validate_input, verify_snapshot


class LocalGitHub:
    """A test/evaluation transport. Its writes are never GitHub publications."""

    def __init__(self, bundle: dict[str, Any], sink: Path) -> None:
        self.responses = bundle["input"]["github"]
        context = bundle["input"]["context"]
        self.targets = {context["parent"]["number"], bundle["case"]["pr"]}
        self.sink = sink

    def get(self, endpoint: str) -> Any:
        if endpoint not in self.responses:
            raise RetrievalError(
                "UNREGISTERED_ENDPOINT", "endpoint was not captured for this case"
            )
        return copy.deepcopy(self.responses[endpoint]["value"])

    def write(
        self, method: str, endpoint: str, value: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = {
            f"repos/Motoki0705/tennis-lab/issues/{n}/comments" for n in self.targets
        }
        if (
            method != "POST"
            or endpoint not in allowed
            or set(value) != {"body"}
            or not isinstance(value["body"], str)
        ):
            raise RetrievalError(
                "FORBIDDEN_WRITE", "only local evaluation comments are supported"
            )
        with locked(self.sink):
            saved: dict[str, Any] = (
                read(self.sink)
                if self.sink.exists()
                else {"schema_version": 1, "local_only": True, "comments": []}
            )
            row = {
                "id": len(saved["comments"]) + 1,
                "endpoint": endpoint,
                "body": value["body"],
                "local_only": True,
            }
            saved["comments"].append(row)
            atomic(self.sink, saved)
        return row


class FrozenRetrieval(ReviewRetrieval):
    tools: dict[str, Tool]

    def __init__(
        self, repo: Path, bundle_path: Path, journal: Path, *, data: Path = DATA
    ) -> None:
        bundle = read(bundle_path)
        if (
            bundle["case"] != case_record(bundle["case"]["pr"], data / "manifest.json")
            or digest(bundle["input"]) != bundle["case"]["input_sha256"]
        ):
            raise ValueError("evaluation bundle differs from the frozen catalog")
        validate_input(bundle["input"], bundle["case"])
        if (bundle_path.parent / "review.diff").read_bytes() != verify_snapshot(
            repo, bundle["case"]
        ):
            raise ValueError("evaluation bundle diff changed")
        snapshot = bundle["case"]["snapshot"]
        commits = sorted(
            {snapshot[k] for k in ("head_sha", "base_sha", "merge_base_sha")}
        )
        super().__init__(
            {
                "repo": str(repo),
                "repository": "Motoki0705/tennis-lab",
                "run_id": bundle["case"]["case_id"],
                "commits": commits,
                "journal_dir": str(journal),
            }
        )
        # Remove every live GitHub, memory, and CPU execution entry. The shared
        # git tools retain immutable revision/path/ancestor and size checks.
        self.tools = {
            name: tool
            for name, tool in self.tools.items()
            if name
            in {
                "code_paths",
                "code_search",
                "code_file",
                "git_history",
                "commit_diff",
            }
        }
        self.frozen = LocalGitHub(bundle, journal / "local-comments.json")
        self.input = bundle["input"]
        self.diff = (bundle_path.parent / "review.diff").read_text()
        self.tools.update(
            {
                "github_get": Tool(
                    "github_get",
                    "Read only a registered historical GitHub response. There is no live fallback.",
                    self.github,
                    {"endpoint": {"type": "string"}, **TEXT},
                    ("endpoint",),
                ),
                "review_context": Tool(
                    "review_context",
                    "Read the frozen requirements and instructions captured before review.",
                    self.context,
                    dict(TEXT),
                ),
                "review_diff": Tool(
                    "review_diff",
                    "Read the pinned full-index binary git diff as text.",
                    self.review_diff,
                    dict(TEXT),
                ),
            }
        )

    def page(self, text: str, artifact: str, **arguments: Any) -> Retrieved:
        data, pagination = text_page(text, **arguments)
        return Retrieved(
            data,
            [
                {
                    "kind": "frozen_evaluation",
                    "artifact": artifact,
                    "as_of": self.input["cutoff"],
                }
            ],
            pagination,
        )

    def github(self, endpoint: str, **arguments: Any) -> Retrieved:
        from worker_logs.schema import json_bytes

        return self.page(
            json_bytes(self.frozen.get(endpoint)).decode(), endpoint, **arguments
        )

    def context(self, **arguments: Any) -> Retrieved:
        from worker_logs.schema import json_bytes

        return self.page(
            json_bytes(self.input["context"]).decode(),
            "historical context",
            **arguments,
        )

    def review_diff(self, **arguments: Any) -> Retrieved:
        return self.page(self.diff, "frozen diff", **arguments)
