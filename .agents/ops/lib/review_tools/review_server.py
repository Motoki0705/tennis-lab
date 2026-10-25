"""Read-only retrieval at multiple fixed review commits, plus approved CPU tests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .mcp_server import serve
from .records import JournalRecorder
from .repro_isolation import LinuxRunner
from .reproduction import Reproduction
from .retrieval import Retrieval, RetrievalError, Retrieved, Tool, strict_json
from .retrieval_git import GitReader


class ReviewRetrieval(Retrieval):
    def __init__(self, config: dict[str, Any]) -> None:
        targets = config["commits"]
        if (
            not isinstance(targets, list)
            or not targets
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("review needs unique pinned commits")
        root, journal = Path(config["repo"]), Path(config["journal_dir"])
        super().__init__(
            root, targets[0], config["repository"], config["run_id"], journal
        )
        readers: dict[str, Retrieval] = {targets[0]: self}
        for sha in targets[1:]:
            readers[sha] = Retrieval(
                root, sha, config["repository"], config["run_id"], journal / sha
            )
        self.revisions = set(readers)
        original = {sha: dict(reader.tools) for sha, reader in readers.items()}

        def handler(name: str, revision: str, **arguments: Any) -> Retrieved:
            if revision not in original:
                raise RetrievalError(
                    "FORBIDDEN_REVISION", "revision is outside the fixed review"
                )
            return original[revision][name].handler(**arguments)

        for tool in GitReader(self).tools():
            name = tool.name
            self.tools[name] = replace(
                tool,
                properties={
                    **tool.properties,
                    "revision": {"type": "string", "enum": targets},
                },
                required=("revision", *tool.required),
                handler=lambda revision, _name=name, **args: handler(
                    _name, revision, **args
                ),
            )
        self.executors = {}
        for sha, nodes in config.get("cpu_tests", {}).items():
            if sha not in readers:
                raise ValueError("CPU tests refer to an unpinned commit")
            recorder = JournalRecorder(
                journal / sha / "evidence", config["run_id"], sha
            )
            self.executors[sha] = Reproduction(
                root,
                sha,
                config["run_id"],
                journal / sha / "cpu",
                recorder,
                nodes,
                runner=LinuxRunner(
                    Path(config["isolation_proof"])
                    if config.get("isolation_proof")
                    else None
                ),
            )
        revision = {"revision": {"type": "string", "enum": targets}}
        self.tools["cpu_test_status"] = Tool(
            "cpu_test_status",
            "Inspect approved tests and enforced isolation at a pinned revision.",
            self.status,
            revision,
            ("revision",),
        )
        for executor in self.executors.values():
            for tool in executor.tools():
                if tool.name == "cpu_test_status":
                    continue
                name = tool.name

                def execute(
                    revision: str, _name: str = name, **arguments: Any
                ) -> Retrieved:
                    chosen = self.executors.get(revision)
                    if chosen is None:
                        raise RetrievalError(
                            "FORBIDDEN_REVISION",
                            "no approved CPU tests for this revision",
                        )
                    if _name == "cpu_tests":
                        return chosen.request(arguments)
                    return chosen.log(**arguments)

                self.tools[name] = replace(
                    tool,
                    properties={**tool.properties, **revision},
                    required=("revision", *tool.required),
                    handler=execute,
                )

    def response_revision(self, arguments: Any) -> str:
        revision = arguments.get("revision") if isinstance(arguments, dict) else None
        return (
            revision
            if isinstance(revision, str) and revision in self.revisions
            else self.base_sha
        )

    def status(self, revision: str) -> Retrieved:
        executor = self.executors.get(revision)
        if executor is None:
            return Retrieved(
                {
                    "enabled": False,
                    "reason": "parent must approve an existing CPU-test allowlist",
                },
                [],
            )
        return executor.status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = strict_json(args.config.read_bytes())
    try:
        serve(ReviewRetrieval(config), sys.stdin.buffer, sys.stdout.buffer)
        return 0
    except (OSError, ValueError, RetrievalError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
