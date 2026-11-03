"""Read committed objects only; neither working files nor executable filters."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .retrieval import (
    COMMIT,
    PAGE,
    STRING,
    TEXT,
    Retrieval,
    RetrievalError,
    Retrieved,
    Tool,
    list_page,
    pagination,
    path_arg,
    text_page,
)

SCOPES = {
    "knowledge": "knowledge",
    "memory": ".agents/memory",
    "ops_memory": ".agents/memory/ops-report",
}


class GitReader:
    def __init__(self, retrieval: Retrieval) -> None:
        self.r = retrieval

    def tools(self) -> list[Tool]:
        scope = {"type": "string", "enum": list(SCOPES)}
        return [
            Tool(
                "code_paths",
                "List committed paths at the fixed base SHA; includes object modes and explicit gitlink boundaries.",
                self.paths,
                {"prefix": STRING, **PAGE},
            ),
            Tool(
                "code_search",
                "Literal UTF-8 text search at the fixed SHA. Binary blobs and submodule internals are excluded.",
                self.search,
                {"text": {**STRING, "minLength": 1}, "prefix": STRING, **PAGE},
                ("text",),
            ),
            Tool(
                "code_file",
                "Read a committed regular UTF-8 file by line range; offset/max_chars continue long lines. Never reads the worktree.",
                self.file,
                {"path": STRING, **TEXT},
                ("path",),
            ),
            Tool(
                "git_history",
                "Page commit metadata reachable from the fixed SHA, optionally restricted to a literal path.",
                self.history,
                {"path": STRING, **PAGE},
            ),
            Tool(
                "commit_diff",
                "Read a commit diff range only when the commit is an ancestor of the fixed SHA. External diff/textconv disabled.",
                self.diff,
                {"commit_sha": COMMIT, "path": STRING, **TEXT},
                ("commit_sha",),
            ),
            Tool(
                "record_paths",
                "Page committed knowledge/shared memory/dedicated ops memory paths. Dedicated memory uses its separately pinned SHA.",
                self.record_paths,
                {"scope": scope, "prefix": STRING, **PAGE},
                ("scope",),
            ),
            Tool(
                "record_file",
                "Read a line/character range within one committed knowledge or memory file; path is relative to the selected scope.",
                self.record_file,
                {"scope": scope, "path": STRING, **TEXT},
                ("scope", "path"),
            ),
        ]

    def source(self, sha: str, path: str = "", **extra: Any) -> dict[str, Any]:
        return {
            "kind": "git",
            "sha": sha,
            "repository": self.r.repository,
            "path": path,
            "url": f"https://github.com/{self.r.repository}/"
            + (f"blob/{sha}/{quote(path)}" if path else f"tree/{sha}"),
            **extra,
        }

    def tree(self, sha: str, prefix: str) -> list[dict[str, str]]:
        prefix = path_arg(prefix, empty=True)
        self.r.acquisitions.append(self.source(sha, prefix))
        args = ["ls-tree", "-r", "-z", sha]
        if prefix:
            # ls-tree accepts literal paths, not git pathspec magic.
            args += ["--", prefix]
        rows = []
        for record in self.r.command.git(*args).split(b"\0"):
            if record:
                header, name = record.split(b"\t", 1)
                mode, kind, oid = header.decode("ascii").split()
                rows.append(
                    {
                        "path": name.decode("utf-8"),
                        "mode": mode,
                        "kind": kind,
                        "object_sha": oid,
                    }
                )
        return rows

    def paths(
        self,
        prefix: str = "",
        page: int = 1,
        per_page: int = 50,
        *,
        sha: str | None = None,
    ) -> Retrieved:
        sha = sha or self.r.base_sha
        data, paging = list_page(self.tree(sha, prefix), page, per_page)
        return Retrieved(
            data,
            [
                self.source(
                    sha,
                    prefix,
                    submodule_policy="gitlinks only; no external source traversal",
                )
            ],
            paging,
        )

    def file(
        self,
        path: str,
        start_line: int = 1,
        line_count: int = 100,
        offset: int = 0,
        max_chars: int = 8000,
        expected_sha256: str | None = None,
        *,
        sha: str | None = None,
    ) -> Retrieved:
        path_arg(path)
        sha = sha or self.r.base_sha
        source = self.source(sha, path)
        rows = self.tree(sha, path)
        row = next((row for row in rows if row["path"] == path), None)
        if row is None:
            raise RetrievalError(
                "NOT_FOUND", "file is absent from the pinned commit", sources=[source]
            )
        if row["kind"] != "blob" or row["mode"] not in {"100644", "100755"}:
            raise RetrievalError(
                "UNSUPPORTED_OBJECT",
                "only regular committed blobs can be read; symlinks/gitlinks are not followed",
                sources=[source],
                mode=row["mode"],
            )
        raw = self.r.command.git("cat-file", "blob", row["object_sha"])
        if b"\0" in raw:
            raise RetrievalError(
                "BINARY_SOURCE",
                "binary file is not a UTF-8 text source",
                sources=[source],
            )
        data, paging = text_page(
            raw.decode("utf-8"),
            start_line,
            line_count,
            offset,
            max_chars,
            expected_sha256,
        )
        source.update(
            object_sha=row["object_sha"],
            start_line=start_line,
            line_count=line_count,
            offset=offset,
            end_offset=paging["end_offset"],
        )
        return Retrieved(data, [source], paging)

    def search(
        self, text: str, prefix: str = "", page: int = 1, per_page: int = 50
    ) -> Retrieved:
        path_arg(prefix, empty=True)
        if "\0" in text or "\n" in text or "\r" in text:
            raise RetrievalError(
                "INVALID_ARGUMENT", "search text must be one literal line"
            )
        sha = self.r.base_sha
        self.r.acquisitions.append(self.source(sha, prefix, search=text))
        args = ["grep", "-I", "-n", "-z", "-F", "--full-name", "-e", text, sha, "--"]
        if prefix:
            args += [":(literal)" + prefix]
        raw = self.r.command.git(*args, ok=(0, 1))
        rows = []
        while raw:
            name, number, rest = raw.split(b"\0", 2)
            line, separator, raw = rest.partition(b"\n")
            if not separator:
                raw = b""
            name_text = name.decode("utf-8")
            if not name_text.startswith(sha + ":"):
                raise RetrievalError(
                    "INVALID_SOURCE", "git grep returned a different revision"
                )
            path = name_text[len(sha) + 1 :]
            rows.append(
                {
                    "path": path,
                    "line": int(number),
                    "text": line.decode("utf-8"),
                    "url": self.source(sha, path)["url"] + f"#L{int(number)}",
                }
            )
        data, paging = list_page(rows, page, per_page)
        return Retrieved(
            data,
            [
                self.source(
                    sha,
                    prefix,
                    search=text,
                    excluded="binary blobs and submodule internals",
                )
            ],
            paging,
        )

    def history(self, path: str = "", page: int = 1, per_page: int = 50) -> Retrieved:
        path_arg(path, empty=True)
        self.r.acquisitions.append(self.source(self.r.base_sha, path))
        if (
            self.r.command.git("rev-parse", "--is-shallow-repository").strip()
            != b"false"
        ):
            raise RetrievalError(
                "INCOMPLETE_HISTORY",
                "shallow history cannot establish the requested scope",
            )
        args = [
            "log",
            "--no-show-signature",
            "--no-notes",
            "-z",
            "--format=%H%x00%P%x00%aI%x00%s",
            f"--skip={(page - 1) * per_page}",
            f"--max-count={per_page + 1}",
            self.r.base_sha,
            "--",
        ]
        if path:
            args += [":(literal)" + path]
        fields = self.r.command.git(*args).split(b"\0")
        if fields[-1] == b"":
            fields.pop()
        if len(fields) % 4:
            raise RetrievalError("INVALID_SOURCE", "malformed git history record")
        rows = []
        for i in range(0, len(fields), 4):
            sha, parents, authored_at, subject = (
                f.decode("utf-8") for f in fields[i : i + 4]
            )
            rows.append(
                {
                    "sha": sha,
                    "parents": parents.split(),
                    "authored_at": authored_at,
                    "subject": subject,
                    "url": f"https://github.com/{self.r.repository}/commit/{sha}",
                }
            )
        return Retrieved(
            rows[:per_page],
            [self.source(self.r.base_sha, path)],
            pagination(page, per_page, len(rows) > per_page),
        )

    def diff(self, commit_sha: str, path: str = "", **ranges: Any) -> Retrieved:
        path_arg(path, empty=True)
        self.r.acquisitions.append(
            self.source(self.r.base_sha, path, commit_sha=commit_sha)
        )
        # A commit outside the pinned ancestry must not leak a later checkout's code.
        self.r.command.git("merge-base", "--is-ancestor", commit_sha, self.r.base_sha)
        args = [
            "show",
            "--format=fuller",
            "--no-show-signature",
            "--no-notes",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            commit_sha,
            "--",
        ]
        if path:
            args += [":(literal)" + path]
        data, paging = text_page(self.r.command.git(*args).decode("utf-8"), **ranges)
        return Retrieved(
            data,
            [
                self.source(
                    self.r.base_sha,
                    path,
                    commit_sha=commit_sha,
                    url=f"https://github.com/{self.r.repository}/commit/{commit_sha}",
                )
            ],
            paging,
        )

    def record_paths(
        self, scope: str, prefix: str = "", page: int = 1, per_page: int = 50
    ) -> Retrieved:
        path_arg(prefix, empty=True)
        root = SCOPES[scope]
        sha = self.r.ops_memory_sha if scope == "ops_memory" else self.r.base_sha
        rows = self.tree(sha, root + "/" + prefix)
        if scope == "memory":
            rows = [
                row
                for row in rows
                if not row["path"].startswith(SCOPES["ops_memory"] + "/")
            ]
        data, paging = list_page(rows, page, per_page)
        return Retrieved(
            data, [self.source(sha, root + "/" + prefix, scope=scope)], paging
        )

    def record_file(self, scope: str, path: str, **ranges: Any) -> Retrieved:
        path_arg(path)
        if scope == "memory" and (
            path == "ops-report" or path.startswith("ops-report/")
        ):
            raise RetrievalError(
                "INVALID_ARGUMENT", "dedicated memory must use scope=ops_memory"
            )
        return self.file(
            SCOPES[scope] + "/" + path,
            sha=self.r.ops_memory_sha if scope == "ops_memory" else self.r.base_sha,
            **ranges,
        )
