"""Structured gh GraphQL I/O, with complete connections and explicit failures."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from typing import Any

from parent.policy import nonempty, number, object_value

PAGE = "pageInfo { hasNextPage endCursor } totalCount"
NESTED_PAGE = "pageInfo { hasNextPage endCursor }"
FIELD = "... on ProjectV2FieldCommon { id databaseId name dataType } ... on ProjectV2SingleSelectField { options { id name } }"
VIEW = f"""id name number layout filter
configuration {{ visibleFields(first:100) {{ nodes {{ ... on ProjectV2FieldCommon {{ id }} }} {NESTED_PAGE} }} }}
groupByFields(first:100) {{ nodes {{ ... on ProjectV2FieldCommon {{ name }} }} {NESTED_PAGE} }}
verticalGroupByFields(first:100) {{ nodes {{ ... on ProjectV2FieldCommon {{ name }} }} {NESTED_PAGE} }}"""
ITEM = """id isArchived content { __typename
... on Issue { id number url repository { nameWithOwner } }
... on PullRequest { id number url repository { nameWithOwner } }
... on DraftIssue { id title } }
stage: fieldValueByName(name:"段階") { ... on ProjectV2ItemFieldSingleSelectValue { name } }
wait: fieldValueByName(name:"待ち理由") { ... on ProjectV2ItemFieldTextValue { text } }
next: fieldValueByName(name:"次の作業") { ... on ProjectV2ItemFieldTextValue { text } }
acceptance: fieldValueByName(name:"親の受入状態") { ... on ProjectV2ItemFieldSingleSelectValue { name } }"""
ISSUE = f"""id number title url body state stateReason repository {{ nameWithOwner }}
parent {{ id number repository {{ nameWithOwner }} }}
assignees(first:100) {{ nodes {{ id login }} {PAGE} }}
labels(first:100) {{ nodes {{ id name }} {PAGE} }}
subIssuesSummary {{ total completed percentCompleted }}
subIssues(first:100,after:$after) {{ nodes {{ id number repository {{ nameWithOwner }} }} {PAGE} }}"""


def complete(connection: Any) -> list[dict[str, Any]]:
    value = object_value(connection)
    rows = value["nodes"]
    if (
        not isinstance(rows, list)
        or len(rows) > 100
        or value["pageInfo"]["hasNextPage"] is not False
    ):
        raise ValueError("incomplete nested GraphQL connection; no partial projection")
    return [object_value(row) for row in rows]


class Projects:
    def api(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        result = subprocess.run(
            ["gh", "api", "graphql", "--input", "-"],
            input=json.dumps(
                {"query": query, "variables": variables}, ensure_ascii=False
            ),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"gh GraphQL failed: {result.stderr.strip()}")
        response = object_value(json.loads(result.stdout))
        if response.get("errors"):
            raise ValueError("GitHub GraphQL returned errors; no partial projection")
        data: dict[str, Any] = object_value(response.get("data"))
        return data

    def project(self, owner: str, project_number: int) -> dict[str, Any]:
        data = self.api(
            """query ProgressProject($owner:String!,$number:Int!) {
              user(login:$owner) { projectV2(number:$number) {
                id number title url closed public viewerCanUpdate owner { ... on User { login databaseId } }
              } }
            }""",
            {
                "owner": nonempty(owner, "Project owner"),
                "number": number(project_number),
            },
        )
        row: dict[str, Any] = object_value(
            object_value(data.get("user")).get("projectV2")
        )
        if (
            row["number"] != project_number
            or row["owner"]["login"].casefold() != owner.casefold()
        ):
            raise ValueError("Project identity mismatch")
        nonempty(row["id"], "Project ID")
        if type(row["viewerCanUpdate"]) is not bool or type(row["closed"]) is not bool:
            raise ValueError("malformed Project permissions/state")
        return row

    def pages(
        self, query: str, variables: dict[str, Any], path: tuple[str, ...]
    ) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
        after = None
        seen_cursors: set[str] = set()
        seen_ids: set[str] = set()
        expected = None
        count = 0
        while True:
            data = self.api(query, {**variables, "after": after})
            parent = data
            for key in path[:-1]:
                parent = object_value(parent.get(key))
            connection = object_value(parent.get(path[-1]))
            total = connection["totalCount"]
            rows = connection["nodes"]
            if (
                type(total) is not int
                or total < 0
                or not isinstance(rows, list)
                or len(rows) > 100
            ):
                raise ValueError("malformed GraphQL connection")
            if expected is not None and total != expected:
                raise ValueError("GraphQL connection changed during pagination")
            expected = total
            validated = []
            for row in rows:
                row = object_value(row)
                node_id = nonempty(row.get("id"), "GraphQL node ID")
                if node_id in seen_ids:
                    raise ValueError("duplicate GraphQL node within/across pages")
                seen_ids.add(node_id)
                validated.append(row)
            count += len(rows)
            page = object_value(connection["pageInfo"])
            has_next = page["hasNextPage"]
            if type(has_next) is not bool:
                raise ValueError("malformed GraphQL pageInfo")
            if has_next:
                cursor = nonempty(page["endCursor"], "GraphQL cursor")
                if not rows or cursor in seen_cursors or count >= total:
                    raise ValueError("invalid/repeated GraphQL cursor")
                seen_cursors.add(cursor)
            elif count != total:
                raise ValueError("GraphQL totalCount differs from fetched rows")
            yield parent, validated
            if not has_next:
                return
            after = cursor

    def connection(
        self, project_id: str, key: str, selection: str
    ) -> list[dict[str, Any]]:
        query = f"""query Progress{key.title()}($id:ID!,$after:String) {{
          node(id:$id) {{ ... on ProjectV2 {{ id {key}(first:100,after:$after) {{ nodes {{ {selection} }} {PAGE} }} }} }}
        }}"""
        result = []
        for parent, rows in self.pages(query, {"id": project_id}, ("node", key)):
            if parent["id"] != project_id:
                raise ValueError("Project connection identity mismatch")
            result.extend(rows)
        return result

    def snapshot(self, owner: str, project_number: int) -> dict[str, Any]:
        project = self.project(owner, project_number)
        fields = self.connection(project["id"], "fields", FIELD)
        views = self.connection(project["id"], "views", VIEW)
        for view in views:
            view["visible"] = [
                row["id"] for row in complete(view["configuration"]["visibleFields"])
            ]
            view["groups"] = [row["name"] for row in complete(view["groupByFields"])]
            view["columns"] = [
                row["name"] for row in complete(view["verticalGroupByFields"])
            ]
        items = self.connection(project["id"], "items", ITEM)
        for item in items:
            content = object_value(item.get("content"))
            nonempty(
                content.get("id"), "Project content ID (deleted/inaccessible item)"
            )
            if type(item["isArchived"]) is not bool:
                raise ValueError("malformed Project archive state")
            values = {}
            for key, name in (
                ("stage", "段階"),
                ("wait", "待ち理由"),
                ("next", "次の作業"),
                ("acceptance", "親の受入状態"),
            ):
                raw = item[key]
                values[name] = (
                    None
                    if raw is None
                    else object_value(raw)[
                        "name" if key in {"stage", "acceptance"} else "text"
                    ]
                )
                if values[name] is not None and not isinstance(values[name], str):
                    raise ValueError("malformed Project field value")
            item["values"] = values
        workflows = self.connection(
            project["id"], "workflows", "id name number enabled"
        )
        return {
            "project": project,
            "fields": fields,
            "views": views,
            "items": items,
            "workflows": workflows,
        }

    def issue(self, repo: str, issue_number: int) -> dict[str, Any]:
        owner, name = repo.split("/")
        result = None
        children = []
        for raw, rows in self.pages(
            f"""query ProgressIssue($owner:String!,$repo:String!,$number:Int!,$after:String) {{
              repository(owner:$owner,name:$repo) {{ issue(number:$number) {{ {ISSUE} }} }}
            }}""",
            {"owner": owner, "repo": name, "number": issue_number},
            ("repository", "issue", "subIssues"),
        ):
            base = {key: value for key, value in raw.items() if key != "subIssues"}
            if result is not None and result != base:
                raise ValueError("issue changed during sub-issue pagination")
            result = base
            children.extend(rows)
        assert result is not None
        if (
            result["number"] != issue_number
            or result["repository"]["nameWithOwner"].casefold() != repo.casefold()
        ):
            raise ValueError("issue identity/cross-repository mismatch")
        for child in children:
            if child["repository"]["nameWithOwner"].casefold() != repo.casefold():
                raise ValueError(
                    "cross-repository sub-issue requires explicit inspection"
                )
        result["children"] = [number(child["number"]) for child in children]
        result["assignees"] = [row["login"] for row in complete(result["assignees"])]
        result["labels"] = [row["name"] for row in complete(result["labels"])]
        return result

    def mutate(
        self, operation: str, input_type: str, payload: dict[str, Any], output: str
    ) -> dict[str, Any]:
        data = self.api(
            f"mutation ProgressWrite($input:{input_type}!) {{ {operation}(input:$input) {{ {output} }} }}",
            {"input": payload},
        )
        result: dict[str, Any] = object_value(data.get(operation))
        return result

    def create_board(
        self, owner_id: int, project_number: int, payload: dict[str, Any]
    ) -> str:
        # REST supports column/group fields at creation. GraphQL's view update
        # has no equivalent; never delete/recreate an existing view to repair it.
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                "--header",
                "Accept: application/vnd.github+json",
                "--header",
                "X-GitHub-Api-Version: 2026-03-10",
                f"users/{number(owner_id)}/projectsV2/{number(project_number)}/views",
                "--input",
                "-",
            ],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"gh REST Project view creation failed: {result.stderr.strip()}"
            )
        view = object_value(object_value(json.loads(result.stdout)).get("value"))
        if any(
            view.get(key) != payload[key]
            for key in (
                "name",
                "layout",
                "filter",
                "visible_fields",
                "group_by",
                "vertical_group_by",
            )
        ):
            raise ValueError(
                "created REST board configuration mismatch; inspect before retry"
            )
        view_id: str = nonempty(view.get("node_id"), "created REST view node ID")
        return view_id
