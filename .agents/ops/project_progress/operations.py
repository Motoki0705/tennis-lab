"""Plan first; only an explicitly invoked, owning parent can apply a projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from parent.github import GitHub
from parent.policy import nonempty, number, object_value

from .github import FIELD, Projects
from .progress import (
    cancelled,
    completed,
    parent_issue,
    progress,
    progress_text,
    read_tree,
    related_pulls,
)
from .schema import (
    ACCEPTANCE,
    BUILTINS,
    EVENTS,
    FIELDS,
    STAGES,
    SUMMARY_PREFIX,
    VIEWS,
    manual_steps,
    named,
    validate_fields,
)

if TYPE_CHECKING:
    from parent.session import ParentSession


def setup_plan(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    fields = validate_fields(snapshot["fields"], allow_missing=True)
    views = named(snapshot["views"])
    plan: list[dict[str, Any]] = [
        {"operation": "create_field", "name": name, "options": list(options)}
        for name, options in FIELDS.items()
        if name not in fields
    ]
    visible = [*BUILTINS, *FIELDS]
    for spec in VIEWS:
        current = views.get(spec["name"])
        if current is None:
            plan.append(
                {
                    "operation": "create_view",
                    **spec,
                    "visible": visible,
                    **(
                        {"groups": ["Parent issue"], "columns": ["段階"]}
                        if spec["layout"] == "BOARD_LAYOUT"
                        else {}
                    ),
                }
            )
        elif (
            current["layout"] != spec["layout"]
            or (current["filter"] or "") != spec["filter"]
            or sorted(current["visible"])
            != sorted(fields[name]["id"] for name in visible if name in fields)
            or any(name not in fields for name in visible)
        ):
            plan.append(
                {
                    "operation": "update_view",
                    "id": current["id"],
                    **spec,
                    "visible": visible,
                }
            )
    return plan


def writable(snapshot: dict[str, Any]) -> None:
    project = snapshot["project"]
    if project["closed"] or not project["viewerCanUpdate"]:
        raise ValueError("Project is closed or current gh account cannot update it")


def setup(
    api: Projects, owner: str, project_number: int, apply: bool
) -> dict[str, Any]:
    snapshot = api.snapshot(owner, project_number)
    writable(snapshot)
    plan = setup_plan(snapshot)
    if any(action.get("columns") for action in plan):
        nonempty(snapshot["project"]["owner"].get("login"), "Project owner login")
        for field in snapshot["fields"]:
            if field["name"] in (*BUILTINS, *FIELDS):
                number(field.get("databaseId"))
    if apply:
        project_id = snapshot["project"]["id"]
        for action in plan:
            if action["operation"] == "create_field":
                options = action["options"]
                payload: dict[str, Any] = {
                    "projectId": project_id,
                    "name": action["name"],
                    "dataType": "SINGLE_SELECT" if options else "TEXT",
                }
                if options:
                    colors = ("GRAY", "BLUE", "YELLOW", "PURPLE", "ORANGE", "GREEN")
                    payload["singleSelectOptions"] = [
                        {"name": name, "description": name, "color": colors[i]}
                        for i, name in enumerate(options)
                    ]
                created = api.mutate(
                    "createProjectV2Field",
                    "CreateProjectV2FieldInput",
                    payload,
                    f"projectV2Field {{ {FIELD} }}",
                )
                nonempty(
                    object_value(created.get("projectV2Field")).get("id"),
                    "created field ID",
                )
            else:
                # New field IDs come from GitHub, including after a previous interrupted setup.
                fields = validate_fields(api.connection(project_id, "fields", FIELD))
                config = {
                    "visibleFieldIds": [
                        fields[name]["id"] for name in action["visible"]
                    ]
                }
                if action.get("columns"):
                    api.create_board(
                        snapshot["project"]["owner"]["login"],
                        project_number,
                        {
                            "name": action["name"],
                            "layout": "board",
                            "filter": action["filter"],
                            "visible_fields": [
                                number(fields[name].get("databaseId"))
                                for name in action["visible"]
                            ],
                            "group_by": [
                                number(fields[name].get("databaseId"))
                                for name in action["groups"]
                            ],
                            "vertical_group_by": [
                                number(fields[name].get("databaseId"))
                                for name in action["columns"]
                            ],
                        },
                    )
                    continue
                if action["operation"] == "create_view":
                    created = api.mutate(
                        "createProjectV2View",
                        "CreateProjectV2ViewInput",
                        {
                            "projectId": project_id,
                            "name": action["name"],
                            "layout": action["layout"],
                            "configuration": config,
                        },
                        "projectV2View { id }",
                    )
                    view_id = nonempty(
                        object_value(created.get("projectV2View")).get("id"),
                        "created view ID",
                    )
                else:
                    view_id = action["id"]
                updated = api.mutate(
                    "updateProjectV2View",
                    "UpdateProjectV2ViewInput",
                    {
                        "viewId": view_id,
                        "layout": action["layout"],
                        "filter": action["filter"],
                        "configuration": config,
                    },
                    "projectV2View { id }",
                )
                if object_value(updated.get("projectV2View")).get("id") != view_id:
                    raise ValueError("updated view identity mismatch")
        snapshot = api.snapshot(owner, project_number)
        if setup_plan(snapshot):
            raise ValueError(
                "Project setup did not converge; rerun with the same input"
            )
    return {
        "dry_run": not apply,
        "project": snapshot["project"],
        "plan": plan,
        "views": [
            {
                "name": view["name"],
                "url": f"{snapshot['project']['url']}/views/{view['number']}",
            }
            for view in snapshot["views"]
            if view["name"] in {spec["name"] for spec in VIEWS}
        ],
        "manual_steps": manual_steps(snapshot["views"]),
        "workflow_check": "UIで既存workflowの対象項目・値・auto-add条件を確認する。APIはname/enabledだけを取得可能。",
    }


def observe(
    gh: GitHub, api: Projects, parent: int, owner: str, project_number: int
) -> dict[str, Any]:
    snapshot = api.snapshot(owner, project_number)
    tree = read_tree(api, gh.repo, parent)
    pulls = related_pulls(gh, tree)
    by_content: dict[str, dict[str, Any]] = {}
    for item in snapshot["items"]:
        content_id = item["content"]["id"]
        if content_id in by_content:
            raise ValueError("duplicate Project item for the same content")
        by_content[content_id] = item
    items = {n: by_content.get(issue["id"]) for n, issue in tree.items()}
    return {"snapshot": snapshot, "tree": tree, "pulls": pulls, "items": items}


def inspection(observation: dict[str, Any]) -> dict[str, Any]:
    snapshot, tree, pulls, items = (
        observation[key] for key in ("snapshot", "tree", "pulls", "items")
    )
    return {
        "project": snapshot["project"],
        "fields": snapshot["fields"],
        "views": snapshot["views"],
        "workflows": snapshot["workflows"],
        "project_item_count": len(snapshot["items"]),
        "manual_steps": manual_steps(snapshot["views"]),
        "issues": [
            {
                "number": n,
                "title": issue["title"],
                "url": issue["url"],
                "state": issue["state"],
                "state_reason": issue["stateReason"],
                "parent": issue["parent"],
                "children": [
                    {
                        "number": child,
                        "title": tree[child]["title"],
                        "url": tree[child]["url"],
                    }
                    for child in issue["children"]
                ],
                "assignees": issue["assignees"],
                "pull_requests": pulls[n],
                "project_item": None
                if items[n] is None
                else {
                    "id": items[n]["id"],
                    "archived": items[n]["isArchived"],
                    "values": items[n]["values"],
                },
                "progress": progress(tree, n),
            }
            for n, issue in tree.items()
        ],
    }


def validate_spec(
    spec: dict[str, Any], observation: dict[str, Any], parent: int, event: str
) -> dict[int, dict[str, Any]]:
    if (
        event not in EVENTS
        or set(spec) != {"schema_version", "items"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise ValueError(
            "expected projection spec schema_version=1 and items, with a known event"
        )
    if not isinstance(spec["items"], list) or not spec["items"]:
        raise ValueError("items must be a nonempty list")
    tree, items, pulls = (observation[key] for key in ("tree", "items", "pulls"))
    result = {}
    for raw in spec["items"]:
        raw = object_value(raw)
        keys = {"issue", "stage", "wait_reason", "next_action"}
        if not keys <= raw.keys() or raw.keys() - (keys | {"acceptance", "pr"}):
            raise ValueError(
                "each item requires issue/stage/wait_reason/next_action; only acceptance/pr are optional"
            )
        n = number(raw["issue"])
        if n in result or n not in tree:
            raise ValueError(f"#{n}: duplicate or outside this formal subtree")
        issue = tree[n]
        is_parent = parent_issue(issue, parent)
        stage = raw["stage"]
        if (stage is not None and stage not in STAGES) or (
            stage is None and not cancelled(issue)
        ):
            raise ValueError(f"#{n}: invalid stage (null is only for cancelled issues)")
        if not isinstance(raw["wait_reason"], str):
            raise ValueError(
                "wait_reason must be text; use an empty string to clear it"
            )
        wait = raw["wait_reason"].strip()
        next_action = nonempty(raw["next_action"], "next_action")
        if SUMMARY_PREFIX in next_action:
            raise ValueError(
                "next_action contains the reserved progress summary prefix"
            )
        acceptance = raw.get("acceptance")
        if is_parent and acceptance not in ACCEPTANCE:
            raise ValueError(f"#{n}: parent acceptance must be explicit")
        if not is_parent and "acceptance" in raw:
            raise ValueError(f"#{n}: acceptance is only for parents")
        if cancelled(issue) and (stage is not None or not wait):
            raise ValueError(
                f"#{n}: cancelled issues require null stage and an explicit wait_reason"
            )
        if (
            not is_parent
            and not cancelled(issue)
            and ((stage == "完了") != completed(issue))
        ):
            raise ValueError(
                f"#{n}: completion stage must match closed/completed, independently of PRs"
            )
        if is_parent and ((stage == "完了") != (acceptance == "受入済み")):
            raise ValueError(f"#{n}: parent completion requires explicit acceptance")
        current = items[n]
        if current is not None and current["isArchived"]:
            raise ValueError(
                f"#{n}: archived Project item; explicitly unarchive it before retry"
            )
        if acceptance == "受入済み":
            old = None if current is None else current["values"]["親の受入状態"]
            if event != "accept" and old != "受入済み":
                raise ValueError(
                    f"#{n}: use the accept event to record parent acceptance"
                )
            if any(
                not cancelled(tree[child]) and not completed(tree[child])
                for child in issue["children"]
            ):
                raise ValueError(f"#{n}: unfinished direct children prevent acceptance")
            leaves = progress(tree, n)["leaves"]
            if leaves["completed"] != leaves["total"]:
                raise ValueError(
                    f"#{n}: unfinished descendant tasks prevent acceptance"
                )
        if event == "start" and (stage != "作業中" or issue["state"] != "OPEN"):
            raise ValueError("start requires an open issue and 作業中")
        if event in {"pr", "merge"}:
            pr = number(raw.get("pr"))
            match = next((row for row in pulls[n] if row["number"] == pr), None)
            expected = "open" if event == "pr" else "merged"
            if match is None or match["state"] != expected:
                raise ValueError(f"#{n}: selected PR must be related and {expected}")
            expected_stage = (
                "レビュー中" if event == "pr" else "受入待ち" if is_parent else "完了"
            )
            if stage != expected_stage:
                raise ValueError(f"{event} requires {expected_stage}")
        elif "pr" in raw:
            raise ValueError("pr is only accepted for pr/merge events")
        if event in {"stop", "resume"}:
            if current is None or current["values"]["段階"] != stage:
                raise ValueError("stop/resume must preserve the existing stage")
            if (event == "stop" and not wait) or (event == "resume" and wait):
                raise ValueError("stop needs a wait_reason; resume must clear it")
        if event == "accept" and (not is_parent or acceptance != "受入済み" or wait):
            raise ValueError("accept requires a parent, 受入済み and no wait_reason")
        result[n] = {
            "段階": stage,
            "待ち理由": wait or None,
            "次の作業": next_action,
            "親の受入状態": acceptance,
        }
    if event == "initial" and result.keys() != tree.keys():
        raise ValueError(
            "initial requires explicit values for every issue in the formal subtree"
        )
    if any(item is None and n not in result for n, item in items.items()):
        raise ValueError(
            "unregistered issues lack explicit values; include them in the spec"
        )
    return result


def sync_plan(
    observation: dict[str, Any], parent: int, event: str, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    snapshot, tree, items = (observation[key] for key in ("snapshot", "tree", "items"))
    writable(snapshot)
    validate_fields(snapshot["fields"])
    updates = validate_spec(spec, observation, parent, event)
    # Refresh ancestors' denominators without changing their acceptance or the parent's next task.
    for n, issue in tree.items():
        if not parent_issue(issue, parent):
            continue
        if n not in updates:
            item = items[n]
            if (
                item is None
                or item["isArchived"]
                or not item["values"]["次の作業"]
                or item["values"]["親の受入状態"] not in ACCEPTANCE
                or (item["values"]["段階"] not in STAGES and not cancelled(issue))
            ):
                raise ValueError(
                    f"#{n}: initialize/unarchive the parent projection first"
                )
            updates[n] = dict(item["values"])
        next_action = updates[n]["次の作業"].split(SUMMARY_PREFIX)[0]
        updates[n]["次の作業"] = (
            next_action + SUMMARY_PREFIX + progress_text(progress(tree, n))
        )
    plan = []
    for n, values in updates.items():
        item = items[n]
        if item is None:
            plan.append(
                {"operation": "add_item", "issue": n, "content_id": tree[n]["id"]}
            )
        for name, value in values.items():
            old = None if item is None else item["values"][name]
            if old != value:
                plan.append(
                    {
                        "operation": "clear_value" if value is None else "set_value",
                        "issue": n,
                        "field": name,
                        "value": value,
                    }
                )
    return plan


def fingerprint(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "tree": observation["tree"],
        "pulls": observation["pulls"],
        "items": observation["items"],
        "project": observation["snapshot"]["project"],
        "fields": observation["snapshot"]["fields"],
    }


def sync(
    gh: GitHub,
    api: Projects,
    parent: int,
    owner: str,
    project_number: int,
    apply: bool,
    event: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    observation = observe(gh, api, parent, owner, project_number)
    plan = sync_plan(observation, parent, event, spec)
    if apply and plan:
        latest = observe(gh, api, parent, owner, project_number)
        if fingerprint(latest) != fingerprint(observation):
            raise ValueError(
                "Project/issues/PRs changed before writing; inspect and retry"
            )
        project_id = observation["snapshot"]["project"]["id"]
        fields = validate_fields(observation["snapshot"]["fields"])
        items = observation["items"]
        for action in plan:
            n = action["issue"]
            if action["operation"] == "add_item":
                added = api.mutate(
                    "addProjectV2ItemById",
                    "AddProjectV2ItemByIdInput",
                    {"projectId": project_id, "contentId": action["content_id"]},
                    "item { id }",
                )
                items[n] = {
                    "id": nonempty(
                        object_value(added.get("item")).get("id"), "added item ID"
                    )
                }
                continue
            item_id = items[n]["id"]
            field = fields[action["field"]]
            payload: dict[str, Any] = {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field["id"],
            }
            if action["operation"] == "clear_value":
                updated = api.mutate(
                    "clearProjectV2ItemFieldValue",
                    "ClearProjectV2ItemFieldValueInput",
                    payload,
                    "projectV2Item { id }",
                )
            else:
                value = action["value"]
                payload["value"] = (
                    {
                        "singleSelectOptionId": next(
                            option["id"]
                            for option in field["options"]
                            if option["name"] == value
                        )
                    }
                    if field["dataType"] == "SINGLE_SELECT"
                    else {"text": value}
                )
                updated = api.mutate(
                    "updateProjectV2ItemFieldValue",
                    "UpdateProjectV2ItemFieldValueInput",
                    payload,
                    "projectV2Item { id }",
                )
            if object_value(updated.get("projectV2Item")).get("id") != item_id:
                raise ValueError("updated Project item identity mismatch")
        observation = observe(gh, api, parent, owner, project_number)
        if sync_plan(observation, parent, event, spec):
            raise ValueError(
                "Project update did not converge; inspect and rerun with the same input"
            )
    return {
        "dry_run": not apply,
        "event": event,
        "plan": plan,
        **inspection(observation),
    }


def parent_command(
    session: ParentSession,
    command: str,
    owner: str,
    project_number: int,
    apply: bool,
    event: str | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session.owner()
    api = Projects()
    if command == "project-setup":
        return setup(api, owner, project_number, apply)
    if spec is None or event is None:
        raise ValueError("project-sync requires event and spec")
    session.public(spec)
    return sync(
        session.gh, api, session.parent, owner, project_number, apply, event, spec
    )
