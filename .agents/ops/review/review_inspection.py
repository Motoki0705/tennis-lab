"""Read current automatic progress and historical records without replaying gates."""

from __future__ import annotations

from typing import Any

from lib.ops_review.merge_gate_impl import state_path, view
from lib.ops_review.progression import read
from lib.ops_review.progression_backend import LiveBackend
from parent.github import GitHub
from parent.state import Store


def inspect_reviews(
    gh: GitHub, store: Store, parent: int, numbers: list[int]
) -> dict[str, Any]:
    with store.consistent_snapshot():
        row = store.parents.get(f"{gh.repo.casefold()}#{parent}", {})
        reviews = list(row.get("reviews", []))
    result: dict[str, Any] = {"parent": parent, "prs": [], "errors": []}
    for number in numbers:
        item: dict[str, Any] = {"number": number, "history": [], "errors": []}
        for registered in reviews:
            request = registered["request"]
            if number not in {p["number"] for p in request["prs"]}:
                continue
            identifier = request["review_run_id"]
            record: dict[str, Any] = {"review_run_id": identifier, "errors": []}
            try:
                if request["schema_version"] == 4:
                    path = state_path(store, identifier)
                    state = read(path)
                    record.update(
                        progress=view(path),
                        changes=LiveBackend(state, store).changes(state),
                    )
                else:
                    directory = store.directory / "reviews" / identifier
                    record.update(
                        legacy=True, request=request, state=registered["state"]
                    )
                    for name in ("result", "execution", "output"):
                        file = directory / (name + ".json")
                        record[name] = read(file) if file.exists() else None
            except (ValueError, OSError) as exc:
                record["errors"].append(str(exc))
            item["history"].append(record)
        result["prs"].append(item)
    return result
