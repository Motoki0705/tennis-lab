"""Fresh human-stop checks shared by both repair-worker launch paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .policy import nonempty, object_value

if TYPE_CHECKING:
    from .session import ParentSession


def require_repair_labels(session: ParentSession, child: int, pr: int) -> None:
    targets = (
        ("parent issue", "issues", session.parent),
        ("child issue", "issues", child),
        ("PR", "pulls", pr),
    )
    with session.gh.fresh_reads():
        for kind, resource, number in targets:
            row = object_value(
                session.gh.get(f"repos/{session.gh.repo}/{resource}/{number}")
            )
            if type(row.get("number")) is not int or row["number"] != number:
                raise ValueError("repair launch target identity mismatch")
            labels = row.get("labels")
            if not isinstance(labels, list):
                raise ValueError(
                    f"repair launch labels unavailable for {kind} #{number}"
                )
            names = {
                nonempty(object_value(label).get("name"), "repair target label name")
                for label in labels
            }
            if "needs-human" in names:
                raise ValueError(
                    f"repair launch blocked by needs-human on {kind} #{number}"
                )
