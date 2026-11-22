"""Recover an issue POST without repeating an ambiguous creation.

The caller serializes lookup through save, matches immutable provenance across
all issue states/pages, and persists the receipt in its existing transaction.
This module has no ownership, adoption or parent/child policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Protocol


class Writer(Protocol):
    repo: str

    def write(
        self, method: str, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


def create_issue_once(
    gh: Writer,
    *,
    draft: dict[str, Any],
    receipt: dict[str, Any],
    matches: list[Mapping[str, Any]],
    save: Callable[[], None],
) -> dict[str, Any]:
    """Save before POST; an unknown outcome requires reconciliation, not retry.

    ``matches`` comes from the caller's source-specific provenance index, never
    a title search. Mutable title/body/state of an existing issue are preserved.
    A response lost before saving its number is recovered by that same index.
    """
    if (
        set(receipt) != {"draft", "attempted", "number"}
        or type(receipt["attempted"]) is not bool
        or (
            receipt["number"] is not None
            and (type(receipt["number"]) is not int or receipt["number"] < 1)
        )
        or (receipt["attempted"] and receipt["draft"] is None)
        or (receipt["number"] is not None and receipt["draft"] is None)
    ):
        raise ValueError("invalid issue creation receipt")
    if set(draft) != {"title", "body", "labels"}:
        raise ValueError("issue draft requires title/body/labels")
    if receipt["draft"] is not None and receipt["draft"] != draft:
        raise ValueError("issue creation draft changed; reconcile the fixed request")
    if len(matches) > 1:
        raise ValueError("duplicate issue creation provenance")
    if matches:
        existing = dict(matches[0])
        n = existing.get("number")
        if type(n) is not int or n < 1 or receipt["number"] not in {None, n}:
            raise ValueError("issue creation identity changed")
        receipt.update(draft=deepcopy(draft), number=n)
        save()
        return existing
    if receipt["number"] is not None:
        raise ValueError("created issue disappeared or lost its provenance")
    if receipt["attempted"]:
        raise ValueError(
            "prior issue creation outcome is unknown and no matching issue was found; "
            "reconcile GitHub before retrying"
        )
    receipt.update(draft=deepcopy(draft), attempted=True)
    save()  # A failed save must precede, and therefore prevent, the POST.
    result = gh.write("POST", f"repos/{gh.repo}/issues", deepcopy(draft))
    n = result.get("number")
    if type(n) is not int or n < 1:
        raise ValueError("invalid issue creation response; outcome is unknown")
    receipt["number"] = n
    save()
    return result
