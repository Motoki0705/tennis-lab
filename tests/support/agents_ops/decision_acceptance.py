"""Acceptance scenarios inspired by #1790, not reconstructed historical runs.

The fetch-only branch is explicitly hypothetical: the historical CI incident
actually led to another review. The recovery reason is grounded in the public
comment on #1698; missing decisions in the runtime-only snapshot stay missing.
All IDs, times and counters here are synthetic. No private logs are copied.
"""

from __future__ import annotations

import copy
from typing import Any

from tests.support.agents_ops.decision_logs import (
    action,
    decision,
    missing,
    observed,
    result,
    schema,
)


def incident(name: str) -> list[dict[str, Any]]:
    before = result()
    before["scope"].update(issue=None, delegation=None)
    before["data"].update(
        summary="CIログ取得失敗。コードの静的確認は保存済み。",
        final_report=missing("not a final-report observation"),
        evidence=[
            {
                "kind": "public",
                "url": "https://github.com/Motoki0705/tennis-lab/issues/1790",
                "revision": None,
                "content_sha256": "d776a6de21a5b5eb9ab7f29411ac014d80434a0a5ee0a85a43968166018741cc",
            }
        ],
    )
    if name == "fetch_only":
        choice = decision([before])
        choice["scope"] = copy.deepcopy(before["scope"])
        choice["data"].update(
            reason="仮想事例: コード確認を再利用し、不足するCIログの取得だけを再試行する。",
            reuse_scope=["保存済みのコード確認"],
            recheck_scope=["Python jobのログ"],
        )
        choice["data"]["next_actions"][0].update(
            summary="Python jobのログを再取得", target="PR 1615 / Python job"
        )
        request = action(choice)
        request["scope"] = copy.deepcopy(before["scope"])
        after = result(previous=before, action=request)
        after["scope"] = copy.deepcopy(before["scope"])
        after["data"].update(
            status="success", summary="CIログの取得完了。AIの追加起動なし。"
        )
        after["data"]["changes"]["conditions"] = "unconfirmed"
        ended = action(choice, previous=request, state="ended", results=[after])
        ended["scope"] = copy.deepcopy(before["scope"])
        return [before, choice, request, after, ended]
    before["data"].update(
        summary="runtime_error。保存済み最終報告にはmajor 1件。正式結果はfailed / not_reviewed。",
        final_report=observed(
            {
                "kind": "unavailable",
                "reason": "保存済み報告の本文は非公開。#1790 の公開監査を参照。",
            }
        ),
    )
    if name == "useful_runtime_failure":
        # The historical intermediate decision was not captured. Do not add one.
        return [before]
    if name != "recovery_refused":
        raise ValueError(name)
    refused = result(previous=before)
    refused["scope"] = copy.deepcopy(before["scope"])
    refused["data"].update(
        status="rejected",
        summary="failed結果のresolveは拒否。保存済み指摘は親が確認し、workerが修正済み。",
    )
    refused["data"]["changes"] = dict.fromkeys(schema.CHANGE_FIELDS, "unconfirmed")
    refused["data"]["changes"].update(code="changed", verified_scope="changed")
    choice = decision([before, refused], operation="review")
    choice["scope"] = copy.deepcopy(before["scope"])
    choice["data"].update(
        reason="公開された再レビュー理由: failed・not_reviewedのため指摘をresolveで記録できず、修正後のHEADを正式にレビューし直す。",
        reuse_scope=["親が確認した保存済みmajor指摘とworkerの修正"],
        recheck_scope=["修正後HEADの正式レビュー"],
        unresolved=["正式なreview coverage"],
    )
    choice["data"]["source"]["evidence"] = [
        {
            "kind": "public",
            "url": "https://github.com/Motoki0705/tennis-lab/issues/1698#issuecomment-6729672245",
            "revision": None,
            "content_sha256": "a320510dff5b7b5123f2ddaffeec6cda583e1efbf593cb673151e6e542bdb511",
        }
    ]
    # Fingerprint of the public comment body read during acceptance design.
    # The fixture stores the reference, not the original comment's full text.
    choice["data"]["next_actions"][0].update(
        summary="修正後HEADに新規レビューを要求", target="PR 1769 / 修正後HEAD"
    )
    request = action(choice)
    request["scope"] = copy.deepcopy(before["scope"])
    return [before, refused, choice, request]


def large_history(count: int, *, many_activities: bool = False) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        row = result()
        row["scope"].update(issue=None, delegation=None)
        if many_activities:
            row["scope"]["activity_key"] = schema.new_id()
        row["data"]["summary"] = (
            f"取得確認 {index}: 観測済みの結果。" + "表示サイズ検証。" * 20
        )
        rows.append(row)
    return rows
