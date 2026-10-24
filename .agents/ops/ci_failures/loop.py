"""Observe, file or reuse, then verify. No assignment or execution entry points."""

from __future__ import annotations

import fcntl
import html
import json
from typing import Any

from ci_watch.store import Store as ObservationStore
from issues import ci_failure_marker, render_standalone_body

from .github import GitHub
from .observations import group_failures, read_observations
from .recovery import chain_order, recovery
from .store import Store, digest


def safe(value: str) -> str:
    return html.escape(value).replace("`", "&#96;")


def issue_body(signature: str, failure: dict[str, Any], repo: str) -> str:
    sections = {
        "このタスクの目的": "分類済みのmain CI回帰を修正し、修正merge後のmain CIで回復を確認する。",
        "具体的な変更内容・対象範囲": (
            f"workflow: {safe(failure['workflow'])}\njob: {safe(failure['job'])}\n"
            f"test: {safe(failure['test_id'] or 'job全体')}\n"
            f"失敗: {safe(failure['error'])}\n署名: {signature}\n"
            f"検出SHA: {failure['sha']}\n"
            f"根拠: https://github.com/{repo}/actions/runs/{failure['run_id']}\n"
            "分類は原因の仮説。コメントの分類根拠・原本参照を確認して修正範囲を決める。"
        ),
        "対象外": "CI再実行、workflow・保護設定の変更、重大障害時の優先割当・merge停止。",
        "依存関係": "先行依存なし。",
        "作業状況・停止事項": (
            "未割当。既存の予定回の自律着手判断へ渡す。起票は要件合意・共有基盤承認ではない。\n"
            "修正PRのmergeやIssueのcloseだけでは回復済みにしない。"
        ),
        "Acceptance checklist": (
            "- [ ] 署名が示す失敗を修正し、必要な回帰検証を通す\n"
            "  文脈: 分類の仮説を確認して原因を修正するため。\n"
            "  下流への影響: main CIの同じ失敗を繰り返さない。\n"
            "  検証方法・証拠: 修正PRとローカル回帰検証を記録する。\n"
            "- [ ] 修正merge後のmain CIで元の失敗対象の成功を確認する\n"
            "  文脈: mergeと回復確認を区別するため。\n"
            "  下流への影響: 推測だけで修正ループを完了しない。\n"
            "  検証方法・証拠: ci_failuresの回復記録のSHA・run・job・原本を確認する。"
        ),
    }
    body: str = ci_failure_marker(signature) + "\n\n" + render_standalone_body(sections)
    return body


def sync(
    gh: GitHub, source: ObservationStore, state: Store, now: str
) -> dict[str, Any]:
    with (source.directory / "collect.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("CI collector is active; retry after collection") from None
        source.db.execute("BEGIN")
        config = source.config()
        if (
            config["schema_version"] != 1
            or config["branch"] != "main"
            or config["repository"].casefold() != gh.repo.casefold()
        ):
            raise ValueError("CI source repository/branch/schema mismatch")
        state.configure(
            {
                "schema_version": 1,
                "source_dir": str(source.directory.resolve()),
                "source_config": config,
            }
        )
        records, collection, checkpoint = read_observations(source, state)
        groups = group_failures(records, source)
        chain_order(records)
        source.db.rollback()  # Release the source snapshot before any remote I/O.
    if collection is None:
        raise ValueError(
            "CI collection result is missing; finish the source collection first"
        )
    index, pulls = gh.inventory()
    user = gh.get("user")
    if (
        not isinstance(user, dict)
        or not isinstance(user.get("login"), str)
        or not user["login"]
    ):
        raise ValueError("GitHub publisher identity missing")
    publisher = user["login"]
    errors = collection["observation"]["errors"]
    for signature, occurrences in sorted(groups.items()):
        n = index.get(signature)
        related = gh.related(pulls, signature, n)
        code = [
            o
            for o in occurrences
            if o["failure"]["classification"]["category"] == "code_regression"
        ]
        publishing = bool(code or n is not None or related)
        row: dict[str, Any] = {
            "signature": signature,
            "source_dir": str(source.directory.resolve()),
            "occurrences": occurrences,
            "issue": n,
            "prs": [p["number"] for p in related],
            "recovery": None,
            "status": "publication_pending" if publishing else "held",
            "reason": "失敗根拠を保存済み。GitHubへの起票・追記は未完了"
            if publishing
            else "コード回帰以外は方針決定まで起票保留",
            "observation_errors": errors,
        }
        state.stage(row, now)  # Evidence survives a failed or lost GitHub write.
        if code and n is None and not related:
            # Reconcile all states immediately before creation, including lost responses.
            index, pulls = gh.inventory()
            n = index.get(signature)
            related = gh.related(pulls, signature, n)
            if n is None and not related:
                failure = code[0]["failure"]
                created = gh.write(
                    "POST",
                    f"repos/{gh.repo}/issues",
                    {
                        "title": f"main CIを修正する: {failure['workflow']} / {failure['job']}"[
                            :240
                        ],
                        "body": issue_body(signature, failure, gh.repo),
                        "labels": ["bug", "ai-proposed"],
                    },
                )
                n = gh.identity(created)
                index[signature] = n
        row["issue"], row["prs"] = n, [p["number"] for p in related]
        if n is not None or related:
            for occurrence in occurrences:
                failure = occurrence["failure"]
                key = digest(failure)
                marker = f"<!-- ci-failure-update:v1 signature={signature} occurrence={key} -->"
                detail = json.dumps(
                    failure, ensure_ascii=False, sort_keys=True, indent=2
                )
                body = (
                    marker + f"\nmain CIの失敗を観測: SHA {failure['sha']}\n"
                    f"https://github.com/{gh.repo}/actions/runs/{failure['run_id']}\n"
                    "分類と根拠:\n\n"
                    f"<pre>{safe(detail)}</pre>"
                )
                for target in sorted(
                    {*([n] if n else []), *(p["number"] for p in related)}
                ):
                    gh.comment_once(target, marker, body, publisher)
        if code or n is not None or related:
            if errors:
                row["status"], row["reason"] = (
                    "observation_incomplete",
                    "CI収集に取得エラーがあり回復判定を保留",
                )
            else:
                row["status"], row["reason"], row["recovery"] = recovery(
                    row, records, related, gh, state
                )
        if row["recovery"] is not None:
            result = row["recovery"]
            marker = f"<!-- ci-failure-recovery:v1 signature={signature} sha={result['sha']} -->"
            body = (
                marker
                + f"\n回復確認: SHA {result['sha']} / 修正PR #{result['pr']}\n"
                + "\n".join(j["job_url"] for j in result["jobs"])
            )
            for target in sorted(
                {*([n] if n else []), *(p["number"] for p in related)}
            ):
                gh.comment_once(target, marker, body, publisher)
        state.save(row, now)
    state.advance(*checkpoint)
    summary = state.summaries(0, 1, True)
    return {
        "schema_version": 1,
        "signatures": len(groups),
        "errors": errors,
        "processed": not errors,
        "unresolved": summary["total"],
        "next_after": checkpoint[0],
    }
