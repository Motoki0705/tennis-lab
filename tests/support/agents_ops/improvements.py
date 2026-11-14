"""Small author-supplied contracts, shared by improvement unit and CLI tests."""

import hashlib
import json
from pathlib import Path
from typing import Any


def extraction_receipt(
    directory: Path,
    *,
    parent: int,
    proposal: int,
    target: int | None,
    body: str,
    decisions: dict[str, str],
) -> Path:
    """Seed the durable send receipt when a fixture supplies canonical events."""
    repo_hash = hashlib.sha256(b'"o/r"').hexdigest()
    path = (
        directory / "improvement-proposals" / repo_hash / f"extraction-{proposal}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "repo": "o/r",
                "parent": parent,
                "keys": sorted(decisions),
                "proposal": proposal,
                "decision_ids": decisions,
                "creation": {
                    "draft": {
                        "title": "改善parent",
                        "body": body,
                        "labels": ["issue:parent"],
                    },
                    "attempted": True,
                    "number": target,
                },
            }
        )
    )
    return path


def candidate_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "再取得の管理を簡素化する",
        "observation": "元の実装は受入済みだが、保存済み結果の取得に重複操作があった。",
        "evidence": "PR #30 の受入確認。再取得操作を2回観測。削減効果は未測定。",
        "scope": "保存済み結果の取得入口とその表示。",
        "out_of_scope": "元のモデル実装、レビュー方針、GPU実行。",
        "behavior": "同じ結果を一度の明示操作で取得できる。",
        "acceptance": [
            "同じ結果の再取得で実装workerを起動しない。",
            "取得失敗を表示し、保存済み結果を保持する。",
        ],
        "stop_conditions": "上記の受入を満たす。追加要件は別途判断する。",
        "judgment": "元の受入条件は達成済み。取得の使い勝手は独立した効果を持つ。",
        "unknowns": "運用上の削減時間は未測定。",
        "expected_effect": "重複した管理操作を減らす。",
        "unmet_acceptance": [],
    }


def finish_for(
    p: Any,
    *,
    parent: int = 1,
    state_dir: Any = None,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    args = ["--state-dir", str(state_dir)] if state_dir is not None else []
    required = json.loads(p.cli(*args, "status", parent=str(parent)).stdout)[
        "completion"
    ]["requirements"]
    assert required["available"], required
    return {
        "schema_version": 1,
        "state": "completed",
        "basis_ref": required["basis_ref"],
        "acceptance": [
            {
                "id": criterion["id"],
                "state": "met",
                "evidence": "親がこの条件を差分・回帰試験・保存済みの受入根拠で確認した。",
            }
            for criterion in required["criteria"]
        ],
        "remaining_required": [],
        "improvements": {"reviewed": True, "candidates": candidates or []},
        "summary": "全体受入を照合した。",
    }
