"""Small author-supplied contracts, shared by improvement unit and CLI tests."""

from typing import Any


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
