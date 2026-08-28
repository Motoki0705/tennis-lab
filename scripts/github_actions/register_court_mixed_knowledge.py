#!/usr/bin/env python
"""Finalize the two one-shot Court mixed-source knowledge nodes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(".agents/skills/knowledge-control/scripts").resolve()
sys.path.insert(0, str(SCRIPTS))

from kg_lib import dump_frontmatter, parse_node  # noqa: E402

DENSE_ID = "run-court-mixed-hierarchical-dense-only-v3-b8-a16-e100"
POSE_ID = "run-court-mixed-hierarchical-pose-consistency-v3-b8-a16-e100"
GROUP_ID = "group-court-mixed-hierarchical-v3-pose-consistency"
NODES = Path("knowledge/nodes")


def _number(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def _metrics(node_id: str) -> dict[str, float]:
    raw = parse_node(NODES / f"{node_id}.md").meta.get("metrics") or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{node_id} metrics must be a mapping.")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(name, str) and isinstance(value, (int, float)):
            result[name] = float(value)
    return result


def _metrics_table(metrics: dict[str, float]) -> str:
    if not metrics:
        return "テストメトリクスは保存されていない。"
    rows = ["| Metric | Value |", "|---|---:|"]
    rows.extend(
        f"| `{name}` | {_number(value)} |" for name, value in sorted(metrics.items())
    )
    return "\n".join(rows)


def _comparison_table(
    dense: dict[str, float], pose: dict[str, float]
) -> str:
    rows = [
        "| Metric | Dense only | Pose + consistency | Δ (後者−前者) |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(set(dense) & set(pose)):
        left = dense[name]
        right = pose[name]
        rows.append(
            f"| `{name}` | {_number(left)} | {_number(right)} | "
            f"{_number(right - left)} |"
        )
    if len(rows) == 2:
        return "共通する数値メトリクスはない。"
    return "\n".join(rows)


def _normalize_artifacts(meta: dict[str, Any]) -> None:
    artifacts = meta.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    output_dir = artifacts.get("output_dir")
    if isinstance(output_dir, str):
        prefix = "/tennis-lab/"
        if output_dir.startswith(prefix):
            artifacts["output_dir"] = output_dir.removeprefix(prefix)


def _rewrite_run(
    node_id: str,
    *,
    title: str,
    pose_enabled: bool,
    relations: list[dict[str, str]],
    body: str,
) -> None:
    path = NODES / f"{node_id}.md"
    node = parse_node(path)
    meta = node.meta
    _normalize_artifacts(meta)
    meta["title"] = title
    meta["provider"] = "codex"
    meta["status"] = "done"
    meta["config"] = {
        "model": "hierarchical (DINOv3 ViT-B/16 + transformer encoder + DPT)",
        "sources": "Synthetic Court V3 + TennisCourtDetector",
        "within_batch_mix": "synthetic_court=4, tennis_court_detector=4",
        "processing": "kp + seg + line",
        "augmentation": "pose_safe (long-side 256)",
        "micro_batch": 8,
        "accumulate_grad_batches": 16,
        "effective_batch": 128,
        "max_epochs": 100,
        "seed": 42,
        "pose": pose_enabled,
        "consistency": pose_enabled,
        "pose_scope": "synthetic_court only" if pose_enabled else "disabled",
    }
    meta["parents"] = []
    meta["relations"] = relations
    meta["tags"] = [
        "court-detection",
        "mixed-source",
        "synthetic-court-v3",
        "tennis-court-detector",
        "hierarchical",
        "effective-batch-128",
        *(
            ["pose", "consistency", "synthetic-only-supervision"]
            if pose_enabled
            else ["dense-only"]
        ),
    ]
    path.write_text(
        f"---\n{dump_frontmatter(meta)}---\n\n{body.strip()}\n",
        encoding="utf-8",
    )


def main() -> int:
    dense_metrics = _metrics(DENSE_ID)
    pose_metrics = _metrics(POSE_ID)
    comparison = _comparison_table(dense_metrics, pose_metrics)

    dense_body = f"""
## 考察 / Findings

### 要約
Synthetic Court V3とTennisCourtDetectorを各batchで4件ずつ混合し、KP・SEG・LINEのみで100 epoch学習したmatched baselineである。poseとconsistencyは明示的に無効化した。

### アーキテクチャ詳細
`hierarchical`のDINOv3 ViT-B/16 encoder、8層spatial transformer encoder、DPT decoderを使用した。両sourceへ同じ`pose_safe`変換を適用し、micro-batch 8、gradient accumulation 16でeffective batch 128とした。

### メトリクスの解釈
{_metrics_table(dense_metrics)}

TennisCourtDetectorには明示的test splitがないため、validationをtestとして代用していない。したがって、保存されたtest metricはSynthetic Court V3の明示的test splitに対する値として解釈する。

### アーキテクチャ⇄メトリクスの因果考察
このrun単独からmixed-source学習の因果効果は断定しない。後続runとsource比、augmentation、seed、batch、epochを一致させ、synthetic限定pose/consistency objectiveの追加効果を比較する基準とする。

### 既存実験との比較
今回のmatched comparisonにおけるbaselineであり、直接のparent runは設定していない。

### 次に有効な実験
pose+consistencyをSynthetic Court V3 sampleだけに適用したmatched runと、共通dense metricおよびpose geometry metricを比較する。
"""
    _rewrite_run(
        DENSE_ID,
        title="Court mixed-source hierarchical dense-only (V3, EB128)",
        pose_enabled=False,
        relations=[],
        body=dense_body,
    )

    pose_body = f"""
## 考察 / Findings

### 要約
dense-only baselineと同じmixed batchを使い、poseおよびKP–pose consistencyをSynthetic Court V3の4 sampleだけへ適用したrunである。TennisCourtDetector sampleには両lossの教師もgradientも与えていない。

### アーキテクチャ詳細
dense branchは8 sample全体でKP・SEG・LINEを学習する。一方、`pose_supervision_mask`でsynthetic sampleだけを抽出し、translation、rotation、log-focal、consistencyを計算する。その他のdata、augmentation、seed、effective batch、epochはbaselineと一致する。

### メトリクスの解釈
{_metrics_table(pose_metrics)}

共通metricの差分を以下に示す。Δはpose+consistencyからdense-onlyを引いた値であり、改善方向は各metricの定義に従って判断する。

{comparison}

### アーキテクチャ⇄メトリクスの因果考察
両run間で意図的に変更した要素はsynthetic限定pose/consistency objectiveであるため、共通metric差はこの追加objectiveに関連する観測として扱える。ただし単一seedであり、ばらつきを超える因果効果かは未確定である。

### 既存実験との比較
`{DENSE_ID}`とのmatched comparisonである。TennisCourtDetector側のpose outputを変更してもpose/consistency lossが変わらず、同sampleへのpose gradientが0になるcontractは単体テストで検証している。

### 次に有効な実験
seedを追加して共通dense metricとpose metricの分散を測り、consistency weightおよびwarmupの感度を評価する。
"""
    _rewrite_run(
        POSE_ID,
        title=(
            "Court mixed-source hierarchical synthetic-only pose+consistency "
            "(V3, EB128)"
        ),
        pose_enabled=True,
        relations=[{"to": DENSE_ID, "rel": "compares"}],
        body=pose_body,
    )

    group_meta: dict[str, Any] = {
        "id": GROUP_ID,
        "type": "group",
        "title": "Court mixed-source hierarchical: pose/consistency比較 (V3)",
        "members": [DENSE_ID, POSE_ID],
        "tags": [
            "court-detection",
            "mixed-source",
            "synthetic-court-v3",
            "pose",
            "consistency",
            "ablation",
        ],
    }
    group_body = f"""
## まとめ

Synthetic Court V3とTennisCourtDetectorを同一batch内で4:4に固定したmatched comparisonである。両runは`hierarchical`、`pose_safe`、seed 42、effective batch 128、100 epochを共有し、synthetic限定pose/consistency objectiveの有無だけを比較する。

{comparison}

各metricの定義と収束挙動の詳細は個別run nodeを正本とする。単一seedのため、差分の再現性は追加seedで確認する必要がある。
"""
    (NODES / f"{GROUP_ID}.md").write_text(
        f"---\n{dump_frontmatter(group_meta)}---\n\n{group_body.strip()}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
