---
title: "独立した目的の変更は別PRに分ける"
summary: "独立した目的の変更は別PRに分ける。同じ目的を成立させる実装・テスト・文書はまとめる。"
applies_to: all
source: "2026-10-13のユーザー指示（Colab・Drive skillと定期監視・review運用のPR分割）"
created: 2026-10-13
last_verified: 2026-11-20
evidence: "#1087の補足と、AGENTS.md・.agents/README.md・現在のPR受入条件を照合する。"
citations:
  - path: ".agents/skills/parent-session/references/review.md"
    symbol: "#依頼する前に"
    note: "元の受入条件とPRの照合"
---

ユーザーは、作業中に見つかった独立した改善を一つのPRへ混ぜず、目的ごとにレビュー・採用できる粒度を希望している。Colabの操作、定期監視の復旧、review運用の修正のように責務が異なる変更は分ける。

一つの目的を成立させる実装・回帰テスト・必要なドキュメントは同じPRに含める。依存がある場合はbaseと依存PRを明示し、切り出した差分が元PRにも残らないよう確認する。ファイル数やPR本数だけを基準に分割しない。

**使い方:** PR作成前に変更の目的と完了条件を一文で説明し、その説明に収まらない変更を別PRへ切り出す。各PRの差分と検証結果を個別に確認する。
