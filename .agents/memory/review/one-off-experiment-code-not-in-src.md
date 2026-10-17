---
title: "特定の実験だけを進めるためのscript・controllerは、常設のsrcに残さない"
applies_to: all
source: "2026-10-14 のユーザー指示（#1097 の整理、#1109）。同じ整理の指示が過去にも繰り返された"
created: 2026-10-14
last_verified: 2026-10-15
evidence: "#1109 本文、commit 44a6c978d（固定10 run専用の script と CLI を除去し、再利用する学習・評価だけを ablation/ に残した）"
---

ユーザーは、一時的な実験のために書いたコードを常設のsrcに残すことを望まず、レビューの段階でこれを確かめるよう求めている。決まったrunの列挙、再試行、一回限りの集計report、Issue番号を名前に持つmoduleなどが手掛かりになる。

**判定:**
- 再利用する学習・評価・ablationの部品は、そのタスクの既存の構成（例: `ablation/`、`scripts/`）に揃えて残してよい。
- 特定の実験の進行だけに使うコードは、srcから除く。再現に必要なscriptと実行時のcommitは knowledge の run bundle に残す（[再現記録の方針](../../../knowledge/README.md)）。`/tmp` や worktree にしか無い状態にしない。
- 個別のPRで指定された置き場所を、全タスク共通の規則に一般化しない。
- 残すべきでないコードがsrcに入っていたら、マージ前に直すべき指摘として扱う。
