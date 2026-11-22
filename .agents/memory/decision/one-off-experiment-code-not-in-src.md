---
title: "特定の実験だけを進めるためのscript・controllerは、常設のsrcに残さない"
summary: "一度限りの実験進行用コードは常設のsrcから除き、再現に必要なscriptと実行時点の記録をknowledgeに残す。"
applies_to: all
source: "2026-10-14 のユーザー指示（#1097 の整理、#1109）。同じ整理の指示が過去にも繰り返された"
created: 2026-10-14
last_verified: 2026-11-20
evidence: "#1109 本文、commit 08ea733650aeb374f10871043313bc031a3b36a2 の ablation 構成と knowledge/README.md の保存基準を照合する。"
citations:
  - path: "knowledge/README.md"
    symbol: "#run成果物の保存基準"
  - path: "src/tasks/blcs/ablation/README.md"
    symbol: "ablation"
    note: "再利用する比較部品の配置例"
---

ユーザーは、一時的な実験のために書いたコードを常設のsrcに残すことを望まず、レビューの段階でこれを確かめるよう求めている。決まったrunの列挙、再試行、一回限りの集計report、Issue番号を名前に持つmoduleなどが手掛かりになる。

**判定:**
- 再利用する学習・評価・ablationの部品は、そのタスクの既存の構成（例: `ablation/`、`scripts/`）に揃えて残してよい。
- 特定の実験の進行だけに使うコードは、srcから除く。再現に必要なscriptと実行時のcommitは knowledge の run bundle に残す（[再現記録の方針](../../../knowledge/README.md#run成果物の保存基準)）。
- 個別のPRで指定された置き場所を、全タスク共通の規則に一般化しない。
- 残すべきでないコードがsrcに入っていたら、マージ前に直すべき指摘として扱う。
