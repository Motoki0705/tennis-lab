---
title: "CI短縮を受けてPRごとに検証し、成功をmergeの必須条件にする"
applies_to: all
source: "#1390 の2026-10-30のユーザー合意（issuecomment-6522108331）と #1541"
created: 2026-10-23
last_verified: 2026-10-30
evidence: "#1390 の2026-10-30の方針変更が現在も有効であることと、.agents/ops/review/README.md のmerge gateが対象HEADのPython成功を求めることを確認する"
---

ユーザーはCIが約12分へ短縮したことを受け、PRへのpushごとに検証し、成功をmergeの必須条件にする判断をした。
待ち時間を理由に、CI未完了・失敗のPRをmergeする運用へ独断で戻さない。毎日のCIと自己修復ループは廃止する判断も同時に示した。

**使い方:** コミットに `[skip ci]` を付けない。レビューと指摘解消に加え、対象HEADの `Python` 成功を
[正規のmerge gate](../../ops/review/README.md#merge-gate) で確認してからmergeする。
運用レポート専用memory PRのレビュー等の例外は [限定経路](../../ops/ops-report/README.md#専用memoryの保存) を参照する。
