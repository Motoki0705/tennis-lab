---
title: "機能や方式が伝わる命名を優先し、Issue番号は追跡情報に使う"
type: decision
applies_to: all
source: "2026-10-13のユーザー指示（DINOv3サイズ比較configとPR分割の見直し）"
created: 2026-10-13
last_verified: 2026-10-13
evidence: "新しいconfig・module・branchの名前が役割や方式を説明できるかをレビューし、命名に関する最新のユーザー指示と照合する"
---

ユーザーは、Issue名・番号など一時的な作業の識別子に頼るより、本質的に何を実装しているかが分かる命名を希望している。例えば学習設定は `train_dinov3_size_ablation.yaml` のように、対象と比較方法が伝わる名前にする。

Issue番号はPR本文・関連リンク・実験metadataなどの追跡情報に置く。命名変更で過去の実行証跡を扱うときは、[knowledgeの再現記録方針](../../knowledge/README.md)を確認する。

**使い方:** 名前だけで機能・方式・責務を説明できるかを確かめ、現行の呼び出し元・設定継承・利用例も同じ命名に揃える。
