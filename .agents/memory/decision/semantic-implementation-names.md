---
title: "機能や方式が伝わる命名を優先し、Issue番号は追跡情報に使う"
summary: "実装名から機能・方式が伝わるようにし、Issue番号は追跡情報に置く。knowledgeの既存run IDは保全する。"
applies_to: all
source: "2026-10-13のユーザー指示（DINOv3サイズ比較configとPR分割の見直し）、#1087 / #1094 の指摘"
created: 2026-10-13
last_verified: 2026-11-20
evidence: "#1087 / #1094 と現行の knowledge/README.md のrun ID規約を照合する。"
citations:
  - path: "knowledge/README.md"
    symbol: "id連番ファイル名"
    note: "追跡に使うrun IDの規約"
---

ユーザーは、Issue名・番号など一時的な作業の識別子に頼るより、本質的に何を実装しているかが分かる命名を希望している。例えば学習設定は `train_dinov3_size_ablation.yaml` のように、対象と比較方法が伝わる名前にする。

Issue番号はPR本文・関連リンク・実験metadataなどの追跡情報に置く。[knowledgeのrun・group IDの命名](../../../knowledge/README.md#id連番ファイル名)に従い、既存のIDは実行時の証跡として保つ。

**使い方:** 名前だけで機能・方式・責務を説明できるかを確かめ、現行の呼び出し元・設定継承・利用例も同じ命名に揃える。
