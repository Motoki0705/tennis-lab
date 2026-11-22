---
title: 機能や方式が伝わる命名を優先し、Issue番号は追跡情報に使う
summary: 実装名から機能・方式が伝わるようにし、Issue番号は追跡情報に置く。knowledgeの既存run IDは保全する。
applies_to: all
source: '2026-10-13のユーザー指示（DINOv3サイズ比較configとPR分割の見直し）、#1087 / #1094 の指摘'
created: 2026-10-13
last_verified: 2026-11-20
evidence: '#1087 / #1094 と現行の knowledge/README.md のrun ID規約を照合する。'
citations:
- path: knowledge/README.md
  symbol: '#id連番ファイル名'
  note: 追跡に使うrun IDの規約
---

人間は機能・方式・責務が伝わる実装名を希望している。学習設定なら `train_dinov3_size_ablation.yaml` のように、対象と比較方法を名前で示す。呼び出し元・設定継承・利用例も名前に合わせる。

Issue番号はPR本文・関連リンク・実験メタデータなどの追跡に使う。knowledgeのrun/group IDは [ID規約](../../../knowledge/README.md#id連番ファイル名) に従う。新規run IDには実在するIssue番号を含められる。既存IDは参照と実行時の証跡を保つため維持する。
