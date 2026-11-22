---
title: Colabで複数条件を学習するときは、1セッションで逐次に実行し、VMの消失から再開できるようにする
summary: Colabの複数条件の学習は1セッションで順に実行し、VMを作り直しても保存済みの段階とcheckpointから再開できるようにする。
applies_to: all
source: 2026-10-30 のユーザー指示（Court Detectionのシーン組合せ比較の計画）
created: 2026-10-30
last_verified: 2026-11-20
evidence: 2026-10-30の人間の指示が逐次実行の根拠。Court実行runは既存8評価を保持し、P23を保存済み29,844更新から再開した経過を記録している。
citations:
- path: .agents/skills/tennis-colab/references/runtime.md
  symbol: '#障害からの再開'
- path: knowledge/nodes/court_detection/000056-run-court-scene-pair-execution-s42.md
  symbol: P23を29,844→33,160へ復元
  note: 完了済み段階を保持した運用例
---

Court Detectionのシーン組合せ比較では、複数条件をColab L4の同じセッションで一つずつ実行し、VMごとのdataset取得の繰り返しを避けることを望むと人間が2026-10-30に伝えた。別VMでの並列化を検討する場合は、入力取得の重複を費用に含める。

段階ごとの完了印と条件別のcheckpointをDriveに残す設計にすると、割当を作り直した後に完了済みの段階を飛ばせる。Courtの実行記録では、それまでの8評価を保持し、P23は同じ条件のcheckpointから再開した。
