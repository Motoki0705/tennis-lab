---
title: 2026-06-21のPLCS整理では、指標記録済みでbackfill不能な旧ckptの手動削除が認められた
summary: 2026-06-21のPLCSローカル整理では、指標記録済みでbackfill不能な旧ckptの手動削除を人間が認めた。通常のpruneは検証済み予測を要求する。
applies_to: all
source: ユーザー判断（2026-06-21、PLCS ckpt 約111GBの整理）
created: 2026-06-21
last_verified: 2026-11-20
evidence: 2026-06-21の人間の判断記録。現行のprune_ckpts.pyは通常モードと--repro-dirの両方でpred_test.npzを検証し、knowledge READMEは削除記録と採用根拠の登録先を定める。
citations:
- path: .agents/skills/training-queue/scripts/prune_ckpts.py
  symbol: def verify_npz
  note: 通常のpruneが要求する予測検証
- path: knowledge/README.md
  symbol: '#削除した成果物の記録'
- path: knowledge/README.md
  symbol: '`deployments.yaml`'
  note: 採用checkpointの登録先
---

2026-06-21のPLCSローカル整理（約111GB）で、人間は空き容量の確保を優先した。保存configが当時のモデルコードと合わずtest推論をbackfillできない旧runでも、指標がknowledgeノードに残っていれば手動でckptを整理してよいと判断した。上書き済みの中間runやsmoke runも当時の整理対象だった。指標のない実験runは保全し、削除は `*.ckpt` に絞り、TensorBoard event・qualitative GIF・`hparams.yaml`・検証済み予測を残す条件だった。

この先例を参照するときは対象の採用状況と保存済み指標を照合する。通常の `prune_ckpts.py` とqueueの削除経路には検証済み `pred_test.npz` が必要。再現に関わる承認済みの削除は [knowledgeの削除記録](../../../knowledge/README.md#削除した成果物の記録) に従って記録する。
