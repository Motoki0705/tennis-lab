---
title: "test推論をbackfillできない古いckptは、結果がknowledgeノードにあれば削除してよい（ユーザー判断）"
summary: "backfillできない古いckptでも、指標がknowledgeにあれば整理できる。再現用の記録や検証済み予測は保全する。"
applies_to: all
source: "ユーザー判断（2026-06-21、PLCS ckpt 約111GBの整理）"
created: 2026-06-21
last_verified: 2026-11-20
evidence: ".agents/skills/training-queue/scripts/prune_ckpts.py の --repro-dir / --delete の条件（検証済み pred_test.npz）。knowledge/runs には予測配列を置かない（knowledge/README.md の保存基準）"
citations:
  - path: ".agents/skills/training-queue/scripts/prune_ckpts.py"
    symbol: "--repro-dir"
  - path: "knowledge/README.md"
    symbol: "#削除した成果物の記録"
---

ディスクを空けることを優先するユーザー判断である。ckpt は次の順で分類して消す。

1. test推論済みのrun（queueのrepro bundle `.training_queue/repro/<jobid>/predictions/`、またはckpt隣の `predictions/` に検証済みの `pred_test.npz` がある）: ckpt を削除する。予測配列はknowledgeへコピーされず、このローカルbundleにだけ残るので、bundleは消さない。
2. 最近のrunで、まだ推論していないがbackfillできるもの: `prune_ckpts.py --backfill --delete` を使う（モデルのコードがworktreeにしかなければ、そのworktreeから実行する）。
3. 古くてbackfillできない（保存したconfigが現行のモデルコードと合わない）が、指標がknowledgeノードに記録済みのもの、上書きされた中間runやsmoke run: そのまま削除してよい。

削除対象を `*.ckpt` に絞り、TensorBoard のイベント、qualitative の gif、`hparams.yaml` は残す。指標が無い場合は ckpt を保全する。再現に影響する削除は [knowledgeの保存基準](../../../knowledge/README.md#削除した成果物の記録) に従って記録する。

**使い方:** backfill は GPU を使うので、training queue が学習している間は実行しない。
