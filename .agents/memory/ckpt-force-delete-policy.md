---
title: "test推論をbackfillできない古いckptは、結果がknowledgeノードにあれば削除してよい（ユーザー判断）"
type: decision
applies_to: all
source: "ユーザー判断（2026-06-21、PLCS ckpt 約111GBの整理）"
created: 2026-06-21
last_verified: 2026-10-10
evidence: ".agents/skills/training-queue/scripts/prune_ckpts.py が存在する。knowledge/runs/<id>/pred_test.npz が test 推論済みの印"
---

ディスクを空けることを優先するユーザー判断である。ckpt は次の順で分類して消す。

1. `knowledge/runs/<id>/pred_test.npz` がある登録済みrun: ckpt を削除する。
2. 最近のrunで、まだ推論していないがbackfillできるもの: `prune_ckpts.py --backfill --delete` を使う（モデルのコードがworktreeにしかなければ、そのworktreeから実行する）。
3. 古くてbackfillできない（保存したconfigが現行のモデルコードと合わない）が、指標がknowledgeノードに記録済みのもの、上書きされた中間runやsmoke run: そのまま削除してよい。

消すのは `*.ckpt` だけにする。TensorBoard のイベント、qualitative の gif、`hparams.yaml` は残す。指標の記録がどこにもない ckpt は消さない。

**使い方:** backfill は GPU を使うので、training queue が学習している間は実行しない。
