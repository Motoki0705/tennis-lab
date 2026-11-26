# PR #2461: 正式レビュー（run parent-1739-c64d9474845f48c388eab8bd88bffe72、round 1、HEAD 997d9ad0）の指摘への対応

## 採用1: 比較ページに未使用視点への再投影指標がない
#2459 は、未使用視点への再投影を clip 別の比較指標に指定している。保存済みの段階1ページの manifest を読むと、16本すべてで旧・新の両方が `null` だった。`visualization/pose_comparison/report.py:35` がこの欄に常に `None` を入れている。
- 未使用視点の観測と照合した値と、その分母を、比較ページに載せる。旧経路の停止などで採点対象がない場合は、理由を表示する。
- 集計の定義は #2470 の `multiview_scene report` の「未使用視点の残差」（採用IDに対応する実2D関節のうち、その3D関節のinlierに入らなかった視点）と揃える。定義が違うなら、理由をREADMEに書く。

## 採用2: 現行mainの SceneResult v3 を採用すると比較を再生成できない
現行mainの v3 export は `event_detection` を直接参照し、その依存先に `scene_assembly` を持つ。`pose_comparison/inputs.py:277` の `_read_scene` は、export が `scene_assembly` を直接指す v2 形式だけを受理する。保存済み v3 の clip_012 のindexを渡すと、正当な参照関係なのに `The adopted scene and its export disagree` で止まった。
- 既存の系譜検証を通して v2 と v3 の両方を読めるようにする。表示元の `scene_assembly` ID を記録する。保存形式の境界で検証するテストを付ける。

## 採用3: #2469 マージ後の互換表の組み直し（#2469 の修正が終わってから行う）
両PRは `source_identity.py`・`inventory.py`・FastHMR と FP32 の互換テスト2ファイルで競合する。比較側の互換表は `fcb97df3` 上の共通2ファイルを基準にしているため、登録と変換呼び出しを手で併合しただけでは完全一致判定を通らない。
- このターンでは、採用1・2 だけを行って push する。#2469 の修正版ができたら、親が改めて「#2469 の最終HEADをこのbranchに merge し、比較側の互換表を新しい基点と合流後の実ファイルに結び直す」作業を指示する。

## 不採用（参考）: イベントのキャッシュを再利用できない
main の #2610 の設計に由来する。対応しない。

## 共通
- main は #2610 のマージで進んでいる。`origin/main` を取り込む（rebase はしない）。関連テスト、pre-commit、knowledge 検査を通して push し、PR本文を更新する。
- 比較ページの再生成は、CPUで可能な範囲で、代表clip数本の保存済み成果物から行い、値が入ることを確かめる。公開用ページの全件再生成と Artifact 公開は、親が後で判断する。
- ベンチマークとデータセットには触れない。GPU は使わない。

新HEAD（完全SHA）、対処の要点、テスト結果を報告し、CI待ちで終了する。
