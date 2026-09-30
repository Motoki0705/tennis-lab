---
id: run-i935-precision-variants-s42-r21-20260930
type: run
task: ball_refiner
sequence: 18
recorded_at: '2026-09-30'
title: 監視walkの消失競合を修正し事前固定3案を再試行
provider: codex
issue: 935
date: '2026-09-30'
status: planned
config: {seed: 42, detector: mixed-e9, context: detector_only}
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i935-precision-variants-s42-r21-20260930
  preflight: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/preflight.json
  retry_audit: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/retry_audit.json
parents: [run-i935-precision-variants-s42-r20-20260930]
relations: []
papers: []
tags: []
---

## 監視障害だけを直す再試行

run21 directiveが明示承認した1job retry。
[失敗run20](000017-run-i935-precision-variants-s42-r20-20260930.md#run21で回収した監視障害)の
事前宣言を維持し、`absolute_12k`・`anchored_3k`・`anchored_12k`を同じ順序で実行する。
監視の失敗はモデル精度を判定する根拠ではなく、比較結果は引き続き未測定。
CPU診断の仮説・実験因子は親ノードに集約する。ここで学習方式を選び直さない。

[修正版script](../../runs/run-i935-precision-variants-s42-r21-20260930/run_variants.py)は
`output_size`だけを変更した。`os.scandir`で子treeごとに走査し、
fileのstatとdirectoryのopen/iterationで生じる`FileNotFoundError`だけを許容する。
消えた部分を数えず、残った兄弟treeの走査を続ける。directory symlinkを辿らない既存仕様も維持する。
権限・I/O等の他エラーはwatchdogへ伝播し、従来どおりjobを止める。
`Path.rglob`は権限エラーを内部で握りつぶすため、catchを外へ広げるだけの修正にはしなかった。
容量は変更中のtreeのサンプル値であり、原子的snapshotやpoll間の瞬間上限の保証ではない。

元runのscript/plan/command/config、queue record、cache、途中出力を保持した。
retryは全training/evaluation/report/compiler cacheを新しい`r21`パスへ分離する。
失敗したstep250のstateを再開せず、事前宣言どおりscratch・seed42で開始する。
時間・VRAM・disk・RAM・CPUの上限、detector epoch9、split、selection規則、
評価のsample数とseed、BallGMM2D＋presence/NLL契約は同じ。person/pose/courtとtestは使わない。
候補残差平均は実験内parameterizationのままでdefaultにしない。

## 固定資料と通常検証

[plan](../../runs/run-i935-precision-variants-s42-r21-20260930/plan.json)、
[command](../../runs/run-i935-precision-variants-s42-r21-20260930/command.txt)、
[新旧hash・差分監査](../../runs/run-i935-precision-variants-s42-r21-20260930/retry_audit.json)が正本。
plan/command/scriptの旧hashを残し、修正版scriptと新出力先により変わった新hashを記録した。
全678入力のうち673はpath/hashとも同じ。3configと2scriptはretry bundleへ移し、
rendererはbyte一致、configの変更は`run.output_dir`だけ、実行scriptの変更は上記walkだけ。
出力先等を正規化すると新旧planが完全一致することを検査した。

[CPU preflight](../../runs/run-i935-precision-variants-s42-r21-20260930/preflight.json)は合格。
各案の4,910窓・source内訳・18選択clip・教師/split/gapのmanifestがr18と一致し、
310frameの旧checkpoint再現は従来と同じ許容差内。出力先は未生成でexistence checkを通る。
投入時と終了時にも全入力hashを照合する。CPU replayは旧モデルの再現検査であり、retry精度ではない。

[unit test](../../../tests/unit/tasks/ball_refiner/test_variant_watchdog.py)は`pytest -n 4`で7件成功。
実directoryを走査途中で削除する回帰テストは元run20実装で失敗し、修正版で成功した。
file消失、消えたtree以外の容量、symlink非再帰、open/iteration時のPermissionErrorとEIO伝播を検査。
3 Pythonファイルのruff/mypy成功。既存モデルや評価コードを変更していない。
GPU・3案の数値比較・独立test・文脈ablationは未検証。TensorBoardはこのrunnerでは生成しない。

## 投入予算と次の回収

元planの**25–45分・peak VRAM2–4 GB・新出力2 GB以内**という見積もりを維持する。
1job/resource=all、timeout3585秒＋15秒KILL、allocator6GiB、device-used7.5GBで停止、
出力4.5GBで停止、起動RAM8GiB以上・運転RAM6GiB以上、CPU2thread/compile2process/loader0。
grantは最大1時間・VRAM8GB・disk5GB。新しいqueue jobは1件だけとし、再失敗時の自動再投入はしない。
job IDと起動時状態は#935 run21の進捗コメントに記録し、終了を待たず引き継ぐ。

次runは先にqueue/log/reproとplanのreport内`resource_usage.json`を回収する。
失敗なら部分出力・原因・資源を保存する。成功なら全3案のcheckpoint選択・同一frame/PTS/教師・
比較表・NLL/HDR coverageと面積・実runtime/VRAM/diskを検査し、このノードを更新する。
