---
id: run-i935-precision-variants-s42-r21-20260930
type: run
task: ball_refiner
sequence: 18
recorded_at: '2026-09-30'
title: 候補残差平均12kは典型誤差とNLLを改善、観測裾の過信は残る
provider: codex
issue: 935
date: '2026-09-30'
status: done
config: {seed: 42, detector: mixed-e9, context: detector_only}
metrics: {meiji_observed_median_px: 3.108590188665046, meiji_observed_nll_px: 6.543255069030169, meiji_gap_nll_px: 9.976188463755792}
artifacts:
  run_dir: knowledge/runs/run-i935-precision-variants-s42-r21-20260930
  preflight: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/preflight.json
  retry_audit: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/retry_audit.json
  collection: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/collection.json
  comparison: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/comparison.md
  log: knowledge/runs/run-i935-precision-variants-s42-r21-20260930/queue.log
parents: [run-i935-precision-variants-s42-r20-20260930]
relations: []
papers: []
tags: []
---

## 監視障害だけを直す再試行

run21 directiveが明示承認した1job retry。
[失敗run20](000017-run-i935-precision-variants-s42-r20-20260930.md#run21で回収した監視障害)の
事前宣言を維持し、`absolute_12k`・`anchored_3k`・`anchored_12k`を同じ順序で実行する。
監視の失敗はモデル精度を判定する根拠ではない。run22で回収した比較結果は下記。
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
投入前にはGPU・3案の数値比較は未検証だった。独立test・文脈ablationは引き続き未検証。
TensorBoardはこのrunnerでは生成せず、各案のlearning_curve.jsonlを保存する。

## 投入予算と次の回収

元planの**25–45分・peak VRAM2–4 GB・新出力2 GB以内**という見積もりを維持する。
1job/resource=all、timeout3585秒＋15秒KILL、allocator6GiB、device-used7.5GBで停止、
出力4.5GBで停止、起動RAM8GiB以上・運転RAM6GiB以上、CPU2thread/compile2process/loader0。
grantは最大1時間・VRAM8GB・disk5GB。新しいqueue jobは1件だけとし、再失敗時の自動再投入はしない。
job IDと起動時状態は#935 run21の進捗コメントに記録し、終了を待たず引き継ぐ。

## run22の回収と数値の解釈

job `1790739175028391561_2386039_i935-precision-variants-s42-r21-20260930` はdone。
[回収検査](../../runs/run-i935-precision-variants-s42-r21-20260930/collection.json)で
108 checkpoint（48/12/48）のepoch/step・finite weight・manifest・選択NLLを検査し、
各curveの最小値（同率は早いepoch）とbest/hashが一致した。
選択epochはabsolute_12k=46、anchored_3k=10、anchored_12k=41（0始まり）。
全hashは[一覧](../../runs/run-i935-precision-variants-s42-r21-20260930/artifact_hashes.json)を参照。
420 val NPZとr18/e9の280参照NPZを照合し、70 clip / 40,144 frameのframe/PTS・
教師・存在mask・固定gapが一致、各案672群の集計を完全再現した。
HDR Monte Carlo自体は再実行せず、保存値と全reductionを照合した。
元動画のhashはstoreから継承し、評価用shardと入力ファイルを検査した。

[資源実測](../../runs/run-i935-precision-variants-s42-r21-20260930/resource_usage.json)は
1999.944秒（33分20秒）、device-used最大1,412,431,872 bytes（1.41 GB）、
監視1814回・failureなし。学習は842.192/322.936/715.941秒、
評価は28.089/30.251/34.856秒。終了時出力は合計1,291,119,267 bytes、
うちcompiler cache 322,209,032 bytes。poll間peakは保証しない。
allocatorの記録は評価phaseの値であり、学習全体のpeakとは扱わない。

数値の正本は[source/camera/half別全表](../../runs/run-i935-precision-variants-s42-r21-20260930/comparison.md)と
[全精度JSON](../../runs/run-i935-precision-variants-s42-r21-20260930/comparison.json)。
Meiji observed中央値はr18 23.66、detector 6.09、absolute_12k 10.52、
anchored_3k 4.20、anchored_12k 3.11 px。長期化だけでも改善するが、
候補平均を残す方式が典型精度の改善の主因と整合する。
ただしhead平均のゼロ初期化も同時に変わるため、parameterizationだけの因果効果とは分離できない。
anchored_12kはdetectorに対し各camera/halfおよびTrackNet/chatのp50/p90/p95を改善する。
一方r18に対してはcalibration/cam0とcam1のp95が悪化し、全層一様な優越ではない。
Meiji selection NLLで選んだcheckpointをそのまま使い、中央値では選び直していない。

Meiji全valのobserved/gap位置NLLは6.543/9.976。
HDR50/90/95はobserved 0.515/0.836/0.883、gap 0.521/0.891/0.923。
**calibration halfだけでは**observed 0.487/0.800/0.847、gap 0.470/0.841/0.880で、
選択側を含めた集計より過信が強い。HDR95面積は全valで15,195/53,833 px²。
2σ component楕円の描画と、混合分布のHDRを区別する。
detectorのgap coverage=1は全画面一様という自明な領域で、較正の成功ではない。

存在NLLはMeiji observed/gap 0.00635/0.00437だが確定負例0。
chatの明示的不在は269/66 frame、存在NLL 2.071/2.972で、存在較正完了とは言えない。
unknown・実遮蔽の推定座標をobserved教師へ混ぜていない。
単一seed・同一収録valの結果であり、独立test、真のamodal GT、3D効果は未測定。

## 計算だけのforwardとCPUのbit一致

forwardの`isinstance`をconstructorでのhead設定選択へ移し、allowlistは変更しない。
state_dict、config schema、tensor演算と順序は保持する。
[修正前snapshot](../../runs/run-i935-precision-variants-s42-r21-20260930/cpu-before.json)と
[bit一致検査](../../runs/run-i935-precision-variants-s42-r21-20260930/cpu-identity.json)は、
r18とanchored_12kそれぞれ5 val clip・先頭/中央/末尾33frame・observed/固定gapの
計60窓（1,980 model-frame、480 tensor）のraw/means/scale/logits/covariance/weights/presenceを
CPU float32で比較し、bytes完全一致・最大差0。入力・checkpoint・修正前後sourceのhashを保存した。
CUDAでのbit一致や再学習結果の一致はこの検査の主張に含めない。
候補残差headの設計採用とpipeline default切替はユーザー判断待ちで、実験用のまま。
