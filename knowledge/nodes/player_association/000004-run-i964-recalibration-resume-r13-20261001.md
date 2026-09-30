---
id: run-i964-recalibration-resume-r13-20261001
type: run
task: player_association
sequence: 4
recorded_at: '2026-09-30'
title: '再較正特徴のtimeout回収と検証済みcamera単位再開'
provider: codex
issue: 964
date: '2026-09-30'
status: running
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i964-recalibration-resume-r13-20261001
parents: [run-i964-recalibration-r12-20260930]
---

run 12の特徴jobは外側GNU timeoutによるrc=124で停止した。
[回収記録](../../runs/run-i964-recalibration-resume-r13-20261001/collection.json)に
全cameraの時間・hash・元row検証・未完了一覧、元log/repro/guard/planを保存した。
9camera、10,269 frame、37,701検出rowはDINO/ViTPose/CLIPまで完了し、
全NPZのhash・provenance・timeline・box/score/元row一致・有限pose・単位外観をCPUで確認した。
10台目video_001/clip_000/cam0の検出は中断され、対応NPZは存在しない。
残る9camera/6,672 frameを再計算する。元の途中出力は変更しない。

GPU待ちのlock取得はtimeout起動より前で、待機時間は制限に含まれない。
7190秒制限に対しguardは7011.68秒でSIGTERMを受信し、差の約178秒は
先行する拡張build/起動に相当する（teardown差を含む概算）。資源上限による停止ではない。
workerの実時刻差6852秒とmonotonic経過には食い違いがある。
WSL/時計補正の可能性はあるが根因は未確認で、mtimeによるstage分解は概算と明示した。
旧logのcamera合計はmonotonic実測、DINOとpose+CLIPの区間はmtimeの代理値。
poseとCLIP個別の時間は保存されておらず復元できない。再開経路ではstage別monotonicを保存する。

peak GPU 4,548,722,688 bytes、最少available RAM 11,544,985,600 bytes、
旧出力約254MB。学習/fit/採点ではなく特徴抽出の失敗で、精度指標・学習曲線はない。
6clip/閾値/既定は維持し、18camera完了前に実データfitをしない。
次は完全なcameraだけの明示的再利用と合成入力のfitテスト、dev2cameraのCPU再追跡を準備する。

再開入口をa1303f02で実装し、合成再開/不一致停止/既存mask/監視の15 testsがpassした。
[明示再利用一覧](../../runs/run-i964-recalibration-resume-r13-20261001/reuse.json)と
[新plan](../../runs/run-i964-recalibration-resume-r13-20261001/resume-plan.json)をCPU検証した。
[再見積り](../../runs/run-i964-recalibration-resume-r13-20261001/estimate.json)は実測最遅cameraに
300秒のbuild/起動と2.2倍の夜間負荷余裕を加え約3.5時間。上限4.5時間、peak見積り6GB、
GPU停止9.5GB、追加出力見積り0.5GBで残り9cameraを1jobにする。

[登録前見積り](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5914000338)の後、
1件だけ `1790780776987996070_272352_i964-recalibration-features-resume-r13-20261001` をenqueueした（[receipt](../../runs/run-i964-recalibration-resume-r13-20261001/queue.json)）。
GPU完了は待たない。

[dev投影/CPU再追跡](../../runs/run-i964-recalibration-resume-r13-20261001/dev-retracking.json)は
clip_000/cam1の1行、video_001/clip_001/cam0の8行をproduction maskへ整合し、
既定StrongSORT++＋pose/CLIPで全長再生成した。元row/box一致、NPZ roundtrip、
GSI読戻し/synthetic非観測を確認。他10cameraは投影後値が同じで旧track/hashを再利用する。
元cache/trackは保持し、人物ラベル・新dev精度は見ていない。

CPU較正モジュールと全18cameraの完了gateを155b6c20で実装した。
[通常検証](../../runs/run-i964-recalibration-resume-r13-20261001/verification.json)と
[合成fit/gate 16 testsのlog](../../runs/run-i964-recalibration-resume-r13-20261001/fit-tests.log)を保存した。
実カメラモデルから合成した6clipを既存associateへ通し、foldの動画分離、全data最終fit1回、
元config不変、停止時の母数保持、候補の同点規則、支持不足/非収束/負slope/不安定の棄却を確認した。
再開/投影/監視の15 testsと合わせ31件（今回対象の重複なし）。独立validatorは指定なし、0回。
実データfit/新dev採点/予約未見は0回。無ラベルfitの結果や下流精度はまだ得ていない。

次runはqueueの全18cameraと旧再利用9cameraのprovenanceを回収・再検証し、
完了manifestをCPU fit入口へ渡す。支持不足/不安定/候補不合格は証拠を保存して採用保留とし、
devを開かない。採用できた場合のみYAML/fit証拠をcommit/pushした後、
投影後の同一trackへ旧/新尺度を適用するdev比較を1回行う。run11の旧値は別条件の参照とする。
全pipeline qualificationと凍結後の予約未見は未実施であり、PRはdraftを維持する。
