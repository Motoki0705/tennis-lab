---
id: run-i1565-segment-physics-offset-20261103
type: run
task: tennis_scene
sequence: 39
recorded_at: '2026-11-03'
title: clip_000の区間物理軌道・二次抵抗・camera相対時刻の同時推定
issue: 1565
provider: codex
date: '2026-11-03'
status: done
config:
  clip_id: meiji_3cam/video_000/clip_000
  num_frames: 1010
  tuning_frames:
  - 0
  - 505
  evaluation_frames:
  - 505
  - 1010
  observations:
  - candidates
  - heatmap
  label_tuning: false
  gpu_used: false
  inference_rerun: false
metrics:
  collected_tests: 99
  passed_tests: 99
  new_test_cases: 13
  pytest_seconds: 61.43
  collection_seconds: 8.64
  baseline_evaluation_lovo_median_px: 9.340188442451916
  baseline_evaluation_lovo_p90_px: 27.078044438775773
  candidate_evaluation_lovo_median_px: 13.636899688402742
  candidate_evaluation_lovo_p90_px: 58.19808932817398
  baseline_evaluation_lovo_count: 825
  candidate_evaluation_lovo_count: 756
  heatmap_failed_folds: 4
  all_method_common_cohort_count: 0
  paired_evaluation_common_count: 756
  output_files_verified: 211
  ci_repair_collected: 495
  ci_repair_passed: 494
  ci_repair_skipped: 1
  ci_repair_pytest_seconds: 279.81
  ci_repair_collection_seconds: 69.16
  ci_replay_prediction_fields_equal: true
  ci_replay_accuracy_fields_equal: true
repro:
  commit: 345ae65f7beb7ceb593e9b16a5e426eb50f65bc1
  command: bash knowledge/runs/run-i1565-segment-physics-offset-20261103/reproduce.sh
    BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON PRIOR_COMPARISON_DIR NEW_OUTPUT_DIR
  prediction_commit: 3ddbe0f458280ce1c3aafcce2faced805d7e2486
artifacts:
  run_dir: knowledge/runs/run-i1565-segment-physics-offset-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/segment_physics_offset_v3
  tests: knowledge/runs/run-i1565-segment-physics-offset-20261103/test-results.json
  test_summary: knowledge/runs/run-i1565-segment-physics-offset-20261103/test-summary.json
  metrics: knowledge/runs/run-i1565-segment-physics-offset-20261103/metrics-summary.json
  inventory: knowledge/runs/run-i1565-segment-physics-offset-20261103/artifacts.json
  verification: knowledge/runs/run-i1565-segment-physics-offset-20261103/artifact-verification.json
  visual_review: knowledge/runs/run-i1565-segment-physics-offset-20261103/visual-review.json
  execution_history: knowledge/runs/run-i1565-segment-physics-offset-20261103/execution-history.json
  ci_test_summary: knowledge/runs/run-i1565-segment-physics-offset-20261103/ci-test-summary.json
  ci_order_probe: knowledge/runs/run-i1565-segment-physics-offset-20261103/ci-order-after.json
  ci_replay_check: knowledge/runs/run-i1565-segment-physics-offset-20261103/ci-replay-comparison.json
  ci_checks_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/segment_physics_offset_ci_checks_v1
parents:
- run-ball3d-foundation-review-20261103
relations: []
papers: []
tags: []
---



## 主結果: 悪化と未収束

後半LOVOの候補点方式は中央値13.637/p90 58.198pxで、baseline 9.340/27.078pxより悪化した。
>20px率は16.36%→35.32%、採点数825→756、coverage56.43%→51.71%（−4.72pp）。
同じ756点でもbaseline 9.067/26.253→13.637/58.198pxへ悪化し、欠測の選別だけでは説明できない。
heatmap方式は全4foldで100評価の上限に達し未収束。有効3D・採点数は0、誤差と不確かさは利用不可。
後半を見て上限・閾値・方式を調整せず、失敗と悪化をそのまま保存した。採用を勧める結果ではない。

## 条件と方式

親 #1554 revision 6、子 #1565。固定bundleの3camera・1010frameをCPUでreplayする。
既存loader、全view/LOVO runner、型付き保存、#1668 readerによる採点、比較動画・図・HTMLを再利用する。
方式の入力には正解・その欠測mask・除外cameraを渡さない。すべての予測を保存してから正解を読む。
候補点・denseの設定は合成検証後に上記commitで固定し、ラベルによる調整は前半を含め行わない。
主区間は後半[505,1010)、前半と全区間は併記する。

標準baseline 3Dの左右の2次fitから速度変化を求め、8m/s以上の局所極大をイベント候補とする。
低い位置で鉛直速度が下向きから上向きになればbounce、他はhit候補とする。長い欠測で分割し、
100frameを超える区間も数値積分のため分割する。後者を物理イベントと数えない。
各区間の位置・速度6変数、clip共通の二次抵抗係数、camera相対時刻を頑健な画像残差で同時fitする。
gravityはcourt Z下向き9.81m/s²、RK4、sparse least squaresとsoft-L1を使う。
候補点方式は距離²/sigma²−2log(score)が最小の有効候補を選ぶ。
dense方式はnative mapのbilinear尤度の−2log平方根を使う。mapを複製・拡大しない。
実装の設定・境界・保存契約の正本は [ball_trajectory README](../../../src/tennis_scene/ball_trajectory/README.ja.md)。

保持cameraの辞書順先頭をoffset=0に固定し、他を絶対値0.99frame以内に制限する。
全viewと各LOVOで独立に初期化・分割・fitする。LOVOの除外cameraは基盤の規約で元校正・offset0のまま。
短い欠測を内部fitが跨いでも出力はbaselineの有効/使用view maskを超えず、欠測を観測へ昇格しない。
未知のbiasを埋めるbaseline fallbackはない。最適化未収束は理由と無効出力を保存する。

## 先行診断との関係

[PR #1670](https://github.com/Motoki0705/tennis-lab/pull/1670) の時刻oracleは前半のラベルを使う採点専用診断で、
推定offsetを本方式へ渡さない。推定完了後の比較だけを保存する。物理的な同期真値ともみなさない。
LOESS除去の悪化を踏まえ、標準LOESSのbaselineを初期化に使う。
[PR #1672](https://github.com/Motoki0705/tennis-lab/pull/1672) の多仮説時間探索はcoverageも低下したため、
本方式ではbaselineの支持を上限とし、誤差と採点数・coverage・共通cohortを同時に示す。
[PR #1673](https://github.com/Motoki0705/tennis-lab/pull/1673) で全view改善とLOVO悪化が両立したことから、
全viewの自己再投影や滑らかさだけで成功とは判定しない。

## テスト設計と実測

実装前に [test-plan.md](../../runs/run-i1565-segment-physics-offset-20261103/test-plan.md) を作成した。
独立DOP853と非対称pinhole式で、drag0.021/m、符号が異なるoffset、frame31のbounceを生成した。
候補点方式はoffset0.27/−0.46/0frameを0.008frame許容、dragを0.0002/m許容、
反発率0.73を0.002許容で復元し、camera順序の入替とdenseのmap尤度・欠測保持も検証した。
main由来の固定配列と既定componentの3D・mask・使用view・2D入力・棄却理由が厳密一致する。
元mainの計算3ファイルのbytesを確認するfixture生成入口は `capture_default.py`。
実設定→definition→assembler→component→codecと基盤adapter/LOVOを実solverで照合した。

新規13case（unit11、integration2）、既存86caseを再利用し、99passed/61.43秒、collect99件/8.64秒。
wrapper時計は実行65.74秒・収集14.30秒であり、pytest表示と区別する。
最大case31.39秒は小さい76frameで6回の実fitとcodec・LOVO隔離を検証する接続integration。
mock化で失う数値契約と5秒超過の理由は設計に記録した。最大setupは0.211秒。
時刻の符号反転を注入すると既知offsetの数値assertが失敗し、復元後に成功した。
全node・各phase時間、収集、故障注入の原値はrun内のJSON/TXTに保存している。


## clip_000: 全view・LOVO・区間別

値は median / p90 px / >20px率 ; 採点数/対象数 (coverage)。

| 区間 | 方式 | LOVO pool | cam0除外 | cam1除外 | cam2除外 |
|---|---|---:|---:|---:|---:|
| evaluation | pipeline_baseline | 9.340/27.078/16.36%; 825/1462 (56.43%) | 7.920/14.235/4.06%; 271/473 (57.29%) | 11.990/25.147/22.94%; 231/484 (47.73%) | 9.262/40.718/21.98%; 323/505 (63.96%) |
| evaluation | segment_physics_candidates | 13.637/58.198/35.32%; 756/1462 (51.71%) | 11.853/56.791/30.16%; 252/473 (53.28%) | 17.427/77.452/33.80%; 213/484 (44.01%) | 13.769/52.601/40.89%; 291/505 (57.62%) |
| evaluation | segment_physics_heatmap | unavailable; 0/1462 (0%) | unavailable; 0/473 (0%) | unavailable; 0/484 (0%) | unavailable; 0/505 (0%) |
| evaluation | annotation_oracle | 7.859/20.722/11.15%; 1345/1462 (92.00%) | 6.258/10.443/0.22%; 447/473 (94.50%) | 10.139/19.360/8.35%; 443/484 (91.53%) | 7.887/41.457/24.62%; 455/505 (90.10%) |
| tuning | pipeline_baseline | 10.716/28.483/16.44%; 821/1497 (54.84%) | 10.498/18.153/8.78%; 262/497 (52.72%) | 14.323/40.693/36.02%; 236/501 (47.11%) | 9.657/19.111/8.36%; 323/499 (64.73%) |
| tuning | segment_physics_candidates | 19.357/202.577/49.73%; 750/1497 (50.10%) | 12.715/29.086/22.76%; 246/497 (49.50%) | 34.508/569.929/76.12%; 201/501 (40.12%) | 27.692/227.192/54.13%; 303/499 (60.72%) |
| tuning | segment_physics_heatmap | unavailable; 0/1497 (0%) | unavailable; 0/497 (0%) | unavailable; 0/501 (0%) | unavailable; 0/499 (0%) |
| tuning | annotation_oracle | 9.655/22.524/11.46%; 1422/1497 (94.99%) | 7.722/11.938/0.43%; 460/497 (92.56%) | 11.394/24.631/13.47%; 475/501 (94.81%) | 9.960/30.659/19.92%; 487/499 (97.60%) |
| all | pipeline_baseline | 10.208/28.196/16.40%; 1646/2959 (55.63%) | 9.708/16.645/6.38%; 533/970 (54.95%) | 13.081/35.881/29.55%; 467/985 (47.41%) | 9.524/29.923/15.17%; 646/1004 (64.34%) |
| all | segment_physics_candidates | 16.225/125.506/42.50%; 1506/2959 (50.90%) | 12.252/40.995/26.51%; 498/970 (51.34%) | 21.335/289.087/54.35%; 414/985 (42.03%) | 17.531/167.949/47.64%; 594/1004 (59.16%) |
| all | segment_physics_heatmap | unavailable; 0/2959 (0%) | unavailable; 0/970 (0%) | unavailable; 0/985 (0%) | unavailable; 0/1004 (0%) |
| all | annotation_oracle | 8.790/21.392/11.31%; 2767/2959 (93.51%) | 7.076/11.072/0.33%; 907/970 (93.51%) | 11.049/20.341/11.00%; 918/985 (93.20%) | 8.698/38.342/22.19%; 942/1004 (93.82%) |

全方式共通LOVO cohortは、失敗したheatmapを含むため3区間とも空集合。誤差0とは扱わない。
追加でbaselineと候補点方式だけを同じ基盤で採点し、以下の共通点比較を保存した。

| 区間 | 共通点数/対象 | baseline median/p90/>20px | 候補点方式 median/p90/>20px |
|---|---:|---:|---:|
| evaluation | 756/1462 | 9.067/26.253/15.61% | 13.637/58.198/35.32% |
| tuning | 750/1497 | 10.729/24.611/13.73% | 19.357/202.577/49.73% |
| all | 1506/2959 | 10.041/25.328/14.67% | 16.225/125.506/42.50% |

| 区間 | 方式 | 全view median/p90/>20px | 全view採点/対象 | 有効3D率 |
|---|---|---:|---:|---:|
| evaluation | pipeline_baseline | 4.625/11.581/3.47% | 1238/1462 | 83.17% |
| evaluation | segment_physics_candidates | 8.339/102.609/30.62% | 1179/1462 | 80.79% |
| evaluation | segment_physics_heatmap | unavailable | 0/1462 | 0.00% |
| tuning | pipeline_baseline | 5.058/12.984/1.46% | 1299/1497 | 86.34% |
| tuning | segment_physics_candidates | 8.112/118.596/30.64% | 1263/1497 | 86.14% |
| tuning | segment_physics_heatmap | unavailable | 0/1497 | 0.00% |
| all | pipeline_baseline | 4.873/12.283/2.44% | 2537/2959 | 84.75% |
| all | segment_physics_candidates | 8.278/108.822/30.63% | 2442/2959 | 83.47% |
| all | segment_physics_heatmap | unavailable | 0/2959 | 0.00% |

全fold・参照kind別・元イベント窓・粗大誤差・隣接移動・加速度・時間はoutputsの `scores.json`。
予測の空白を0誤差として埋めず、人間提供の観測/推定の出自と除外範囲を基盤のまま保持した。

## 時刻ずれと物理診断

候補点方式の値。±は局所標準誤差、anchorの0は固定した時刻基準である。

| fold | 基準camera | 他cameraのoffset ± SE (frame) | drag /m | 区間/イベント候補 |
|---|---|---|---:|---:|
| all_views | cam0 | cam1: -0.922 ± 0.244; cam2: -0.282 ± 0.297 | 0.03028 | 39/26 |
| without_cam0 | cam1 | cam2: 0.402 ± 0.317 | 0.10000 | 22/0 |
| without_cam1 | cam0 | cam2: -0.151 ± 0.394 | 0.06438 | 16/0 |
| without_cam2 | cam0 | cam1: -0.989 ± 0.301 | 0.03020 | 17/0 |

全viewの26件はすべてhit候補で、bounce候補は0。LOVOは全foldでイベント候補0となり、
分割の観測支持不足・初期値依存が残る。cam0除外のdragは上限0.1/mに達し、cam2除外のcam1 offsetは
−0.989frameと境界付近。全viewのcam1/cam2 offset −0.922/−0.282frameは、#1558のラベル由来診断
−0.249/−0.775frameとも一致しない。後者を同期真値とはせず、方式入力へは渡していない。
ODE残差が小さいのは制約した関数族の結果で、画像精度の改善ではない。
分割・イベント・物理残差・offsetと不確かさは `physics-diagnostics.png`、数値は `diagnostics.json`。
未収束heatmapのoffsetは最終失敗iterateとしてだけ表示し、成功した推定値と混同しない。

## 適用限界と次の検証

単一clip・固定の近似校正・推定値を含む2D正解であり、独立した3D真値はない。
相対時刻の基準はfoldごとに異なり得る。drag・時刻・校正bias・分割誤りの交絡は識別できない。
局所曲率の標準誤差は分割・候補選択を条件とした近似で、較正済み信頼区間ではない。
rank不足は明示的に利用不可、境界到達はflagを保存する。
区間間の位置連続性とbounce法則は強制しないため、位置差・鉛直反発率も診断へ残す。
event検出は初期値に依存し、Magnus・rolling shutter・校正変更は未実装。
採用・既定化・mergeは統合比較後の人間の判断とし、このrunでは行わない。
学習を行わないためTensorBoard曲線は対象外である。

## 成果物と再現の照合

最終版は `segment_physics_offset_v3/`。v1は固定設定での16結果、採点、1010frame動画を作成した原記録。
v2は後処理のcommit前起動により最終出自として採用しなかった表示draftで、上書きせず保持した。
v3はcommit済み後処理からv1の全207ファイルをhash照合し、型付き16結果を再読込・再採点して
元の全採点fieldの一致を確認した。baseline/候補点の共通cohort図表と失敗表示だけを追加した。
全211ファイルをサイズ・SHA-256で再照合し、188個の予測関連ファイル・動画・元scoresはv1と同じbytes。
v3作成時点（後処理head 345ae65f7）のsrc差分は予測headに対して0、既定baselineの4foldも `foundation_v2` と配列単位で一致した。

動画は1200×1128・59.94006fps・1010frameをv1で全decodeしたbytesを再利用した。
代表0/252/505/757/1009、粗大誤差109/108/107/105/104/103の3camera crop、指標図と物理診断を目視した。
unknownとcam0末尾外挿に参照を描かず、観測/承認推定/予測の記号を区別した。HTML参照切れ0。

v1 script timer541.0秒、GNU time wall515.31秒、最大RSS1,117,368KiB・swap0。
時計の原値が一致しないため統合しない。v3は再fit・再encodeなしの検証と表示追加である。
再現は本runの `reproduce.sh`（5引数）を新規出力先で実行する。
checksum・全test node・実測時間・目視・資源・履歴の原値をrun bundleへ保存した。
予測配列、動画、大きなHTMLはoutputsに残し、Gitにはコピーしていない。

実装・テスト・実データ比較と可視化は完成した。heatmapの未収束、候補点方式の悪化は未解決の研究上の制限。
残る手続きは親の統合レビュー、人間の採否判断であり、mergeは実行しない。

## CI契約違反・camera順序依存の修正

初回HEAD e067580d6のCI run 41622391931は、設定監査の収集とcamera順序テストで失敗した。
初回の99件は設定監査を含まず、新規Configの契約違反を見逃していた。

`SegmentPhysicsConfig` の15fieldからPython既定値を取り除き、compositionの値を必須引数として受ける。
評価adapterのsettings/baselineの暗黙生成と、積分helperのgravity/steps既定値も取り除いた。
production YAMLの値は一切変更せず、replayとfixtureもHydra compositionから明示的に渡す。
既存configuration監査389件を含め、495件収集・494passed・1skip、pytest279.81秒、collect69.16秒。
skipは既存visualizationの空parametrizeによるNOTSETで、成功件数には含めない。
最大75.09秒は既存のcold import順序検証、最大setup0.099秒。新方式integrationは15.15秒で、
実solverと全foldを通す既存の理由により残す。全node/phaseは重複node IDをまとめたJSONへ損失なく保存した。
元の大きなstdout/JSONはci_checks directoryへ保存し、runの容量制限を守る。

順序依存は同一初期3Dで初期値・boundsが完全一致し、残差/Jの行だけが並び替わる場合にも発生した。
修正前は両方11評価・ftol終了ながら、局所optimality0.01404/0.01517、offset差8.684e-6秒
（CIでは1.7e-5秒）だった。LSMRの丸めと相対cost終了の経路差であり、anchorの取り違えではなかった。
solverの残差行・自由clock列をcamera ID順へ固定し、offset・使用view・共分散・診断を呼出元の軸へ戻す。
終了閾値・許容誤差は広げていない。むしろ既存testを非対称maskと、offset/位置/共分散の厳密一致へ強化した。
修正後のprobeは両順序の初期値・残差行・疎行列・cost・評価回数が同一で、offset差0秒、位置/共分散もbitwise一致。
`ci_inspect_order.py --revision <commit> --output <new.json>` で旧/新両方を再現できる。

修正source head 345edc98dで、固定bundleからbaseline/候補点/heatmapの全12実行をCPU再実行した。
全viewと全LOVOは元からcamera IDが辞書順で、composed設定もv3と完全一致する。
3D/mask/offset/共分散に加え、イベント・残差・収束診断を含むBall3DResultの全fieldがv3とbitwise一致した。
既存の採点専用oracle4結果を束縛し直し、実行時間2fieldを除く採点全項目も一致した。
正規化scoresのdigestは旧・新とも `ci-replay-comparison.json` の同じ値である。

v3の全211ファイルを事前照合し、v3は上書きしていない。予測・誤差・coverageは不変のためv4は作らない。
既知の候補点方式の悪化とheatmap未収束も不変である。照合は新規 `segment_physics_offset_ci_checks_v1/replay/`。
元v3 manifestと照合receiptのchecksumは `ci-proof.json`、再現scriptは `ci_replay_check.py`。
GNU time wall237.48秒、最大RSS649,596KiB、swap0。動画再生成・GPU・検出器再推論は行っていない。
再pushのCI run 41641759130では設定収集とcamera順序を含むCPU 9レーンが成功したが、
レーン9の既存DINO importテストが失敗した。偽の親packageを隔離しても、実DINOのlossテスト等が
読み込んだ子moduleのMSDA参照が残るため、productionの異なるbinary拒否契約に到達した。
子module未読込/読込済みの2ケースへ既存unit testを拡張し、後者だけが同じ例外で失敗することを再現した。
子moduleもmonkeypatchで隔離・復元する修正後、選択binaryを使い、stale参照を明示的に拒否する両方の検証が成功した。
追加1ケースは実モデル・GPUを使わず、DINO関連33件はcollect5.21秒・実行5.62秒、33passed。
全node/時間と修正前の失敗は `ci-dino-{collection,durations}.txt`、`ci-dino-isolation-before.txt`、
照合は `ci-dino-isolation.json` に保存した。ruff/mypyも成功した。
追加差分はtestと記録だけで、345edc98dからsrc差分0を確認したため、上記12実行の予測不変の照合を引き続き適用する。
PRの最新HEADで、push後の全CI jobとPython集計の成功を確認する。
