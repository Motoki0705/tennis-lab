---
id: run-ball3d-foundation-review-20261103
type: run
task: tennis_scene
sequence: 33
recorded_at: '2026-11-03'
title: 'ball 3D基盤レビュー修正: court入力・同一性検証・LOVO coverage'
issue: 1556
provider: codex
date: '2026-11-03'
status: done
config:
  reviewed_head: 3abc5e7595bccabf4ed558ce209a9b4bc117a73c
  fixed_input_bundle_sha256: 7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a
  reference_sha256: dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09
  clip_id: meiji_3cam/video_000/clip_000
  tuning_frames: [0, 505]
  evaluation_frames: [505, 1010]
  court_measured_frames: [0]
  raw_line_available: false
  baseline_requires_court_lines: false
  scoring_binding: FixedInputBundle
  common_cohort: all_listed_methods_including_oracle
  gpu_used: false
  inference_rerun: false
metrics:
  pytest_passed: 96
  new_test_cases: 16
  collection_seconds: 7.42
  pytest_elapsed_seconds: 16.95
  max_case_seconds: 4.251
  html_broken_references: 0
  video_decoded_frames: 1010
  unchanged_prediction_runs: 8
  baseline_evaluation_lovo_coverage: 0.5642954856361149
  baseline_evaluation_all_view_valid_rate: 0.8316831683168316
  common_evaluation_reference_count: 768
  common_evaluation_reference_coverage: 0.5253077975376197
  baseline_common_evaluation_median_px: 8.971945791945725
  baseline_common_evaluation_p90_px: 25.403380157997688
  without_loess_common_evaluation_median_px: 9.696874001852741
  without_loess_common_evaluation_p90_px: 28.024329326423384
repro:
  commit: a635f5577507d8277e4307a8ac4b2936b3592b74
  command: bash knowledge/runs/run-ball3d-foundation-review-20261103/repro.sh BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON CHECKPOINT_ROOT PRIOR_COMPARISON NEW_OUTPUT_DIR
artifacts:
  run_dir: knowledge/runs/run-ball3d-foundation-review-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/foundation_v2
  reproduction: knowledge/runs/run-ball3d-foundation-review-20261103/repro.sh
  inventory: knowledge/runs/run-ball3d-foundation-review-20261103/artifacts.json
  inputs: knowledge/runs/run-ball3d-foundation-review-20261103/inputs.json
  qualification: knowledge/runs/run-ball3d-foundation-review-20261103/verification.json
  table: knowledge/runs/run-ball3d-foundation-review-20261103/metrics-table.png
  plot: knowledge/runs/run-ball3d-foundation-review-20261103/trajectories.png
  scores: knowledge/runs/run-ball3d-foundation-review-20261103/run-summary.json
  test_measurements: knowledge/runs/run-ball3d-foundation-review-20261103/test-results.json
  test_delta: knowledge/runs/run-ball3d-foundation-review-20261103/review-test-delta.json
  visual_review: knowledge/runs/run-ball3d-foundation-review-20261103/visual-review.json
parents:
- run-ball3d-evaluation-foundation-20261103
relations: []
papers: []
tags: []
---

## レビュー指摘への対応と入力の限界

PR #1669 の固定HEAD `3abc5e7595bccabf4ed558ce209a9b4bc117a73c` に対する親の明示レビューは
major 3件だった。前runの固定入力・baselineの数値・人間参照の採否を変えず、共通基盤の契約と表示を直す。
元storeの実行コードは引き続き `1f77a3bf8d637708d02f8f7ecd69549f93aa0438` である。

`pr-1669-1`: `ViewInput.court` にcamera別のraw KP14画素・score/valid、元のhomographyで採用したsubset、
テンプレートorientation、実観測frameを型付きで保持した。raw courtはmodelの検出であり人間提供の球の正解ではない。
このclipではframe0だけが実観測で、1010frameへbroadcastしたfitted14をraw観測と偽らない。
cam0/1/2の元homography採用点は5/7/8点で、残るraw点も保持する。sourceのcamera/frame/nativeサイズと不足を拒否し、
LOVOには選んだviewのcourtしか渡さない。

元の `CourtCalibrationOutput.court.diagnostics` にこれらのraw KPが既に保存されている。
したがって固定bundle v1は再生成せず、上記digestと元bytesを保持して使う。
元storeにはLINE座標・heatmapは存在せず、support/RMSの要約しかない。この欠落は親へ報告した。
LINEは型付きで渡せるが、実clipでは `lines=None` と欠落理由を明示し、要約から観測を捏造しない。
各方式はboolの `requires_court_lines` を必ず宣言する。trueならrunnerが選択viewの不足を検出し、
方式を呼ぶ前に例外で止まる。`court.require_lines()` / `load_bundle(..., require_court_lines=True)` でも停止する。
baseline/LOESSなしは固定cameraを使うのでfalseと宣言する。元LINEを使う #1567 の処理は別途の入力取得が必要で、
今回の実行はそのデータ取得やLINEによる校正・独立評価が完了したことを意味しない。推論再実行・GPU使用はない。

`pr-1669-2`: `score_comparison` と残差抽出の `error_samples` は検証済み `FixedInputBundle` を必須にした。
runのbundle digest・camera/frame軸と、人間参照のclip ID・camera・frame数・fps・nativeサイズ・動画hashを照合する。
同名camera・同じframe数の別入力も拒否する。oracle・plot・cropも同じbundleを受ける。
旧 `MethodInput` だけの採点入口は残さない。方式の入力へ正解を渡す変更ではない。

`pr-1669-3`: HTML主表とPNGに共通の表データを使い、LOVO誤差の隣にcount/label_count/coverageを出す。
全view有効率は別の見出しにし、方式間の採点集合の違いを表の近くに明記した。
全登録方式（oracleを含む）の (除外camera, 元frame) の共通採点集合でも比較する。
共通集合は誤差の大小ではなく可用性の共通部分で選び、欠測や空集合を0誤差で埋めない。
共通集合のidentity、観測/推定別の内訳と元の全対象へのcoverageを保存する。

## 検証の設計と保存

[前runの実装前設計への追記](../../runs/run-ball3d-evaluation-foundation-20261103/test-plan.md)に基づく。
camera/frameごとに異なるraw courtと選択subset、5/13/25pxの独立した手計算、825/1462=56.43%と
420/505=83.17%を区別する表fixtureを使う。実component codecを通したraw courtの保存読込も確認する。
96件のうち新規16件、前runの基盤28件、再利用のreader50件と指定2 nodeを実行した。
メモリの重い処理は逐次、pytestは `-n 0`、動画は1frameずつ処理する。

bundle digestの拒否guardを無効にする故障注入で、専用testが `DID NOT RAISE` を検出して失敗した。
元コードへ復元後に同nodeが成功し、全96件も元コードで成功している。
collectは96件/7.42秒、全caseは96 passed/16.95秒。最大caseは小動画integrationのsetup+call+teardown
4.251秒、最大setupは丸め表示0.01秒で、5秒/1秒の目安内。復元後の拒否node単独のpytest全体は
6.62秒（importを含む）で、caseの時間とは分けて残した。ruff/mypyも成功した。
追加16件のnodeと実測時間はtest delta、全nodeとstage時間はtest measurementsを参照する。
学習はなくTensorBoard曲線はない。

## 新しい比較成果物と実データ検証

出力先は `outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/foundation_v2/`。
`foundation_v1/`・固定入力v1・元store・人間正解は上書きしていない。コード版はrepro欄のcommit。
baseline/LOESSなしの各4foldを新規実行・保存・型付き再読込し、v1と位置/有効mask/使用view maskが8結果すべてbitwise一致した。
原storeへのbaseline照合も有効856frame・mask一致、3D最大差9.5367431640625e-7m（許容2µm）だった。

| 方式 | 全LOVO対象の中央値 / p90 px | 採点数 / 全対象 | 共通点の中央値 / p90 px | 共通採点数 / 全対象 |
|---|---:|---:|---:|---:|
| baseline | 9.340 / 27.078 | 825/1462 = 56.43% | 8.972 / 25.403 | 768/1462 = 52.53% |
| LOESSなし | 9.798 / 29.810 | 793/1462 = 54.24% | 9.697 / 28.024 | 768/1462 = 52.53% |
| 注釈oracle | 7.859 / 20.722 | 1345/1462 = 92.00% | 7.263 / 19.193 | 768/1462 = 52.53% |

上表は主評価frame [505,1010)。共通集合はoracleを含む全3方式で定め、全区間では1538点だった。
baselineの全view有効3D率83.17%をLOVO coverage56.43%の代わりには出さない。
新しいPNGとHTML主表の列・数値と注記を目視し、HTMLを別途parseして825/1462/56.43%/83.17%の列を照合した。
比較動画は再生成した1010frameを全decodeし、入力3動画も各1010frameを全decodeした。
代表0/252/505/757/1009、誤差上位6frameのcrop、軌道/分布/時系列図を目視し、HTML参照切れ0を確認した。
同じ予測で再生成したMP4のSHAはv1と同じだった。表示確認の範囲と各画像hashはvisual reviewに記録した。


## 比較の解釈

LOVOの各方式の対象数と共通点だけでの誤差は、欠測による選択を理解する補助である。
共通点の結果も可用性を条件とする比較で、欠測に隠れた誤差を回復したものではない。
独立3D正解はなく、oracleは同じ近似cameraによる幾何参照で、数学的下限を保証しない。
LINEの欠落と1frameだけのcourt観測は後続の校正方式の入力制約として残す。
標準pipelineの既定・方式の採否・checkpointの採用判断は変更しない。


## 出力checksumと再現

出力109ファイルのサイズ・SHAを再照合した。artifact inventoryのSHA-256は
`4de8e6156b4cfa3d01ddd6231f4ba58725bd0231881ed93ba5d9b8f44f9af761`。
HTMLは `2dc9697e405726365e5002fc1621eba5276c52ec2a3cef240b2177694b72b1e8`、
指標表PNGは `0f090bef6b324ec20aefce6155a18c280ecb8a05c76192ec425b4313281a9c62`、
全指標JSONは `cac01801c55fc399b22ce2900287490789260e62290104972dca52b2508d51ef`。
全指標JSONは297,583 bytesのためoutputsへ残し、knowledgeには区間集計とchecksumだけを保存した。
動画・予測配列・大きなcropもoutputsに置く。再現引数の5番目はv1の比較directory、6番目は新規出力先である。
実行script timerは420.4秒だった。同時に別のworkerが動く共有laptopであり、速度差のbenchmarkには使わない。
資源とGNU timeの原値は `resource-usage.txt` に保持する。
