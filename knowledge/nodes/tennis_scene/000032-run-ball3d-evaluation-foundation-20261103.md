---
id: run-ball3d-evaluation-foundation-20261103
type: run
task: tennis_scene
sequence: 32
recorded_at: '2026-11-03'
title: clip_000の固定入力・人間参照によるball 3D評価と比較基盤
issue: 1556
provider: codex
date: '2026-11-03'
status: done
config:
  clip_id: meiji_3cam/video_000/clip_000
  cameras: [cam0, cam1, cam2]
  num_frames: 1010
  tuning_frames: [0, 505]
  evaluation_frames: [505, 1010]
  reference_observed: 2589
  reference_estimated: 370
  reference_excluded_extrapolation: 28
  reference_missing: 43
  gpu_used: false
  inference_rerun: false
  source_store_commit: 1f77a3bf8d637708d02f8f7ecd69549f93aa0438
  scoring_threshold_px: 20
  event_radius_frames: 3
metrics:
  baseline_evaluation_lovo_median_px: 9.340188442451916
  baseline_evaluation_lovo_p90_px: 27.078044438775773
  baseline_evaluation_lovo_above20_rate: 0.16363636363636364
  baseline_evaluation_lovo_reference_coverage: 0.5642954856361149
  baseline_evaluation_valid3d_rate: 0.8316831683168316
  without_loess_evaluation_lovo_median_px: 9.797591870007757
  without_loess_evaluation_lovo_p90_px: 29.81043880847947
  without_loess_evaluation_lovo_above20_rate: 0.17906683480453972
  without_loess_evaluation_lovo_reference_coverage: 0.542407660738714
  without_loess_evaluation_valid3d_rate: 0.7900990099009901
  oracle_evaluation_lovo_median_px: 7.8586408238936745
  oracle_evaluation_lovo_p90_px: 20.721912427346656
  baseline_valid_frames_all: 856
  max_baseline_position_difference_m: 0.00000095367431640625
  test_passed: 80
  pytest_elapsed_seconds: 12.92
  comparison_decoded_frames: 1010
  html_broken_references: 0
repro:
  commit: ca63330c31fcf48363037e2cb0f681c0f7284743
  command: bash knowledge/runs/run-ball3d-evaluation-foundation-20261103/repro.sh BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON CHECKPOINT_ROOT NEW_OUTPUT_DIR
artifacts:
  run_dir: knowledge/runs/run-ball3d-evaluation-foundation-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/foundation_v1
  reproduction: knowledge/runs/run-ball3d-evaluation-foundation-20261103/repro.sh
  executed_reproduction: knowledge/runs/run-ball3d-evaluation-foundation-20261103/executed-repro.sh
  inputs: knowledge/runs/run-ball3d-evaluation-foundation-20261103/inputs.json
  inventory: knowledge/runs/run-ball3d-evaluation-foundation-20261103/artifacts.json
  qualification: knowledge/runs/run-ball3d-evaluation-foundation-20261103/verification.json
  measurements: knowledge/runs/run-ball3d-evaluation-foundation-20261103/test-results.json
  scores: knowledge/runs/run-ball3d-evaluation-foundation-20261103/run-summary.json
  visual_review: knowledge/runs/run-ball3d-evaluation-foundation-20261103/visual-review.json
  table: knowledge/runs/run-ball3d-evaluation-foundation-20261103/metrics-table.png
  plot: knowledge/runs/run-ball3d-evaluation-foundation-20261103/trajectories.png
parents:
- run-scene-ball-loess-20261026
relations: []
papers: []
tags: []
---

## 比較条件と観測

親 #1554 の合意 revision 6、子 #1556 に従い、保存済みの3camera・1010frameをCPUだけでreplayした。
元storeの実行コードは `1f77a3bf8d637708d02f8f7ecd69549f93aa0438`、比較コードはrepro欄のcommitである。
元storeを現行mainが生成したとは扱わない。再開前の未コミット27ファイルは、保全snapshotの全SHA一致を
確認し `924803091` に保存した。正解reader #1668 は `5ee7e5e12` でmergeし、GPT専用の独自loaderを削除した。

固定bundleのSHA-256は `7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a`、
人間提供の統合JSONは `dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09`。
採点可能2959点のうち直接観測2589、承認された推定370を別kindで保持した。cam0 frame982〜1009の
外挿28点と、座標のない遮蔽26・不明17点は補完も採点もしない。readerが元6CSVと入力identityを検証する。
注釈と保存済みbaseline参照3Dは `MethodInput` へ渡さない。固定校正は元の全cameraを使って作られたものを
各方式・LOVOで共有しており、camera推定段階まで含む完全な交差検証ではない。

baselineは元のdecode/gate観測からconsensus・LOESS・三角測量を再実行する。
「LOESSなし」は2Dの採用maskを共通にして元の採用検出座標を三角測量し、座標平滑化だけの差を調べる。
この評価adapterはmaskを再現するための共通処理としてLOESSも計算後に破棄する。
計算時間はこのadapterの実際のCPU時間で、最適化したproduction方式の速度差を表さない。

## 主評価（frame 505〜1009）

| 方式 | LOVO中央値 / p90 (px) | >20px率 | LOVO採点数 / 対象数 | 全view有効3D率 |
|---|---:|---:|---:|---:|
| baseline | 9.340 / 27.078 | 16.36% | 825 / 1462 (56.43%) | 83.17% |
| LOESSなし | 9.798 / 29.810 | 17.91% | 793 / 1462 (54.24%) | 79.01% |
| 注釈oracle | 7.859 / 20.722 | 11.15% | 1345 / 1462 (92.00%) | 98.81% |

全採点点ではbaselineの誤差が小さいが、復元の有効frameと採点対象集合も異なるため、同じ点での
対になった精度改善を示した値ではない。欠測を0誤差として埋めず、coverageを同時に読む必要がある。
直接観測だけのLOVO中央値はbaseline 10.197px、LOESSなし10.316px、推定だけは7.840 / 8.100px。
推定点を観測へ混ぜるとこの違いが消えるため、全fold・3区間で内訳を残した。

全viewの主評価再投影中央値は4.625 / 5.321px、粗大再投影frame率は9.76 / 10.03%。
後者は採点可能frameのどれか1cameraで20pxを超えた割合で、独立3D正解への粗大誤差率ではない。
隣接移動量・加速度の診断は欠測・splitを跨がず、バウンド・打球の元25行はcamera・注記ごとに保存する。
同じ事象の1frame差を保持し、窓の和集合で各残差を1度だけ採点する。物理イベント数は推定しない。
cam2の推定区間16件も保持した。

oracleは同じ近似cameraで保持viewの正解2Dだけを三角測量する採点用の幾何参照である。
校正の誤差や参照のnoiseがあるので、理論的な下限の保証や独立した3D精度は主張しない。
先行run `run-scene-ball-loess-20261026` の検出点への自己再投影と、今回の人間参照への誤差は評価契約が違う。

## 検証と再現

再現には同じローカルのbundle・3動画・正解JSONと隣のsource CSV・checkpointを与え、新しい出力先を指定する。
上のrepro commandの引数はこの順序で、動画・重み・出力を全メモリへ載せない。
実行時のwrapperは `ca63330c3` に固定し、同一bytesを `executed-repro.sh` に保全した。
現行 `repro.sh` は静的再現path検査が解釈する `SCRIPT_DIR` と `PYTHONPATH=.` へ変数表記を直した。
呼ぶPythonと引数・thread設定は同じであり、実行時の記録を新wrapperの実行結果とは扱わない。
入力3動画はそれぞれ1010frame、比較MP4は1200×1128・1010frameを逐次decodeした。HTML参照切れ0。
代表frame 0・252・505・757・1009、指標表と軌道/分布/時系列図、誤差上位765・581・580・980・448・447の
3camera cropを目視確認した。[目視記録](../../runs/run-ball3d-evaluation-foundation-20261103/visual-review.json)は
表示・glyph・欠測の保持の確認であり、全frameの真の座標を保証しない。

[verification](../../runs/run-ball3d-evaluation-foundation-20261103/verification.json)で、baselineの有効856frame・
有効mask・使用view mask、LOESS後の座標/観測mask/confidenceの完全一致を確認した。decodeのmask/confidenceは
完全一致するが復号座標には最大0.0001220703125pxの丸め差がある。引継ぎ文の「decode完全一致」は
座標のbitwise一致ではなく、旧 `baseline_verification_v1.json` にも同じ差が記録されていた。
3D最大差は9.5367431640625e-7mで、2µmの許容差内、bitwise一致ではない。確認したproductionの
ball reconstruction・LOESS・triangulation・decode/gateのコードは元store版から差分0だった。

[テスト測定](../../runs/run-ball3d-evaluation-foundation-20261103/test-results.json)は `-n 0`、collect 80件/4.73秒、
80 passed/12.92秒。新規・変更28件、reader再利用50件、指定の既存node再利用2件。最長caseは小動画integrationの
setup+call+teardown 3.12秒、最大setupは丸め表示0.01秒で、5秒/1秒基準内だった。
`valid_for_scoring` を `observed` へ変える故障注入は手計算testが件数(3,2,5)→(2,2,4)を検出し失敗、
復元後に全80件が成功した。変更Pythonのruff/mypyを通した。同じintegration nodeで非対称な変更camera・共分散・時刻オフセットの保存復元も確認した。

実処理のscript timerは241.2秒、GNU timeのwallは233.44秒、同ツールの最大RSSは1,112,636KiB、swap 0。
時計の値は統合せず[資源記録](../../runs/run-ball3d-evaluation-foundation-20261103/resource-usage.txt)に原値を残した。
方式4foldのCPU時間合計はbaseline 50.37秒、LOESSなし46.25秒、oracle 8.67秒。読み込み・描画時間とは別である。

出力109ファイルのサイズと全SHAを再照合した。[artifact inventory](../../runs/run-ball3d-evaluation-foundation-20261103/artifacts.json)の
SHA-256は `43c56e13992a6bb9357c7fb399c5d92e6af154a0b74acc7b080ac459daee7e07`。
`index.html` は `bfed48651beac586adcc219a232e007a63c8033623fa53a2dc1122428f8b6306`、
`comparison.mp4` は `d943e6d4f8d6729d5e331b63745d850180ae79fa4a1080c16e273f26dc29d70c`。
予測結果8件・採点oracle4件を共通形式で保存し、全件をchecksum/type loaderで再読込した。
全指標JSONは265,505 bytesでknowledgeの256KiB上限を超えるためoutputsに残し、
小さな[区間集計](../../runs/run-ball3d-evaluation-foundation-20261103/run-summary.json)とchecksum目録だけを保存した。
学習はなくTensorBoard曲線はない。予測配列・動画・HTMLはoutputsに残し、knowledgeへは小さな記録だけを保存する。

## 制限と次の比較

1clipだけの固定camera比較であり、別clipへの汎化・独立3D精度・方式の採用は未決定。
65m/s・500m/s²・イベント±3frameの診断閾値は後半で調整していない。
後続6方式と #1568 は、共通runner・結果loader・採点・N方式rendererを再利用し、coverageと
方向転換付近の誤差を合わせて比較する。標準pipelineの既定・checkpointの採用判断は変更しない。
