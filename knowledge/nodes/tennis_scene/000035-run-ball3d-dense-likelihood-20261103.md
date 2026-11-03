---
id: run-ball3d-dense-likelihood-20261103
type: run
task: tennis_scene
sequence: 35
recorded_at: '2026-11-03'
title: clip_000のdense heatmap尤度による連続3D推定のCPU比較
issue: 1564
provider: codex
date: '2026-11-03'
status: done
config:
  clip_id: meiji_3cam/video_000/clip_000
  num_frames: 1010
  tuning_frame_rule: range(0,505,5)
  primary_frames:
  - 505
  - 1010
  selected:
    acceleration_scale_mps2: 100.0
    background: 0.05
    coarse_keep: 16
    coarse_levels: 3
    coarse_spacing_m: 2.0
    event_acceleration_mps2: 350.0
    initialization: candidates
    interpolation: bicubic_logit
    max_iterations: 40
    min_peak_probability: 0.05
    min_ray_angle_deg: 1.0
    multistart: 2
    support_logit_drop: 5.0
    temperature: 1.0
    temporal_weight: 0.0
    view_backgrounds:
    - - cam1
      - 0.2
    - - cam2
      - 0.2
    view_temperatures:
    - - cam0
      - 2.0
    - - cam2
      - 2.0
    window_size: 9
    x_bounds:
    - -12.0
    - 12.0
    y_bounds:
    - -24.0
    - 24.0
    z_bounds:
    - -0.2
    - 12.0
  gpu_used: false
  inference_rerun: false
metrics:
  baseline_primary_lovo_median_px: 9.340188442451916
  baseline_primary_lovo_p90_px: 27.078044438775773
  dense_primary_lovo_median_px: 16.46810054551058
  dense_primary_lovo_p90_px: 106.91403300537782
  dense_primary_lovo_over20_rate: 0.4204389574759945
  baseline_primary_lovo_count: 825
  dense_primary_lovo_count: 1458
  dense_primary_lovo_coverage: 0.9972640218878249
  paired_primary_count: 825
  paired_dense_primary_median_px: 12.67629619324541
  paired_dense_primary_p90_px: 36.828141314239524
  tests_passed: 121
  pytest_wall_seconds: 26.1
artifacts:
  run_dir: knowledge/runs/run-ball3d-dense-likelihood-20261103
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/dense_likelihood_v2
  metrics: knowledge/runs/run-ball3d-dense-likelihood-20261103/run-summary.json
  fold_metrics: knowledge/runs/run-ball3d-dense-likelihood-20261103/fold-metrics.csv
  checksum_inventory: knowledge/runs/run-ball3d-dense-likelihood-20261103/checksums.json
  verification: knowledge/runs/run-ball3d-dense-likelihood-20261103/verification.json
  test_results: knowledge/runs/run-ball3d-dense-likelihood-20261103/test-results.json
parents:
- run-ball3d-foundation-review-20261103
relations: []
papers: []
tags: []
repro:
  commit: 43dd9f4f6f1cb6a3da64e11a694d9d81de9d6d58
  command: bash knowledge/runs/run-ball3d-dense-likelihood-20261103/repro.sh BUNDLE
    VIDEOS GROUND_TRUTH_JSON FOUNDATION_V2 NEW_OUTPUT
---

## 結果: coverageは増え、精度は悪化

主評価[505,1010)のLOVOはbaseline 9.340/27.078pxから、前半で選んだdense方式の
16.468/106.914pxへ悪化した。>20px率は16.36%→42.04%。
採点数は825→1458/1462、coverageは56.43%→99.73%（+633点、+43.30pp）へ増えた。
同じ825点だけでも12.676/36.828px・>20px 28.00%となり、共通点でも悪化している。
全8条件の共通643点（43.98%）ではbaseline 9.025/24.848、選定方式12.374/34.083px。
欠測を回復したことと正確な推定を区別し、この結果から採用は勧めない。
後半を見て設定を変えていない。

| 区間 | 方式 | LOVO中央値/p90 px | >20px | 採点数/対象・coverage | 全view有効率 |
|---|---|---:|---:|---:|---:|
| 主 [505,1010) | pipeline_baseline | 9.340/27.078 | 16.36% | 825/1462・56.43% | 83.17% |
| 主 [505,1010) | dense_selected | 16.468/106.914 | 42.04% | 1458/1462・99.73% | 100.00% |
| 主 [505,1010) | dense_temporal | 16.580/108.543 | 42.22% | 1452/1462・99.32% | 97.62% |
| 前半 [0,505) | pipeline_baseline | 10.716/28.483 | 16.44% | 821/1497・54.84% | 86.34% |
| 前半 [0,505) | dense_selected | 17.618/142.590 | 46.05% | 1494/1497・99.80% | 100.00% |
| 前半 [0,505) | dense_temporal | 17.725/143.828 | 46.25% | 1481/1497・98.93% | 97.62% |
| 参考 [0,1010) | pipeline_baseline | 10.208/28.196 | 16.40% | 1646/2959・55.63% | 84.75% |
| 参考 [0,1010) | dense_selected | 17.007/126.547 | 44.07% | 2952/2959・99.76% | 100.00% |
| 参考 [0,1010) | dense_temporal | 17.075/127.731 | 44.26% | 2933/2959・99.12% | 97.62% |

各camera除外・全view・3区間の全条件は [fold-metrics.csv](../../runs/run-ball3d-dense-likelihood-20261103/fold-metrics.csv)。
直接観測と承認推定の内訳、イベント窓、粗大誤差、運動/加速度、元イベント25行と注記はoutputsのscores.jsonに保持する。
候補GMMのコード・結果には依存していない。

主評価の他のdense条件は、局所二次logit 15.586/223.710px（1320点）、
局所二次log 18.505/274.325px（1332点）、coarse 19.677/173.002px（1458点）。
3D時間priorは16.580/108.543px（1452点）で改善しなかった。
単純な補間精度の改善や出力増加がLOVO改善を保証しない。誤modeの選択と支持の緩さは原因の仮説で、
今回の比較から各寄与を因果的に分離していない。
独立推定から検出した急変境界は930箇所あり、時間版の窓を強く分割した。
これは検出器の雑音も含む境界数で、物理的なバウンド/打球の真値数ではない。
時間版では全区間24frame、後半12frameが欠測になった。実native mapの厳密な1は0画素で、
飽和値の扱いは合成テストの証拠に限る。



## 条件と方式

親 #1554 の合意revision 6、子 #1564。対象は固定bundleの公開clip_000の3camera・1010frameだけ。
GPU・検出器再推論・新しい入力取得は行わず、基盤 #1669 のloader、全view/3つのLOVO runner、
#1668の正解reader、採点・保存・可視化をそのまま使う。GTを方式入力へ渡さない。
方式・単位・支持と失敗の契約は [dense尤度README](../../../src/tennis_scene/ball_likelihood/README.ja.md) に集約した。
標準pipelineの既定triangulationとside/alignment入力は維持し、新方式はgate前のdense evidenceを使う。

先行 #1670 の候補回収余地とLOESS除去の悪化、#1672 の時間探索による欠測と誤選択、
#1673 の全view改善がLOVO改善へ一般化しなかった結果を読んで設計した。
これらは設計の根拠で、先行oracleや並行GMMの出力を推定入力には使っていない。

対数尤度は信号密度と一様背景の混合。bicubic-logit・局所二次logit・局所二次logを比べ、
候補pair初期化／初期値なしのcoarse-to-fine、3D加速度Cauchy prior版を保存する。
時間版は実測支持のある3Dだけを動かし、入力由来の急変と欠測で窓を切る。
2D LOESSや正解イベントは使わない。収束失敗を初期値で隠さず、欠測と理由を残す。

## 前半だけの選択とテスト

前半[0,505)の5frameおき101frameを事前指定し、各cameraの温度1/2、背景混合率0.05/0.2を
一度の座標探索で比較した。選択目的は共通scorerのmedian + 0.2*p90 + 30*(1−採点数/標本対象数)。
選択に使う予測は他frameをすべて無効にし、後半の採点値を読み出す入口はない。
同じscaleで3補間を比較し、選定後に設定を固定して全1010frameへ実行する。
時間priorの重み0.5は事前固定し、前半・後半どちらの採点にもfitしない。

[実装前のテスト設計](../../runs/run-ball3d-dense-likelihood-20261103/test-plan.md) に
契約・故障・層・既存node・コストを記録した。非対称camera/frame、独立pinhole式の既知投影、
交差項を持つ山、双線形の格子吸着、欠測とdistractor、m単位の勾配、急変とgapを検証する。
新規24case（unit20・integration4）。mainの計算ファイルと同じbytesから得た固定期待配列を、
既定config→definition→assembler→componentの出力と厳密比較する。
既存97caseを再利用し、全121case成功、収集6.43秒・実行26.10秒（-n 0）。
最大caseは実codec/動画/HTMLのintegrationで2.689秒、全setupは1秒未満だった。
全nodeと各phaseは [test-results.json](../../runs/run-ball3d-dense-likelihood-20261103/test-results.json)。

16frameのCPU試走は候補1.215秒・coarse1.158秒、両方16/16有効、プロセス最大RSS629928KiB。
全heatmapを対数領域へ複製せず、frame／短い窓の作業領域だけを作る。
実装と実行scriptをcommitしてから、固定設定による本実行を開始した。
この推定実験にはTensorBoard、学習checkpoint、seed間学習曲線は適用しない。

## 解釈の制約

単一clip・近似校正と推定点を含む2D正解の比較で、独立3D正解はない。
oracleは幾何参照で、数学的下限ではない。方式ごとの誤差と採点集合、全方式共通cohort、
baselineと選定方式だけの共通集合を区別する。欠測をゼロ誤差として扱わない。
候補seedとvolumeは限られた探索で大域最適の保証がなく、二次近傍の切替はline searchを難しくする。
入力由来の急変検出は雑音による過分割、有限窓は境界不連続を生じ得る。
採否・統合比較・mergeは人間と親の後続判断に残す。

## 成果物・目視・実行履歴

成果物は `/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/dense_likelihood_v2/`。550ファイルのサイズ/SHAを全照合、未宣言ファイルなし。
baselineとLOESSなしの全8結果はfoundation_v2のpositions/valid/used_viewsとbitwise一致した。
元storeとの照合はmask一致・3D差2µm許容内で、bitwise一致とは区別する。

比較MP4は1200×1150・1010frameを全decodeし、3つの入力動画も全decodeした。
frame 0/252/505/757/1009、指標表・軌道図、誤差上位782/811/810/409/827/826のcropを目視した。
元の観測/推定marker、unknown/末尾外挿の非描画、粗大な軌道の跳びを確認した。HTML参照切れ0。
尤度断面は正常564、遮蔽281、強い誤peak554frameのXZ面を、背景混合あり/なしで比較する。
代表frameの選択は設定凍結後の診断で、採点/前半調整以外に正解を方式へ渡す経路はない。

v1はcommit `6cc07b2ea708f3e636479c7d8fe9c4d113f5d854` で予測28結果と採点を完了した後、
遮蔽のNaNを図のJSONへ書こうとして停止した（wall1512.91秒、最大RSS1026484KiB、終了1）。
`executed-run-v1.py` とログ・partial inventoryを保全した。
v2はcommit `43dd9f4f6f1cb6a3da64e11a694d9d81de9d6d58` で欠測をnullへ修正し、
固定selectionと予測をchecksum検証して再利用した。設定の再調整・予測再計算は0回。
v1とv2の全scoresはruntime field以外で厳密一致する。v1ディレクトリを変更していない。
v2の後処理はwall235.62秒、最大RSS1187744KiB（約1.13GiB）、swap0。
予測CPU時間は各MethodRunに記録した元の実測で、後処理時間と区別する。

加速度priorを実行process内で無効化する故障注入では、揺れ低減の数値assertが失敗し、
別processで復元した同nodeが成功した。sourceの前後SHAは一致する（mutation.json）。
ruff/mypy成功。これらの検証はGPUを使わず、pytestは-n 0だった。

| 成果物 | SHA-256 |
|---|---|
| artifacts.json | `3b53a53ff5ca34bfdb6c4c764039bf1e3fe8eb3a4ea20b2a363f2fd6d97c9d6c` |
| scores.json | `630563ee095e30bf2d13f37a9c0659aebc46e83f1ecf8b0d613246eea4edc306` |
| index.html | `f0bbfde24fd92ffaaff0f139799f3964d54a5f93a98541c4ca7f5440d8647cca` |
| comparison.mp4 | `dfb07b17167cac63fc592f86ce6c1829e06895dabb311efe5c5f266a4dd4d82c` |
| selection.json | `8fc64fa4f1c58c11ca176ed405ace504ee08212eaece358bc87fc5fae160257d` |

全入力/成果物のchecksumと正確な実行履歴はchecksums.json、verification.json、execution-history.json。
新規出力への全CPU replayはrepro.sh（5引数）、保存済み予測の比較媒体再生成はexecution-v2.logとresource-usage.txtのコマンドを参照する。
人間の採否・#1568の統合比較・親のレビュー・mergeは後続の判断であり、この子はmergeしない。
