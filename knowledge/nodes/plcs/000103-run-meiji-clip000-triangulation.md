---
id: run-meiji-clip000-triangulation
type: run
task: plcs
sequence: 103
recorded_at: '2026-09-21'
title: Meiji clip_000 COCO17三角測量の初期3Dポーズ検証
provider: codex
date: '2026-09-21'
status: done
config:
  model: score-weighted DLT + pixel Gauss-Newton, no learned model
  loss: confidence-weighted pixel squared error, 12 iterations
  data: data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
  min_score: 0.3
  device: cpu
metrics:
  joint_coverage_percent: 98.90215492137449
  reprojection_mean_px: 3.2497588311352383
  reprojection_median_px: 2.403119239851321
  reprojection_p95_px: 9.731673956899835
  pair_solution_disagreement_median_m: 0.260362892657745
  pair_solution_disagreement_p95_m: 0.8008198452624251
  limb_instances_over_1m: 28
  person_frames_with_limb_over_1m: 26
  worst_limb_length_m: 3.0382538855924532
repro:
  commit: 893d0ca4
  branch: codex/meiji-clip000-triangulation
  command: bash knowledge/runs/run-meiji-clip000-triangulation/repro.sh
artifacts:
  run_dir: knowledge/runs/run-meiji-clip000-triangulation
  predictions: knowledge/runs/run-meiji-clip000-triangulation/predictions.npz
  metrics: knowledge/runs/run-meiji-clip000-triangulation/metrics.json
  output_dir: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/meiji-clip000-triangulation
  log: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/meiji-clip000-triangulation/render.log
parents:
- run-tennis-scene-meiji-clip000-plcs-epoch19-assoc-fix
relations: []
papers: []
tags:
- plcs
- triangulation
- real-data
- cpu
- geometry
---

## 考察 / Findings

### 要約

指定クリップの3カメラ・2人・1,010フレーム（59.94006 fps、16.85秒）から、コート座標のCOCO17を復元した。
全34,340関節の98.90%を復元でき、入力2Dとの再投影誤差は平均3.25px、中央値2.40px、95%点9.73pxだった。
人物位置・全身姿勢の初期値として検討できる一方、遠側選手の腕には欠測と大きな外れ値が残る。
**独立3D正解はなく、MPJPEや真の位置誤差をcm単位で測定した結果ではない。**

### アーキテクチャ詳細

- 入力は `scene.npz` の `human_kp_2d` と `human_kp_vis`。保存済みViTPose由来の検出値であり、手動の人体アノテーションではない。
- 人物対応はP0=`[cam0 local0, cam1 local0, cam2 local1]`、P1=`[local1, local1, local0]`。sceneには適用済みであり再適用しない。
- 同じframe indexを同期時刻として使用。clip_studioの既存同期を使用し、追加の同期最適化は行わない。
- カメラはmetadataの `K/R/t` を再利用。CourtKP14だけから、主点=画像中心・`fx=fy`・歪みゼロを仮定して推定した近似値であり、実測済み校正ではない。
- コート14点への座標RMSEはcam0/1/2で4.008/3.418/2.164px。これは `sqrt(mean(dx²,dy²))` で、人体評価のEuclidean距離とは定義が異なる。保存済み値を再計算して一致確認した。
- confidence ≥ 0.3の有限観測を使用し、重みは `clip(score,0,1)`。confidenceは校正済みの誤差分散ではない。
- 重み付きDLTから開始し、12回のGauss–Newton更新で重み付き画素誤差を最小化。2視点未満・数値退化・使用カメラで負深度の場合はNaNとvalid=false。時間平滑化、骨長制約、GTによる整列、PLCS補正は使わない。
- 出力は `X_init[2,1010,17,3]`、単位m、XY=地面、Z=上。各人ごとのshapeは要求通り `[1010,17,3]`。
- `reprojection_residual_px = p_original - project(X_init)` も保存。生の残差とconfidence/maskを分けて保存し、未使用の観測を使用済みに見せない。

### メトリクスの解釈

| 指標 | P0 | P1 |
|---|---:|---:|
| 復元できた関節 | 100.00% | 97.80% |
| 17関節すべて復元できたframe | 100.00% | 71.49% |
| 腕・脚の骨長の「自身の時系列中央値」からの偏差：中央値 | 1.80cm | 2.44cm |
| 同偏差：95%点 | 5.90cm | 22.76cm |

骨長は各人・各骨ごとに時系列中央値を計算し、その長さからの変動を測る。
人体の真の骨長・関節位置との比較ではない。P1の右手首は80.69%、左手首は87.92%のframeで復元できた。
欠測関節は補間・ゼロ埋めしない。

2台だけで復元し、その計算に使用しなかった1台の2D検出で確認した結果は次の通り。
各行は全3カメラでconfidence ≥ 0.3かつ復元可能な30,075関節を評価している。

| 復元に使用 | 確認カメラ | 平均px | 中央値px | 95%点px |
|---|---|---:|---:|---:|
| cam1 + cam2 | cam0 | 12.00 | 6.98 | 42.51 |
| cam0 + cam2 | cam1 | 17.91 | 10.62 | 64.82 |
| cam0 + cam1 | cam2 | 21.44 | 12.33 | 75.19 |

異なる2台組から得た3D関節同士の距離は中央値26.04cm、95%点80.08cm。
3種類の組合せ間の距離を集約した90,225値であり、観測選択への感度を示す。真の3D誤差や信頼区間ではない。
同じ3D点でも各cameraで人物の画面上の大きさが違うため、pixel誤差の大小だけをcamera精度の順位と解釈しない。

失敗例はP1のframe 320。左前腕の復元長が3.038mとなった。左手首confidenceはcam0=0.317、cam1=0.216、cam2=0.896で、cam0とcam2だけを使用した。
そのframeの使用2視点の視線角は約14.63度。遠景の低いconfidenceを持つ観測を含んだ復元で、cam1への投影は大きく外れる。
P1では26/1,010フレーム、合計28個の四肢骨長が1mを超えた。これらを集計から除外していない。

### アーキテクチャ⇄メトリクスの因果考察

再投影最適化に使った観測との誤差が小さいこと自体は期待される挙動で、高い3D精度を保証しない。
未使用cameraへの誤差増大と骨長異常は、初期値に修正すべき誤差が残ることを示す。
カメラ校正誤差・レンズ歪み・残る同期誤差・2D検出誤差の寄与は、本検証では分離できていない。
特にcam0は広角映像であり、歪みゼロの近似が誤差へ寄与する可能性は仮説として残る。

提案中のPLCSへの入力にはX_initと再投影残差に加え、元confidence、各cameraの使用mask、joint valid mask、視線角を保持する必要がある。
2視点の残差が小さくても、誤った対応・観測や悪い交差条件では3Dが大きく外れ得るためである。
欠測点へのΔXだけではNaNを修復できないので、欠測関節の直接補完と残差補正の契約は実装前に分けて決める。

### 既存実験との比較

親run `run-tennis-scene-meiji-clip000-plcs-epoch19-assoc-fix` の人物対応と保存済み2D観測を再利用した。
既存PLCSのrootと本runの17関節は評価対象が異なるため、以前のroot再投影誤差との差を改善率として報告しない。
モデルやデータの再学習・再推論は行っておらず、全計算はCPUで実施した。

### 次に有効な実験

まず、既知3D関節を持つ合成マルチビューから同じinitializerを評価し、2Dノイズ・camera誤差・欠測別のMPJPEを測る。
実クリップでは手首の誤検出・同期・広角歪みを個別に確認し、幾何の外れ値処理と時系列補正を同じ評価指標で比較する。
これにより、PLCSが人物観測の誤差を直すのか、camera校正の誤差を吸収するのかを区別できる。

通常検証：既知3Dの厳密復元、欠測/低confidence view、再投影誤差の改善、退化/負深度、未使用cameraの投影面特異点を含むpytest 4件が成功。
geometry/evaluate/renderのmypyを通過。source metadataのcamera順・人物対応、CourtKPの並べ替え、再投影RMSEの再計算も検証した。
validatorの実行は今回の依頼では指定されていない。

再現用の当時のanalysis CLIはbundleのuncommitted.patchに含まれる。再現時は記録commitの専用worktreeへこのpatchを適用してからrepro.shを実行する。現在の残差profile CLIとは別の歴史的記録である。
