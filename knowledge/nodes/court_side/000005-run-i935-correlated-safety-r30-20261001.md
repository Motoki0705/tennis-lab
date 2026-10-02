---
id: run-i935-correlated-safety-r30-20261001
type: run
task: court_side
sequence: 5
recorded_at: '2026-10-01'
title: 実GMM残差とconfidenceを結合した固定court_side安全bench
issue: 935
provider: codex
status: failed
config:
  conditions: 28
  scenes: 400
  seed: 1
  residual_seed: 30001
  min_presence: 0.9
  max_area_px2: 30000
  side_margin: 0.15
  pass_rule: 0 wrong / 11200
metrics:
  original_wrong: 0
  joint_unfiltered_wrong: 0
  joint_filtered_wrong: 2
  joint_unfiltered_stop_rate: 0.35348214285714274
  joint_filtered_stop_rate: 0.2842857142857143
  elapsed_seconds: 315.2561029329663
artifacts:
  run_dir: knowledge/runs/run-i935-correlated-safety-r30-20261001
parents:
- run-i935-filtered-side-safety-r29-20261001
- run-i935-covariance-loco-s42-r23-20260930
relations: []
papers: []
tags: []
date: '2026-10-01'
repro:
  commit: f862049b5034174c5c9091e2c977e536c9f4c6c5
  branch: campaign930/i935-14-pipeline-candidate
---

## 判定: FAIL（全件実行は完了）

[実行前に投稿した方法・0件合格規則](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5923148819)に対し、
**filtered誤判定2/11,200でFAIL**。confidence規則・court_side .15・ball-onlyを変更していない。
clip_000 qualificationは提案/投入せず、次の方針は未選択とする。独立validator指定/試行/完了0。

| 入力 | wrong / 11,200 | 平均停止率 |
|---|---:|---:|
| 元#932摂動（全28条件の元集計を完全再現） | 0 | 18.5804% |
| 同じ摂動＋joint実GMM残差、選別なし | 0 | 35.3482% |
| 同一GMM＋固定confidence規則 | 2 | 28.4286% |

[report](../../runs/run-i935-correlated-safety-r30-20261001/result/report.json)、
[全33,600判定のevidence](../../runs/run-i935-correlated-safety-r30-20261001/result/evidence.jsonl.gz)、
[実行log](../../runs/run-i935-correlated-safety-r30-20261001/bench.log)を保存した。
CPU単一process/torch1thread、315.26秒。GPUは使用していない。

## 方法と外的妥当性

較正済みbank `0697fe92…` / calibration `2c9cd7ba…`のobservedだけを使用。
Meiji valのclip_001/002/003/006/007/010、各cameraの同一実frameから4成分error_uv・
scale_tril・mixture logits・presence logitsを一緒に抽出する。clip_000/video_001は含まない。
最大30frame block/stride2、続いていない教師やartifact境界で切る。cameraを最初の3viewへ
permutation、4台目は独立抽出。全frameが同じrowに結びつくことをテストした。

元の28条件・seed1・heldout test[600:1000]の校正、同期、欠測、pixel noise、static false
摂動を保持し、そこへ実残差全成分を加える。平均を画像範囲にclipし、productionの
point_confidenceで全GMMの保守的90%包含面積を計算する。共分散は既存倍率のまま。
詳細は[事前protocol](../../runs/run-i935-correlated-safety-r30-20261001/protocol.md)を正本とする。

| 実際の点の誤差（全条件の可視camera-frame） | 選別保持 median / p90 px | 棄却 median / p90 px |
|---|---:|---:|
| 実GMM移植分（元の摂動点との差） | 1.600 / 9.975 | 58.738 / 271.196 |
| 総2D誤差（各streamのsource frameの真の投影との差） | 2.859 / 198.923 | 70.163 / 364.621 |

保持3,829,464点、棄却1,277,940点。総2D誤差にはstatic false/pixel noiseを含むが、同期ずれ・校正誤差による3D整合性への影響は含まない。real residualのconfidence/error相関は保たれ、
低誤差点を多く残す挙動を確認した。それでも残したstatic false等の追加stressは
実refinerが見分けたという仮定を置いていない。E2E映像への推論結果ではなく、
JPEG bankの選択/calibrationとsynthetic geometryへの移植の限界がある。
run29との差はmaskだけではなく入力誤差モデルの追加でもあるため、両runの停止率差を
モデルの改善量とは解釈しない。元#932とrun29の失敗記録は維持する。

## 新しい2wrong case

| 条件 / scene | camera順 | 正解→採用 | distinct / pair(01,02,12) | margin |
|---|---|---|---:|---:|
| false_0.30 / scene_007579 | cam_2,cam_1,cam_3 | FTF→FFF | 39 / 28,21,14 | .234289 |
| false_shared_0.30 / scene_007141 | cam_0,cam_1,cam_2 | FFT→FFF | 38 / 23,10,23 | .280742 |

元入力・保持mask・confidence・実bank row ID・真の投影をcase別NPZに保存した。
両件ともpairの全edgeは8以上で、run29の直接edge消失だけを必要原因とはできない。
この結果から必要frame数やmarginを事後設定しない。

## run29の3件の分解

[解析本文](../../runs/run-i935-correlated-safety-r30-20261001/run29-analysis.md)と
[全camera/point/source-frame/仮説支持](../../runs/run-i935-correlated-safety-r30-20261001/run29-analysis/analysis.json)を保存。
全仮説のfloat・pair数・distinct数が既存記録に完全一致した。

- false_0.10: cam_3の真の27点が全て消え、8偽点だけが多視点支持に残る。
  referenceとのpair0、別cameraを介する8frame edgeは既存条件を満たす。
- false_0.20: cam_3の真の32点が全て消え、14偽点だけが残る。元判定はmargin .044でSTOP、
  filter後は偽点14frameを支持する誤仮説がmargin .168で通過する。
- window_150f: 偽点0、保持点も全て20px以内だが、cam_2とreferenceの直接pair61→2。
  残した区間では誤仮説が27/27を支持し、正解19/27を上回る。

## 未選択の次案

本runのdirectiveに従い、以下は選択肢の提示のみ。どれも自動実装/新GPU投入しない。

- 最小支持条件・camera/pairごとの識別可能性条件を新しいruleとして事前定義し、別の開発/検証分割で調べる。
  新規ユーザー判断が必要で、今回のcaseを消すための数値調整はしない。
- 映像上の誤検出とconfidenceを一緒に観測できる独立refiner実出力benchを追加する。
  synthetic static falseの相関限界は減るが、入力・正解・追加予算の決定が要る。

既存の安全判定FAILによるqualification保留は維持する。Meiji context-cacheは別の承認済み作業であり、
このFAILを球経路の合格に読み替えず生成だけをqueueへ投入する。TensorBoard/学習曲線は対象外。

## 条件別集計

| 条件 | 元 wrong/stop | joint未選別 wrong/stop | joint選別 wrong/stop |
|---|---:|---:|---:|
| nominal | 0 / 0.00% | 0 / 6.25% | 0 / 4.25% |
| missing_0.00 | 0 / 0.00% | 0 / 7.50% | 0 / 2.75% |
| missing_0.30 | 0 / 0.00% | 0 / 7.75% | 0 / 6.00% |
| missing_0.50 | 0 / 2.50% | 0 / 9.75% | 0 / 13.25% |
| missing_0.70 | 0 / 23.00% | 0 / 29.50% | 0 / 43.75% |
| missing_0.85 | 0 / 97.00% | 0 / 98.25% | 0 / 100.00% |
| false_0.05 | 0 / 4.50% | 0 / 24.50% | 0 / 14.00% |
| false_0.10 | 0 / 6.50% | 0 / 31.00% | 0 / 16.50% |
| false_0.20 | 0 / 13.50% | 0 / 64.00% | 0 / 37.00% |
| false_0.30 | 0 / 39.00% | 0 / 86.75% | 1 / 65.25% |
| false_0.50 | 0 / 86.25% | 0 / 99.00% | 0 / 95.75% |
| false_shared_0.10 | 0 / 0.25% | 0 / 8.75% | 0 / 4.75% |
| false_shared_0.30 | 0 / 3.00% | 0 / 17.25% | 1 / 10.75% |
| sync_1f | 0 / 0.75% | 0 / 9.75% | 0 / 5.25% |
| sync_2f | 0 / 3.75% | 0 / 28.75% | 0 / 12.75% |
| sync_4f | 0 / 40.25% | 0 / 82.75% | 0 / 58.75% |
| sync_8f | 0 / 93.00% | 0 / 97.50% | 0 / 95.00% |
| pixel_sigma_0px | 0 / 0.00% | 0 / 6.50% | 0 / 4.50% |
| pixel_sigma_5px | 0 / 0.00% | 0 / 5.00% | 0 / 2.25% |
| pixel_sigma_10px | 0 / 0.00% | 0 / 10.50% | 0 / 4.00% |
| calibration_x0 | 0 / 0.00% | 0 / 4.75% | 0 / 1.75% |
| calibration_x2 | 0 / 7.50% | 0 / 26.75% | 0 / 15.75% |
| calibration_x4 | 0 / 54.50% | 0 / 74.00% | 0 / 62.00% |
| window_30f | 0 / 24.75% | 0 / 44.50% | 0 / 53.00% |
| window_60f | 0 / 5.50% | 0 / 23.50% | 0 / 21.75% |
| window_150f | 0 / 0.25% | 0 / 10.25% | 0 / 3.75% |
| cameras_4 | 0 / 0.00% | 0 / 16.50% | 0 / 7.50% |
| combined | 0 / 14.50% | 0 / 58.50% | 0 / 34.00% |
