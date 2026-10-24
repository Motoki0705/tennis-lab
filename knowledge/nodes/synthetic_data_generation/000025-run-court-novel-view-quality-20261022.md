---
id: run-court-novel-view-quality-20261022
type: run
task: synthetic_data_generation
sequence: 25
recorded_at: '2026-10-22'
title: B00〜B03の新規視点と領域・画角・jitterの描画品質診断
provider: codex
issue: 1350
date: '2026-10-22'
status: done
config:
  generation_seed: 20261022
  nht_commit: 9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6
  output_resolution: [959, 539]
  camera_batch: 4
  allocation: all
  continuous_points_per_condition_scene: 96
  sparse_records_per_condition_scene: 36
  sample_design: intentionally_stratified_not_population_estimate
metrics:
  scene_condition_pairs: 16
  logical_camera_records: 2112
  unique_rendered_views: 2080
  capacity_probe_renders: 5
  total_rendered_images: 2085
  renderer_wall_seconds: 485.40677634693566
  capacity_probe_wall_seconds: 11.738596630981192
  gpu_device_used_peak_mib: 3869
  videos: 16
  contact_sheets: 16
  portable_cpu_tests_passed: 4
  related_cpu_tests_passed: 32
repro:
  commit: 53e5a6336a660cada3694f1532fc24bbf5daaee3
  command: 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-novel-view-quality-20261022/render_probe.py --scenes-root /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes --plan-root tmp/court-render-quality/run-01'
artifacts:
  run_dir: knowledge/runs/run-court-novel-view-quality-20261022
  output_dir: tmp/court-render-quality/run-01
parents: [run-court-nested-plan-cpu-20261022]
relations: []
papers: []
tags: [court, novel-view, render, diagnostic]
---

## 結論と範囲

B00〜B03を、提案中の0%拡張・0.5m内側・HFOV 75/92.5/110°・点ID独立1m jitterと、
拡張5%・HFOV 45/67.5/90°・jitter 0mの3つの一因子代替で新規描画した。
GPU処理は全てローカル共有training queue、公開`nht-render`だけで実行した。
動画16本とcondition×sceneのcontact sheet16枚、良例・失敗例を作成した。
**穴率0でも地面/ネットのぼけや線の不一致が残り、alpha/depth gateは画質の十分条件にならなかった。**
狭角では対象courtのKP支持が減り、特にB02/B03の採択が減った。
jitter 0mは視線の振れを減らしたが、複数courtのtarget切替と描画のぼけは解消しなかった。

採用前の診断であり、本runの完了は方式・品質閾値へのユーザー合意を意味しない。
判断依頼と2026-10-22の追加合意の正本は[設計文書§0/§5](../../reports/court-scale-trajectory-design.md)。
生成器の本実装・既存preset変更・Colab・Drive・学習・精度測定は対象外。TensorBoardは使用していない。

## 入力・比較の統制

[前回CPU run](000024-run-court-nested-plan-cpu-20261022.md)の実資産と幾何prototypeを用いた。
入力32ファイルは描画前後でSHA-256が一致した。最終bundleの計画器で別directoryへ再計画し、
plan/request等36ファイルのbyte一致を確認した（[検証記録](../../runs/run-court-novel-view-quality-20261022/input-validation.json)）。
B01旧configの欠損keyは前回と同じ実験内コピーの補完のみで、元の資産は変更していない。
NHTはメインtreeに既にinstallされたcleanな版を使用し、setup・source変更をしていない。

各scene/条件は連続circle 96点と、6軌道の各HFOV層から幾何のmin/max可視KPで選んだ36点。
2点の重複をcache再利用し130視点ずつ描いた。表示の132 recordと実描画130枚を区別する。
描画前に選んだstress probeであり、棄却率を本番全分布の推定値にしない。
near/far曖昧点・対象0KPも含め、productionならpre-renderで落ちる候補も診断として描画した。
連続動画は92.5°対67.5°の中央HFOV層を比較し、端の層はcontact sheet/QAに含む。

D2-bは位置・poseが完全一致しKだけが変わり、D6は位置・Kが完全一致して注視jitterだけが変わる。
D2-aは元の曲線パラメータと弧長割合を対応付けるため、位置・長さ・時にはnearest targetも変わる。
全12組×132 logical pointの対応を検証した（[pair checks](../../runs/run-court-novel-view-quality-20261022/pair-checks.json)）。
層と不足範囲は[層別表](../../runs/run-court-novel-view-quality-20261022/strata.csv)。
全体として高さ1.32〜3.07mだが、B01の標本は2.07〜2.51mで高さ範囲を網羅しない。
今回の標本でB02/B03のfullは0であり、層別選択だけではfullの支持を保証できない。

## 既存gateと対象court支持

分母は各条件130 unique camera。既存gateは全court合計のgeometry KP≥4かつrenderer KP>0、
near/far曖昧は棄却する。renderer支持は既存の3×3近傍α≥.01かつ正depth。
「対象4」はこれにtarget renderer KP≥4を追加する候補。
残存F/N/Pはgeometryのfull/near_full/partialで、nominal HFOVの層名とは別である。

| scene | 条件 | 既存gate採択 | 対象4採択 | 対象4の棄却率 | 残存F/N/P |
|---|---|---:|---:|---:|---|
| B00 | 推奨候補 | 113/130 | 106/130 | 18.5% | 0/31/75 |
| B00 | D2-a 拡張5% | 116/130 | 108/130 | 16.9% | 1/34/73 |
| B00 | D2-b 45–90° | 100/130 | 96/130 | 26.2% | 0/5/91 |
| B00 | D6 jitter 0m | 118/130 | 110/130 | 15.4% | 0/32/78 |
| B01 | 推奨候補 | 113/130 | 111/130 | 14.6% | 2/54/55 |
| B01 | D2-a 拡張5% | 113/130 | 112/130 | 13.8% | 3/62/47 |
| B01 | D2-b 45–90° | 102/130 | 100/130 | 23.1% | 0/24/76 |
| B01 | D6 jitter 0m | 112/130 | 110/130 | 15.4% | 2/53/55 |
| B02 | 推奨候補 | 120/130 | 120/130 | 7.7% | 0/2/118 |
| B02 | D2-a 拡張5% | 120/130 | 120/130 | 7.7% | 0/2/118 |
| B02 | D2-b 45–90° | 93/130 | 93/130 | 28.5% | 0/0/93 |
| B02 | D6 jitter 0m | 122/130 | 122/130 | 6.2% | 0/2/120 |
| B03 | 推奨候補 | 113/130 | 113/130 | 13.1% | 0/1/112 |
| B03 | D2-a 拡張5% | 114/130 | 114/130 | 12.3% | 0/1/113 |
| B03 | D2-b 45–90° | 78/130 | 78/130 | 40.0% | 0/0/78 |
| B03 | D6 jitter 0m | 111/130 | 111/130 | 14.6% | 0/1/110 |

推奨候補の既存gateは、B00でtargetのsparse6枚・none1枚、B01でsparse2枚も通していた。
対象4はそれらを除くが、B02/B03では既存gateから採択数が変わらず、ほぼpartialだけが残る。
同じgateでも残存coverageは揃わない。全sceneへtarget10以上を強制するとB02/B03の大半を失う。
全条件の可視KP分位点・正depth率・alpha/depth穴率は
[summary.json](../../runs/run-court-novel-view-quality-20261022/summary.json)、
採択後のHFOV/shape/target分布は
[gate-distributions.json](../../runs/run-court-novel-view-quality-20261022/gate-distributions.json)。

## D2の観測

全4sceneで狭角代替のtarget4採択数は推奨候補より少なかった。B02は120→93、B03は113→78。
狭角が3DGSを不良にした証拠ではなく、同じ位置・姿勢でcourtを画面に収めにくくなった結果。
広角75〜110°を比較の出発点に保ち、狭角を採る場合はpartial/negativeの扱いを別に決める材料になる。

5%拡張のtarget4採択は推奨に対してB00 +2、B01 +1、B02 ±0、B03 +1。
画質上の優位は確認できず、位置の追加外挿を必要とする根拠には弱い。
重要な限界として、今回の全点は拡張前hullの内側で、5%条件の最小clearanceも約0.86m。
「hullの外へ出ても安全」という試験にはなっていない。
hull内でもネット/地面のぼけが出るため、0%を品質保証として扱うこともできない。

## D5の感度と取り逃し

ROIはtargetのphysical corner polygonをnear plane/画面でclipしたもの。
数式・白線ridge指標・nullの扱い・完全な再現手順の正本は
[bundle README](../../runs/run-court-novel-view-quality-20261022/README.md)。
全16条件でROIのα<.01または非正depthの穴率は0、ROIの正depth率は1だった。
全画面の正depth率の最低は99.737%、alpha/depthの穴率の最大は0.699%だった。
全画面の低alpha穴は一部あるが、ROIの穴だけでは良い像とぼけた像を分けられない。
以下は推奨条件のtarget4に追加した閾値候補の残存数。各分母130で、採用値ではない。

| scene | 対象4のみ | ROI穴≤1% | ≤5% | ≤10% | ROIのα<.9面積≤1% | ≤5% | ≤10% | ≤25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B00 | 106 | 106 | 106 | 106 | 104 | 106 | 106 | 106 |
| B01 | 111 | 111 | 111 | 111 | 84 | 103 | 105 | 111 |
| B02 | 120 | 120 | 120 | 120 | 101 | 119 | 120 | 120 |
| B03 | 113 | 113 | 113 | 113 | 62 | 85 | 89 | 107 |

α<.9面積≤5%ではB01のpartialが55→47、B03のpartialが112→84となり、
より厳しいalpha条件はシーン・coverageを偏らせる。RGB品質の人手正解がないので、
この棄却を品質改善とは断定しない。1%・5%・10%を比較表示する候補として残す。
対象4と低alpha穴≤1%は最低限の生成不良検査の候補だが、画質gateの完成とは呼べない。

小さい定性図は[支持の多い例](../../runs/run-court-novel-view-quality-20261022/cases-clearer.png)と
[失敗・不確実な例](../../runs/run-court-novel-view-quality-20261022/cases-failure.png)。
B01 continuous 0はtarget renderer KP4、ROI穴0でもネット/地面がぼけていた。
B00 continuous 48もKP4・穴0で近いネット/地面が不鮮明。
B00 sparse 18はtarget1点でも既存gateを通り、投影線と明るいridgeの距離中央値が6.20px。
B02 sparse 12はtarget0点で、穴0でも対象が画面外の負例になる。
暗いB02/B03の「支持の多い例」でもridge距離が上限24pxになる。
これは簡易画像特徴が暗い細線を捉えられない例で、24pxのalignment誤差を意味しない。

したがって、対象KP支持の最低条件と、動画で確認する破綻/ぼけ/線一致は分けて判断する。
明るいridge距離を白線正解の再投影誤差へ読み替えず、固定閾値の自動quality gateは未確定とする。

## D6の観測

96点の閉軌道で隣接forward方向の角度差を測った。単位は度/隣接点で、角速度ではない。
複数courtのtarget切替があるB00/B01と、単一courtのB02/B03を分ける。

| scene | target切替 | 1mの中央値/p95 | 0mの中央値/p95 | 最大角1m/0m |
|---|---:|---:|---:|---:|
| B00 | 2 | 5.08/22.62 | 3.59/13.10 | 81.13/80.98 |
| B01 | 4 | 4.43/27.98 | 3.61/19.82 | 89.31/96.11 |
| B02 | 0 | 5.27/12.00 | 3.54/3.71 | 16.80/3.72 |
| B03 | 0 | 6.79/19.08 | 3.03/5.92 | 26.54/6.07 |

0mは特にB02/B03の振れを減らした。B00/B01ではtarget切替の大きなjumpが0mでも残る。
0mのtarget4採択は110/110/122/111で、推奨1mの106/111/120/113に対して一様な増減ではない。
描画品質や学習精度が0mで改善するという結論は出せない。
独立静止画の視線分布を優先する設計なら、stable IDの1mを第一候補に保ち、
動画の滑らかさを要求する用途なら0mを候補にする。連続jitterは未描画で採用しない。
最終選択は比較動画を見てユーザーが行う。

## 動画と通常検証

ローカルの閲覧入口:

```text
/home/kamimura/projects/tennis-lab/.claude/worktrees/court-scale-trajectory-design/tmp/court-render-quality/run-01/review/index.html
```

同じdirectoryに各sceneの`<scene>-recommended-rgb-overlay.mp4`、
`<scene>-compare-hull5.mp4`、`<scene>-compare-hfov45_90.mp4`、`<scene>-compare-jitter0.mp4`、
`<scene>-<condition>-contact.jpg`、`representative-cases.jpg`がある。
完全な一覧・hash・codec/寸法は[media manifest](../../runs/run-court-novel-view-quality-20261022/media-manifest.json)。
MP4は全16本H.264/yuv420p、各96 frame・8秒で、全frameをffmpegでdecodeした。
RGB/overlayは960×338、上下比較は960×676。動画・JPEG・raw配列はgit管理外。

本比較の描画485.41秒、容量測定11.74秒、合計497.15秒（約8分17秒）。
GPU全体の観測peakは3869MiB。batch1の1枚は7.54秒/2446MiB、batch4の4枚は4.20秒/3687MiB。
CPU QA/動画生成は6分25秒・最大RSS約930MiB。renderer終了後にNHTのCUDA processが残らないことを確認した。
ディスクraw出力を一括でhost/GPUに保持しない。今回のrateをColab・大規模生成の予算値へ流用しない。

[検証記録](../../runs/run-court-novel-view-quality-20261022/validation.txt)にコマンドと結果を残した。
追加4テストはpaired変数の不変性、unbounded hullからの再fit、near clipping、
ROI穴率と既存visibilityを検査する。前回8テストと併せた12件も実行した。
中断後に`__pycache__`を除去し、再現入口もbytecodeを生成しない設定にした。
32入力・36再計画ファイル・34 mediaのhash、16動画の全frame decode、保存QAの計数を
[残存成果監査](../../runs/run-court-novel-view-quality-20261022/resume-audit.json)で再確認した。
再描画は行っていない。既存8＋追加4テストとCourt Reviewの20テスト、計32件が通った。
全Pythonへのruff/mypy、knowledge validation、repro参照検査、コミット時pre-commitは検証記録を参照。

## 残る確認

今回の標本にRGBの独立品質正解はなく、白線ずれのrenderer/alignment要因も分離しない。
B01の低い/高い高さ、B02/B03のfull、hull外側、他の連続形状・視線の支持は未検証。
本番の候補分布での採択率・残存coverage、VRAM/RAMの安全上限は#1356の全分布試走で確認する。
今回の診断用pathや数値を本番planへ流用しない。
2026-10-22の合意により、合成は全てtrain、実画像の動画単位val/testは#1354、
枚数は全採択候補の実測値となった。合成評価・評価guard・枚数揃えのquota/補充は不要。
D2/D5/D6は推奨と動画を材料に#1352のUI比較で最終判断する。候補仕様と診断材料が揃う#1350は完了可能。
