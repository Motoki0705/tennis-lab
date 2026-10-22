---
id: run-court-hybrid-fit-search-20261021
type: run
task: court_detection
sequence: 45
recorded_at: '2026-10-21'
title: Court 2Dの候補再探索：PROSAC向き修正後のval先頭300件
issue: 1207
provider: codex
session: 01a15e3f-f433-79b2-8cef-f72bfe4e4cda
date: '2026-10-21'
status: done
config:
  checkpoint: court_detection/multiscale_depth3/b863df1f01f0.ckpt
  device: cuda
  source: tennis_court_detector
  split: val
  search:
    min_kp_only_error_px: 10.0
    max_hybrid_error_px: 3.0
    max_samples: 300
    top_k: 20
metrics:
  scanned_count: 300
  qualified_count: 2
  kp_only_exists_count: 300
  hybrid_exists_count: 300
  hybrid_at_most_3px_count: 282
repro:
  commit: 12efecfab4aad0565a949e09b835055efa15d010
  branch: codex/court-2d-hybrid-fit
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.court_detection.scripts.visualize_hybrid_fit
    mode=search paths.data_root=/home/kamimura/projects/tennis-lab/data paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    search.min_kp_only_error_px=10 search.max_hybrid_error_px=3 search.max_samples=300 search.top_k=20
artifacts:
  run_dir: knowledge/runs/run-court-hybrid-fit-search-20261021
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/0-8207417109857210616_2026-10-21T11-41-30-142789383Z_2911667_court-hybrid-fit-search.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/court_detection/visualize/hybrid_fit/20261021T121234.070016Z-5a06c602/presentation
  ranking: knowledge/runs/run-court-hybrid-fit-search-20261021/ranking.json
  contact_sheet: knowledge/runs/run-court-hybrid-fit-search-20261021/contact_sheet.png
  verification: knowledge/runs/run-court-hybrid-fit-search-20261021/artifact-verification.json
  manifest: knowledge/runs/run-court-hybrid-fit-search-20261021/manifest.json
parents: []
relations: []
papers: []
tags:
- presentation
- hybrid-homography
- gpu-validation
- material-selected
---

## 観察

PROSACの向き修正を含むcommit `12efecfab4aad0565a949e09b835055efa15d010`で、
TennisCourtDetectorのval先頭300件をGPUで再推論した。前回探索とのsample ID順序、
JPEG・RGB・注釈recordのhash、dataset descriptor/index、checkpointのhashは一致した。
これは素材探索であり、会場を分離したtest評価ではない。

条件はKPだけのHが存在して誤差10px以上、hybridが成功して誤差3px以下、最大300件・上位20件。
条件を緩めずに2件が該当した。誤差は同じ可視GT KP14での原画像pixelの平均距離である。
KPだけのHとhybridはともに300件で成立し、hybridの3px以下は282件だった。
前回の保存結果ではKPだけのHが全件不成立・該当候補0件で、hybridの3px以下も282件だった。

| 順位 | sample_id | KPだけ (px) | KP＋LINE (px) | 改善量 (px) |
|---|---|---:|---:|---:|
| 1 | BiOnysEJdYE_2200 | 16.396553305626206 | 1.7338136461290754 | 14.66273965949713 |
| 2 | UwuccciC0Jg_100 | 10.625403295558375 | 2.8945211888325653 | 7.73088210672581 |

順位は改善量の降順、hybrid誤差の昇順、sample IDの昇順で決めた。
2件だけなので3件目は追加していない。
[2026-10-22のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/1207#issuecomment-6331369515)で、
1位の`BiOnysEJdYE_2200`の描画済みGPU版を採用し、2位は採用しないことが決まった。
採用した素材と再生成手順は[描画runの採用記録](000046-run-court-hybrid-fit-bionys-20261021.md)を参照する。

[全300件の結果](../../runs/run-court-hybrid-fit-search-20261021/ranking.json)と
[候補一覧](../../runs/run-court-hybrid-fit-search-20261021/contact_sheet.png)を保存した。
[検証記録](../../runs/run-court-hybrid-fit-search-20261021/artifact-verification.json)には
検索・GPU描画2件の全成果物のSHA-256、フレーム検査、入力・checkpointのhash照合を含めた。
GPU描画の誤差は探索と完全一致した。描画の個別記録は
`run-court-hybrid-fit-bionys-20261021`と`run-court-hybrid-fit-uwuccci-20261021`に分ける。

## 解釈と次の確認

この2件は改善を見せるために条件で選んだ例であり、全画像への精度優位や下流3D精度の証拠にはしない。
CPU参考版とGPU版には小さな誤差差があり、原因を分離していない。素材に添える数値はGPU版を使う。
画像・予測・seed・fitはmeasured、色と時間の演出はschematicとmanifestに記録している。
推定器の変更後には同じ入力とcheckpointで候補条件・投影を再確認する。
学習は実行していないのでTensorBoardの学習曲線はない。
