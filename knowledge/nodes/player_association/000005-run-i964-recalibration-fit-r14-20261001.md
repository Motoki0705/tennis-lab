---
id: run-i964-recalibration-fit-r14-20261001
type: run
task: player_association
sequence: 5
recorded_at: '2026-10-01'
title: '無ラベル6clipのLOVO較正と固定後dev一回比較'
issue: 964
provider: codex
date: '2026-10-01'
status: running
metrics: {lovo_positive_recall: 0.8592682453416148, lovo_negative_false_join: 0.0, sigma_m: 0.7381677290433, slope: 36.70286491794044, center: 0.8298172161822686}
repro: {commit: 6bafe87c635eaaf2d1136300c469a9dcc1031fdf}
artifacts:
  run_dir: knowledge/runs/run-i964-recalibration-fit-r14-20261001
parents: [run-i964-recalibration-resume-r13-20261001]
relations: [{to: run-i933-association-meiji, rel: compares}]
---

[run12事前protocol](../../runs/run-i964-recalibration-r12-20260930/protocol.md)を変更せず、
全18cameraの検証済み特徴から6clipをCPU再追跡してfitした。
[fit全証拠](../../runs/run-i964-recalibration-fit-r14-20261001/fit.json)は全pair-window、
除外理由、階層重み、camera×近遠支持、3fold尺度/loss/停止を含む。
[入力identity](../../runs/run-i964-recalibration-fit-r14-20261001/identity.json)、
[出力hash](../../runs/run-i964-recalibration-fit-r14-20261001/output_manifest.json)、
[dev前の記録](../../runs/run-i964-recalibration-fit-r14-20261001/pre-dev.json)を固定した。
fit呼出し1回、全data最終fit1回、dev採点0回の時点で本記録と
[名前付きYAML](../../../src/tasks/player_association/configs/association_i964_r14_lovo_a.yaml)をcommit/pushする。
既定configは変更せず、ユーザーの既定変更・凍結判断を待つ。

| candidate | margin / runner-up | positive recall | negative誤結合（各動画） | 正例棄却 | 採否 |
|---|---:|---:|---:|---:|---|
| A | 1.0 / .5 | .859268 | 0 / 0 / 0 | .140732 | 採用可能・同点順で選択 |
| B | 2.0 / .4 | .859268 | 0 / 0 / 0 | .140732 | 採用可能 |
| C | 4.0 / .3 | .687393 | 0 / 0 / 0 | .312607 | recall不足 |

| held video | 検証pos/neg | fit外観pos/neg | sigma | slope | center | loss | A/B recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| video_000 | 48/48 | 37/62 | .646524 | 36.544439 | .827895 | .301390 | 1.000000 |
| video_001 | 33/58 | 50/51 | .828189 | 39.362058 | .836327 | .277392 | .549658 |
| video_002 | 24/24 | 55/81 | .723926 | 34.886602 | .825331 | .349214 | 1.000000 |

全foldの支持条件を満たし、sigma最大/最小=1.28099（上限2）、
slope比=1.12829（上限3）。A/Bはvideo_001/clip_020で停止し、Cはさらに
video_000/clip_003で停止した。停止の正例はrecall=0として母数に残した。
動画ごとの正例80%は要求していないためvideo_001の54.97%を隠さない。
全体recallはprotocolの階層重みで計算し、動画別recallの単純平均ではない。

全dataの幾何pos105/neg130 pair-window中、外観有効は71/97。
最終sigma=.7381677290433、slope=36.70286491794044、center=.8298172161822686、
固定L2付きlogistic loss=.3104585352447031、14iterationで収束した。
Aは旧閾値と同じで、変更は3尺度のみ。学習曲線は該当せずoptimizerのloss/iterationを記録した。
較正の擬似一致は人間ラベルの精度を保証しない。近距離正例の切断バイアス、
CLIPを含む固定選別による依存、同一収録・選手、注釈ball由来のsideという限界を維持する。

次はこのYAML/hashをpushした後、投影後の同一track/CLIPに旧/新尺度を適用する
4dev一回比較を行う。devによる再fit/再選択を行わず、予約未見は閉じておく。
