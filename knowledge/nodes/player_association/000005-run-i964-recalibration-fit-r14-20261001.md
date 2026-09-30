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
status: done
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

## 固定後のdev一回結果

YAML/fit証拠のcommit **b03eb152** をpushした後、commit内のblobとoriginへの到達、
旧config・入力hashをCPU入口で検証し、**937da271**のコードで1batchだけ採点した。
[比較JSON](../../runs/run-i964-recalibration-fit-r14-20261001/dev/comparison.json)と
[回収記録](../../runs/run-i964-recalibration-fit-r14-20261001/dev-collection.json)を正本とする。
旧値は保存精度をコピーせず、run13でproduction maskへ整合した同じtrack/CLIPから再計算した。
投影に伴う2camera再追跡後でも、旧尺度の全4clipのstatus/metricsはrun11と完全一致した。
旧/新の予測ID配列も全12cameraで完全一致し、尺度変更によるdev改善はなかった。

| 条件 | decided | pair TP/FP/FN | F1 | group accuracy | exclusion TP/FP/FN | label box coverage |
|---|---:|---:|---:|---:|---:|---:|
| 旧尺度/閾値 | 4/4 | 18459/6/1648 | .957119 | 2663/3489=.763256 | 4432/840/3 | 25055/25147 |
| 新尺度/A | 4/4 | 18459/6/1648 | .957119 | 2663/3489=.763256 | 4432/840/3 | 25055/25147 |

| clip | 旧/新 pair TP/FP/FN | 旧/新 F1 | 決定 |
|---|---:|---:|---|
| video_000/clip_000 | 5867/0/0 | 1.000000 | 両者ok |
| video_000/clip_007 | 3142/0/512 | .924662 | 両者ok |
| video_001/clip_001 | 6380/6/1044 | .923968 | 両者ok |
| video_002/clip_013 | 3070/0/92 | .985237 | 両者ok |

停止は両者0件。除外precision=.840668 / recall=.999324。
2D追跡は両者共通でraw IDF1=.955395、switch/fragment=4/21、
選別後group IDF1=.969915、switch/fragment=1/21、選手unit保持19659/20558、
既知非選手3/4454、人物50%保持8/8。
対応評価のtrack上の人物変化はtrue3/predicted5/matched1（camera内MOT switchと定義が異なる）。
未照合予測boxは15114、うちID付き67で、誤検出とはみなさない。
既存COCO box由来の部分ラベルであり、coverageが高くても完全検出recallではない。

| camera/近遠 | raw/group IDF1（共通） | 対応後保持unit 旧/新 | label player units |
|---|---:|---:|---:|
| cam0 near | .996471 / .996471 | 3247 / 3247 | 3270 |
| cam0 far | .878737 / .950176 | 3253 / 3253 | 3270 |
| cam1 near | .998686 / .998686 | 3421 / 3421 | 3430 |
| cam1 far | .896410 / .915072 | 2893 / 2893 | 3430 |
| cam2 near | .968760 / .968760 | 3163 / 3163 | 3367 |
| cam2 far | .983547 / .983547 | 3258 / 3258 | 3367 |

近遠は既存評価と同じframe内のGT box下端順位。unknownもCSV/圧縮unit記録に残した。
camera別label box照合はcam0=8283/8308、cam1=8213/8241、cam2=8559/8598。
全指標・除外・混同行列・停止診断・予測ID・unitをdev/に保持する。
この結果は「旧尺度がpair F1低下の主因」という見方を裏付けず、
選別/coverage/短い区間の扱いなどを区別した後続診断が必要である。
このdevを使った再fit/候補再選択は実施しない。
合成fit/gate16件と、push前の採点禁止・一回gate・棄却の母数・元trackへの投影4件が成功した。
独立validatorの指定はなく0回。次は名前付き設定を明示したclip_000全pipelineの別GPU資格確認。
既定変更とfreezeはユーザー判断、予約未見は未開封のまま。
