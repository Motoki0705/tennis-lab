---
id: run-i936-provisional-degradation-r5-s936
type: run
task: ball_refiner_3d
sequence: 8
recorded_at: '2026-09-30'
title: 保存済み文脈なしpilotの全4成分を用いた暫定合成劣化
provider: codex
status: done
config:
  source_partition: calibration
  components: 4
  block_frames: 16
  out_of_frame_presence_logit: -4.0
metrics:
  calibration_rows: 14652
  observed_frames: 12068
  evidence_gap_frames: 2584
  confirmed_negative_frames: 0
artifacts:
  run_dir: knowledge/runs/run-i936-provisional-degradation-r5-s936
parents:
- run-i935-calibration-hdr-ft-e13-r5-20260928
- run-i936-synthetic-smoke-r3-s936
relations: []
papers: []
tags: []
issue: 936
date: '2026-09-30'
repro:
  command: PYTHONPATH=. .venv/bin/python knowledge/runs/run-i936-provisional-degradation-r5-s936/reproduce.py
    --source "$PWD/knowledge/runs/run-i935-calibration-hdr-ft-e13-r5-20260928" --output
    /absolute/new/calibration
---

#935の保存済み文脈なしpilotを、96ラリー開発用の**暫定劣化**へ接続した。
新detector epoch/person contextの最終較正ではなく、full 640ラリーは待機する。
実ボールの3D軌道評価、GPU再推論、学習は行っていない。TensorBoardは対象外。

## 出典と抽出

[#959の較正側validation](../ball_refiner/000004-run-i935-calibration-hdr-ft-e13-r5-20260928.md)に保存された
6時刻clip群×3camera×観測/人工証拠欠損=36 NPZを再利用した。
全てvideo_000の較正側で、testやcheckpoint選択側を混ぜない。採点済み正例を全て保持した。
checkpoint epoch009 SHAは `ed659e142dd228949ec772bcfdf9db0f4c682081778853c15088f7202a848c76`、
入力manifest SHAは `254dbd4d36025b2bc136b8869d6cf612d81be998cd5500259218840a0fc6b27a`。
全NPZのSHA・frame件数・source/checkpoint情報・統計の正本は
[calibration.json](../../runs/run-i936-provisional-degradation-r5-s936/calibration.json)。
bank SHAは `aea209600cfe218e239612cfaa828706b7ffabedebc9291ea0634acce31c482c`。

保存GMMは旧合成K=3と異なるK=4だった。成分を削らず、全4成分のuv残差・Cholesky因子・
混合logit・amodal存在logitを1行として保存した。そのため新datasetの3Dは全125成分となる。
cameraごと・観測/欠損ごとにblock再標本化し、全成分の残差を合成投影へ移す。
blockは最大16frameで、元frameが連続する採点区間だけを辿り、clip/camera/条件/未採点穴を跨がない。
先頭rowはそのstratumの全行から一様抽出する。重み・共分散を平均せず、採用rowを全frameへ保存する。

## 観測統計

誤差/共分散比は `r=(mixture mean−GT)^T mixture covariance^-1 (mixture mean−GT)/2`。
混合covarianceは成分内・成分間を含む。Gaussianの較正度を厳密に表す指標ではなく、
二次momentに対する誤差の診断である。bankはこの過信/裾誤差も保持し、r=1へ補正しない。

| camera/条件 | frame | r平均 | r p95 | 平均存在確率 | 降順の平均成分重み |
|---|---:|---:|---:|---:|---|
| cam0/観測 | 3952 | 2.096 | 9.510 | .99089 | .599/.223/.125/.053 |
| cam0/欠損 | 876 | 1.664 | 7.811 | .98911 | .574/.193/.136/.097 |
| cam1/観測 | 4171 | 2.817 | 13.825 | .99174 | .729/.150/.085/.036 |
| cam1/欠損 | 873 | 1.938 | 8.345 | .98942 | .522/.220/.165/.093 |
| cam2/観測 | 3945 | 1.718 | 7.095 | .99050 | .687/.173/.102/.038 |
| cam2/欠損 | 835 | 1.072 | 4.404 | .98962 | .566/.197/.142/.095 |

存在BCE/Brierと確率bin別正例率もJSONへ保存した。ただし確定負例0件であり、
不存在を含む存在較正の正当性は評価できない。画面外は明示的仮定logit −4を用いる。
遮蔽を不存在へ変換せず、画面内ではその条件のpilotが出した存在logitを使う。

## 限界と次の判断

16frameより長いgapの出力は保存されていないため、32/64frame gapは短いgap blockの連結で外挿する。
camera間は独立に抽出するので同時誤検出の相関は未較正。元uv残差の移動とhead範囲へのclipで
合成側の統計が変わり得る（clip件数は保存）。camera番号は対応する設置位置のproxyであり、
録画間の分布変化は検証していない。これらはdevelopment用の仮定で、実refiner出力の完全な再現ではない。

次は収束判定つきの96ラリーを生成し、metadataを読んでdiffusion開発を進める。
full生成前に新detector/person contextの保存出力でbankを作り直し、負例と長いgap・相関を再検討する。
[採用した暫定判断](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5898321426)。
