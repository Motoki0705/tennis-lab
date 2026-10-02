---
id: run-i935-context-fullframe-pilot-r9-20260928
type: run
task: ball_refiner
sequence: 6
recorded_at: '2026-09-28'
title: 3source文脈pilotが古いDINO拡張で停止し、CPU dispatchと再ビルドで原因を診断
issue: 935
provider: codex
session: 01a0e82d-a490-71b3-b824-86a92b90919a
date: '2026-09-28'
status: failed
config:
  sources:
  - tracknet
  - meiji
  - chat_annotation
  split: train
  planned_clips: 3
  planned_frames: 561
  max_tracks: 64
  person_region_policy: full_frame
  torch_at_diagnosis: 2.13.0+cu130
metrics:
  completed_clips: 0
  queue_exit_code: 1
repro:
  commit: f65c91f5c8a1a0e60025237ef424fc90afd7bcfa
  branch: campaign930/i935-8-context-cache
  remote: git@github.com:Motoki0705/tennis-lab.git
  captured_command: knowledge/runs/run-i935-context-fullframe-pilot-r9-20260928/run.json
artifacts:
  run_dir: knowledge/runs/run-i935-context-fullframe-pilot-r9-20260928
  log: knowledge/runs/run-i935-context-fullframe-pilot-r9-20260928/queue.log
  captured_repro: knowledge/runs/run-i935-context-fullframe-pilot-r9-20260928/captured-repro.sh
parents:
- run-i935-evidence-ft-e13-trainval-r3-20260928
relations: []
papers: []
tags:
- context
- failed
- dino
- extension-compatibility
- cpu-diagnosis
---

## 結果

全source文脈生成の最初の3clip pilotは、最初のDINO forwardで停止した。
[queue log](../../runs/run-i935-context-fullframe-pilot-r9-20260928/queue.log)の終端は
`RuntimeError: Undefined backend is not a valid device type`、exit code 1。
[失敗時manifest](../../runs/run-i935-context-fullframe-pilot-r9-20260928/failed-context-manifest.json)は
`status=building`、完了clipは0。未生成を欠損観測へ変換しておらず、読込不能のまま保持している。
pose/courtの品質・生成速度・文脈ablationの指標は得られていない。学習ではないためTensorBoard/learning curveもない。

対象はtrainの各sourceで窓33以上の最短clipをラベル非依存で選んだ、TrackNet 35・Meiji 151・chat 375frame。
[選択一覧](../../runs/run-i935-context-fullframe-pilot-r9-20260928/pilot-selection.json)、
[scene設定](../../runs/run-i935-context-fullframe-pilot-r9-20260928/scene-context.yaml)、
[資産preflight](../../runs/run-i935-context-fullframe-pilot-r9-20260928/preflight-final.json)、
[JPEG監査](../../runs/run-i935-context-fullframe-pilot-r9-20260928/jpeg-preflight.json)を保存した。
全561 JPEGをCPUでdecodeしてdetector cacheのshard hashと一致した事前監査は、モデル実行の成功とは別の証拠である。

## 原因の診断と修正方針

run 9はmain checkoutに残っていた2026-08-20の共有拡張をPYTHONPATHで指定した。
そのバイナリはimportできたが、現在のtorch `2.13.0+cu130`でtensor/backend dispatchが壊れる。
対応する古いstaged sourceには`.type().is_cuda()`が残り、今回のcheckoutのbuild.pyに既にある
`.is_cuda()`への互換修正が反映されていなかった。古いビルド環境の完全な記録はないため、
旧APIとビルド時ABIのどちらだけで障害を説明できるかまでは切り分けていない。

run 10でGPUを不可視にした別Pythonプロセスから、同じ小さなCPUテンソルを両拡張のforward/backwardへ渡した。
[古い拡張](../../runs/run-i935-context-fullframe-pilot-r9-20260928/legacy-cpu-dispatch.json)は
`Unrecognized tensor type ID: PythonTLSSnapshot`で両方停止。
これ以前の同形状probeでは`Undefined`となり、エラー文字列自体は安定しなかった。
[run専用再ビルド](../../runs/run-i935-context-fullframe-pilot-r9-20260928/rebuilt-cpu-dispatch.json)は
両方とも上流の意図した`Not implemented on the CPU`へ到達した。両processともCUDAは未初期化。
CPU非対応のままで正しくdispatchできる確認であり、CPU推論への切替ではない。
[probe script](../../runs/run-i935-context-fullframe-pilot-r9-20260928/probe_dino_dispatch.py)と各binary hashを保存した。

生成のmodel identity/preflightにこのCPU検査を追加する。想定外の例外・CPU成功・欠けたentry pointは停止し、
自動再ビルドや別binaryへの切替は行わない。GPU kernelの正しさは再投入したpilotで別途確認する。
今後のqueue入口はrunごとに追跡済みbuild scriptから拡張を作り、生成したlibだけを明示指定する。

もう一つ、旧jobの未到達の`python -c`検証コードは多重shell引用を展開して`ast.parse`すると
26行目のunterminated string literalで失敗する。これは今回観測したGPU障害の原因ではない。
再投入版では追跡済みの別プロセス検証moduleを呼び、fake-model cacheの実CLIテストで検証する。

## 再現性の限界

自動取得されたrun.json、patch、git status、元のshellは内容を変えず保存した。
元shellは[captured-repro.sh](../../runs/run-i935-context-fullframe-pilot-r9-20260928/captured-repro.sh)として**履歴資料**にする。
共有の古いbinaryに依存し、既存出力の再使用・未到達検証の構文エラーもあるため、実行可能な`repro.sh`とは称さない。
旧binaryはgitへ複製していない。保存hashと一致するbinaryが残る環境なら、上記probeを
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python <probe> --extension-directory <lib>`で再実行できる。
新しいbuildを使う回復実行は、この失敗と同一runにはせず、新しいqueue job・出力先・証跡を持つ。

## 次の実験と現在の判断

同じ3clip/561frame・モデル設定・全画面方針で再投入し、別プロセスで全NPZ/frame/PTS/JPEGを検証する。
成功後に対象人物と観客・隣接court、pose score飽和・欠落、court品質を画像上で確認し、全329clipの生成単位を決める。
この失敗はデータや文脈仮説の精度評価ではなく、文脈の採否やdetector deployの判断を変えない。
学習設計のユーザー合意、同一母数の文脈ablation、RGB遮蔽対照、較正fitと最終Meiji testは未完了。
