---
id: run-ball-temporal-colab-pilot-20261024
type: run
task: ball_detection
sequence: 68
recorded_at: '2026-10-24'
title: Colab L4でDPTとhybridの長系列を上限付き試走
issue: 1376
provider: codex
date: '2026-10-24'
status: done
config:
  gpu: L4
  architectures:
  - dpt
  - hybrid
  frames:
  - 32
  - 64
  - 128
  max_updates_per_condition: 10
  max_condition_seconds: 1200
  max_session_seconds: 14400
  stop_work_seconds: 12600
  input_frames_per_update: 128
  microbatch_windows: 1
  frame_chunk_size: 4
  activation_checkpointing: true
  precision: bf16
  compile_mode: 'off'
  jpeg_decoder: nvjpeg
  seed: 42
  augmentation: none
  checkpoint_interval: 5
  evaluation_interval: 10
  native_image_size:
  - 1280
  - 720
  scope: bounded_diagnostic_performance_only
metrics:
  environment_setup_seconds: 41.85834838000119
  observed_gpu_memory_mib: 23034
  conditions_completed: 6
  optimizer_updates_total: 60
  download_cli_verification_seconds: 3249.166046098999
  independent_sha256_seconds: 454.94380481400003
  peak_allocated_bytes: 13604039680
  peak_reserved_bytes: 16995319808
  owned_gpu_hours: 1.561201102777778
  observed_account_cu_decrease: 2.4
repro:
  commit: b4178691cb509529bd2761dd4bd53d63de994e10
  command: See knowledge/runs/run-ball-temporal-colab-pilot-20261024/receipts/executed-requests.json
    (exact historical managed argv; use a fresh session/time for reproduction)
artifacts:
  run_dir: knowledge/runs/run-ball-temporal-colab-pilot-20261024
  measurements: knowledge/runs/run-ball-temporal-colab-pilot-20261024/receipts/measurements.json
  budget_proposal: knowledge/runs/run-ball-temporal-colab-pilot-20261024/receipts/budget-proposal.json
  stop_evidence: knowledge/runs/run-ball-temporal-colab-pilot-20261024/receipts/session-stop.json
parents:
- run-ball-colab-checkpoint-resume-20261024
- run-ball-all-frame-protocol-20261022
relations: []
papers: []
tags: []
---

## 結果と範囲

DPT/hybrid × T=32/64/128の**6条件すべて**で、native 1280×720の実RGB復号→FP32 MDD→
forward/backward→10 optimizer更新→評価→checkpointのDrive保存・サイズ/hash照合まで完了した。
OOM・条件失敗・未測定条件は0件。各条件は210.6〜233.9秒で、最大10更新・20分を守った。
本学習は実施していない。T上限・学習予算の決定は、下の提案に対する人間の判断待ちである。

全条件は5ch・本番の幅/深さ・1/4格子180×320のheatmap・共通Gaussian/Focal BCEを維持した。
BF16、nvJPEG、compile off、activation checkpointing on、frame_chunk_size=4、microbatch=1窓、
128入力frame/更新（4/2/1窓を蓄積）、augmentation none、loader workers=0、upfront検証を共通にした。
試走recipeはAdamW、LR=1e-4、warmup=2、cosine最小LR比0.1、weight decay=0.01、clip=1、
sigma比0.012、gamma=2、seed42。評価は更新10、checkpointは5/10で公開した。

## 実測表

GBは十進数。stepは復号・MDD・forward/backward・optimizerを含み、評価・保存を含まない。
host待ちはnext batchとimage-input呼出しの壁時計で、非同期GPU復号を完全に分離した時間ではない。
「更新2–10平均」は初回を除いた9点であり、長時間定常性能を保証しない。

| 条件 | 更新 | 初回 step 秒 | 更新2–10平均 秒 | host待ち平均 秒 | peak allocated / reserved GB | 評価 秒 | checkpoint Drive保存平均 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dpt-T32 | 10 | 10.48 | 7.26 | 0.113 | 4.12 / 5.62 | 5.05 | 34.18 |
| hybrid-T32 | 10 | 8.92 | 7.45 | 0.102 | 4.19 / 5.70 | 5.10 | 44.53 |
| dpt-T64 | 10 | 9.14 | 7.77 | 0.100 | 7.21 / 9.31 | 5.17 | 35.16 |
| hybrid-T64 | 10 | 9.28 | 7.83 | 0.100 | 7.33 / 9.44 | 5.20 | 34.58 |
| dpt-T128 | 10 | 9.45 | 8.09 | 0.099 | 13.46 / 16.77 | 4.26 | 30.88 |
| hybrid-T128 | 10 | 9.56 | 8.16 | 0.099 | 13.60 / 17.00 | 4.27 | 39.41 |

保存欄は完成checkpoint直後の同期publishの平均（更新5/10）。ローカルのweights/checkpoint書出しと
hash計算は、条件ごとの合計2.54〜4.28秒を別に観測した。初期化と完了markerにも別のpublishがある。
Store内部のhash照合はpublish時間に含める。
各更新の全数値・local保存内訳・評価のpeak・全publication receiptは
[measurements.json](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/measurements.json)と
[条件別receipt](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/conditions/matrix.json)を正本とする。
各条件で10更新分のpack ID/抽出位置・入力frame数・累積frame数・kind別教師数・LRが一致した。
解像度・幅・Tを片側だけ下げた条件、補間値、追加の試走条件はない。

## 入力・準備・保存・停止

採用済みtrain/validation配布版の157.029 GB・2,926ファイルを新VMへ全量取得した。
取得＋CLI照合54.15分、追加の独立SHA-256（4 workers）7.58分、18窓のRGB/教師/実PTS照合18.23秒、
データjob本体は合計62.04分。全件内容照合と18窓の意味一致を確認した。
CLI照合の途中では、local treeのSHA-256/SHA-1/MD5を計算するrcloneが18分超・CPU約110%で
動作していた。この時間は取得＋CLI照合に含む（独立した追加計測値として二重加算しない）。

先行runのinput generationを元の固定pathへ復元し、markerと内容hash、取得済みindex/metadataの
identityを照合した。保存済みfull T32 manifestの内容を保持し、新しい診断T32/64/128 manifestは
旧診断集合とcommon/target/teacher/scored/pack identityが一致した。復元・診断manifest準備・
小さな設定のpublishは関数内54.8秒で、full manifestを再生成した値ではない。

全量取得と、性能試走のデータ範囲は区別する。試走は先行の独立診断4clip（trainはTrackNet/Meiji/Chat、
valはChat）で、train共通589 frame、val共通162 frame（observed 33、不在129）。全Tで同じ集合を使う。
valの処理入力はT32=320、T64=320、T128=256 frameであり、全体の128,668 unique val frameを
評価した時間ではない。full validation・testの実時間、cold random I/O、収束・精度差は未測定。
T64/128は各モデルの10更新T32 **pilot重み**から独立に開始した。これは本学習の親重みではない。

一台だけ確保したGPUはNVIDIA L4。nvidia-smiは23,034 MiB、Torchのtotal_memoryは
23,659,151,360 bytesを返した（APIの値を区別して記録）。CPython 3.11.16・uv 0.12.21、
Torch 2.14.1+cu130、CUDA 13.0、cuDNN 92400、driver 580.82.07を記録した。
環境構築41.86秒。空き容量は導入前206.410 GB→導入後199.609 GB→データ取得後42.467 GB→
最終40.665 GB。このVMの観測値であり、将来VMの容量・VRAM・速度・rateの保証ではない。

24 run generation＋1 input generationのmarker ID/MD5・file数・合計sizeが、最終Drive inventoryと一致した。
公開時は独立SHA-256とremote MD5/sizeを照合しており、VM内だけのcheckpointを成功に数えていない。
同じsnapshot rootへの周期mirrorは使わず、workerは別のreceipt treeを保存した。
全5 managed jobのreturncode=0・Drive保存完了を確認後、自分のsessionを停止した。
停止receiptはtracked差分0 byte、untracked source 0件で、除外1,542件はDrive管理の入力storageだった。
大きい停止receiptは正規化してDriveのoperationsへ保存し、Gitには小さな照合要約を残した。

確保2026-10-24 01:02:43 UTC→停止receipt 02:36:23 UTCは5,620.3秒（1.5612 GPU割当時間）。
assignment 0確認まで5,659.3秒で、3時間30分の打切り・4時間の全体上限より前に終了した。
開始前/停止後assignmentは0、確保後rateは今回のdoctorで1.54 CU/h、停止後rateは0。
残CUは340.54→338.14（account差分2.40）。観測rateが全期間一定なら2.404 CUの推計となり、
個別請求の確定値とは区別する。保存・停止の証拠は
[session-stop.json](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/session-stop.json)、
[compute-units.json](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/compute-units.json)、
[doctor-stopped.json](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/doctor-stopped.json)を参照。

## 本学習予算の判断依頼（未承認）

T上限128、F=128、microbatch=1、蓄積4/2/1、今回と同じruntime/幅を推奨する。
10更新では収束や最適LRを決められないため、次は**固定予算の比較**として提案する。
seed42は両モデル32事前学習2本＋T32/64/128への独立FT6本。追加seedの対象は、結果を見て
選び直さず両モデルの32対128へ43/44を固定する。

| 案 | Upre / Upost | 追加seed 43/44 | 予備25%込みGPU割当時間 / CU |
|---|---:|---|---:|
| A（推奨） | 6144 / 1536 | seed42の親を共有し、両モデルのT=32/128を独立FT | 172.9 h / 266.3 CU |
| B | 3072 / 1024 | 各seedで32事前学習を再実行し、両モデルのT=32/128をFT | 180.1 h / 277.3 CU |

Aは1つの親重み当たりの学習量を確保する案として推奨する。追加seedが評価するのはFTの変動に限る。
初期化からの変動も含めたい場合はBを選ぶ。両案とも追加の予算・データ購入は行っていない。
共通の提案LRは1e-4、linear warmup→cosine、最小比0.1。warmupはAがpre256/post64、
Bがpre128/post64。checkpoint間隔256、valはAがpre2048/post1536、Bがpre1536/post1024、
固定予算末尾を主評価とする。その他の提案値と各run一覧は
[budget-proposal.json](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/budget-proposal.json)にある。

見積もりは実測更新時間と、小集合の評価時間×full入力frame比からの外挿で、実測表へは混ぜていない。
全量validation/testのcalibration・予測書出し・cold I/Oは小集合と異なり、25%は保証された上限ではない。
8時間ごとのVM区切り（寿命の保証ではない）、毎VMの実測に基づく再取得/復元、入力検証の余裕、
初回manifest準備1.5時間、test専用VM準備0.35時間、保存時間2倍の余裕を仮定した。
初回test uploadと既存評価契約の修正は未測定のカレンダー時間としてさらに加わる。
提案する共通上限は**200 GPU割当時間・310 CUの両方**。残338.14 CUに対して余裕を残し、
到達時は保存・停止して親へ戻す。利用rateやVM寿命が変わっても上限を自動延長しない。
未収束延長の実験はこの見積もりに含めず、別の固定予算の合意を要する。本学習は開始していない。

## 先行範囲の統合課題（未修正）

本学習前に親へ返す事項として、別test配布版と現在の評価契約が接続できない点がある。
`evaluation/temporal_heatmap_cli.py:80–85`は学習snapshot・全RGB clip集合・全split identityの
完全一致を要求し、独立したtest配布版を拒否する。さらに`evaluation/temporal_heatmap.py:137–147`は、
test manifest内のval集合と較正済みval集合を照合するため、test専用版の空valとは一致しない。
今回のfull train/val manifestのtestは0 frame、元test共通集合は225区間・143,074 frameである。
Driveの予定test版pathも今回のinspectでは不存在だった。

入力版間の明示的なlineageとvalidation較正の来歴を整え、test版を配置する必要がある。
同run再開にはsource identityが入るので、コード修正は本学習開始前に済ませるのを推奨する。
試走のval経路には影響せず、先行コード・checkpoint metadata・guardは変更していない。
根拠は[upstream-test-contract.json](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/upstream-test-contract.json)と
[test asset inspect](../../runs/run-ball-temporal-colab-pilot-20261024/receipts/test-asset-inspection.json)に残した。

## 検証と次の作業

CPU controller/集計テスト9件、ruff、mypyを通過した。実行したコマンドはrunの検証receiptへ残す。
本番srcは変更しておらず、試走専用controllerと計測/予算集計はknowledge内に閉じている。
本学習の採用・予算固定・追加seedの解釈は親と人間の判断に委ねる。全量val/testの時間を初回に
確認し、hard capを維持する。採用モデル・deployの判断はこの性能試走では更新しない。
